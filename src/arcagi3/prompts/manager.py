import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Set, Union

from .names import PromptName


PromptVars = Dict[str, Any]
PromptCallable = Callable[[PromptVars], str]
PromptSource = Union[str, PromptCallable]


_FILENAME_BY_PROMPT: Dict[PromptName, str] = {
    PromptName.SYSTEM: "system.prompt",
    PromptName.ACTION_INSTRUCT: "action_instruct.prompt",
    PromptName.ANALYZE_INSTRUCT: "analyze_instruct.prompt",
    PromptName.FIND_ACTION_INSTRUCT: "find_action_instruct.prompt",
    PromptName.COMPRESS_MEMORY: "compress_memory.prompt",
}


_ALLOWED_PLACEHOLDERS: Dict[PromptName, Set[str]] = {
    PromptName.SYSTEM: set(),
    PromptName.ACTION_INSTRUCT: {
        "available_actions_list",
        "example_actions",
        "json_example_action",
    },
    PromptName.ANALYZE_INSTRUCT: {"memory_limit"},
    PromptName.FIND_ACTION_INSTRUCT: {"action_list", "valid_actions"},
    PromptName.COMPRESS_MEMORY: {"current_word_count", "memory_limit", "memory_text"},
}


_PLACEHOLDER_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


def _load_default_prompt_text(name: PromptName) -> str:
    """Load the default prompt text for a given prompt name via importlib.resources."""
    filename = _FILENAME_BY_PROMPT[name]
    try:
        package_files = resources.files(__package__)
        path = package_files / filename
        return path.read_text(encoding="utf-8")
    except AttributeError:
        return resources.read_text(__package__, filename, encoding="utf-8")


def _extract_placeholders(template: str) -> Set[str]:
    """
    Extract placeholder names of the form {{name}} or {{ name }}.

    Single-brace patterns (e.g. {x}) are ignored to avoid accidental matches
    with f-string or format-style placeholders.
    """
    return {match.group(1) for match in _PLACEHOLDER_RE.finditer(template or "")}


def _validate_placeholders(name: PromptName, template: str) -> Set[str]:
    """Validate that all placeholders in template are allowed for this prompt."""
    used = _extract_placeholders(template)
    allowed = _ALLOWED_PLACEHOLDERS[name]
    unknown = used - allowed
    if unknown:
        raise ValueError(
            f"Prompt '{name.value}' contains unknown placeholders: {sorted(unknown)}. "
            f"Allowed placeholders: {sorted(allowed)}"
        )
    return used


def _substitute_placeholders(template: str, vars: Mapping[str, Any]) -> str:
    """Substitute {{name}} placeholders using the provided vars mapping."""

    def repl(match: re.Match) -> str:  # type: ignore[name-defined]
        key = match.group(1)
        if key not in vars:
            raise ValueError(
                f"Missing value for placeholder '{key}' when rendering prompt."
            )
        value = vars[key]
        return str(value)

    return _PLACEHOLDER_RE.sub(repl, template)


@dataclass
class _LoadedPrompt:
    text: str
    placeholders: Set[str]


class PromptManager:
    """
    Manage agent prompts, including default templates and optional overrides.

    Overrides can be provided as:
      - A string path pointing to a file containing the template text.
      - A literal template string (when the path does not exist).
      - A callable taking a mapping of variables and returning the final prompt text.
    """

    def __init__(self, overrides: Optional[Mapping[Union[str, PromptName], PromptSource]] = None):
        self._defaults: Dict[PromptName, _LoadedPrompt] = {}
        self._overrides: Dict[PromptName, PromptSource] = {}

        # Pre-load defaults and validate their placeholders.
        for name in PromptName:
            text = _load_default_prompt_text(name)
            placeholders = _validate_placeholders(name, text)
            self._defaults[name] = _LoadedPrompt(text=text, placeholders=placeholders)

        # Normalise and store overrides (validation happens when used).
        if overrides:
            for key, source in overrides.items():
                prompt_name = self._normalise_name(key)
                self._overrides[prompt_name] = source

    @staticmethod
    def _normalise_name(name: Union[str, PromptName]) -> PromptName:
        if isinstance(name, PromptName):
            return name
        try:
            return PromptName(name)
        except ValueError:
            raise KeyError(f"Unknown prompt name: {name!r}. Valid names: {[n.value for n in PromptName]}")

    def get_template(self, name: Union[str, PromptName]) -> str:
        """Return the raw (unrendered) template text for the given prompt."""
        pname = self._normalise_name(name)
        source = self._overrides.get(pname)

        if source is None:
            return self._defaults[pname].text

        if callable(source):
            raise RuntimeError(
                "Callable prompt overrides must be rendered via 'render', not 'get_template'."
            )

        path = Path(source)
        if path.exists():
            text = path.read_text(encoding="utf-8")
        else:
            text = source

        _validate_placeholders(pname, text)
        return text

    def render(self, name: Union[str, PromptName], vars: Optional[PromptVars] = None) -> str:
        """
        Render the given prompt with the provided variables.

        For string-based templates, {{name}} placeholders are substituted from vars.
        For callable overrides, vars are passed directly and the returned string
        is then validated (but not further substituted).
        """
        pname = self._normalise_name(name)
        source = self._overrides.get(pname)

        if source is None:
            template = self._defaults[pname].text
            placeholders = self._defaults[pname].placeholders
            if not placeholders:
                return template
            if vars is None:
                raise ValueError(
                    f"Prompt '{pname.value}' requires variables {sorted(placeholders)} but none were provided."
                )
            missing = placeholders - set(vars.keys())
            if missing:
                raise ValueError(
                    f"Missing values for placeholders {sorted(missing)} when rendering prompt '{pname.value}'."
                )
            return _substitute_placeholders(template, vars)

        if callable(source):
            provided_vars: PromptVars = dict(vars or {})
            text = source(provided_vars)
            if not isinstance(text, str):
                raise TypeError(
                    f"Callable override for prompt '{pname.value}' must return a string, "
                    f"got {type(text).__name__}"
                )
            _validate_placeholders(pname, text)
            return text

        path = Path(source)
        if path.exists():
            template = path.read_text(encoding="utf-8")
        else:
            template = source

        placeholders = _validate_placeholders(pname, template)
        if not placeholders:
            return template

        if vars is None:
            raise ValueError(
                f"Prompt '{pname.value}' requires variables {sorted(placeholders)} but none were provided."
            )

        missing = placeholders - set(vars.keys())
        if missing:
            raise ValueError(
                f"Missing values for placeholders {sorted(missing)} when rendering prompt '{pname.value}'."
            )

        return _substitute_placeholders(template, vars)


__all__ = ["PromptManager", "PromptSource", "PromptVars"]


