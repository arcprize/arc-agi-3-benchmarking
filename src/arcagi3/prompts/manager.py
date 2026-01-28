from __future__ import annotations

import inspect
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, Optional, Union

from jinja2 import Environment, StrictUndefined, Template, UndefinedError

PromptVars = Dict[str, Any]
PromptCallable = Callable[[PromptVars], str]
PromptSource = Union[str, PromptCallable]


class PromptManager:
    """
    Minimal prompt loader/renderer with jinja2 templating support.

    **No built-in prompt registry**: prompts are discovered relative to the *caller*.

    If a file at `/foo/bar/file.py` does:
      `PromptManager().load("myprompt")`
    this will load, in order:
      - `/foo/bar/prompts/myprompt.prompt`
      - `/foo/bar/prompts/myprompt`

    Templates support jinja2 syntax including conditionals, loops, and filters.
    """

    __lock = Lock()
    __cache: Dict[Path, str] = {}
    __template_cache: Dict[Path, Template] = {}

    def __init__(self):
        """Initialize jinja2 environment."""
        self._jinja_env = Environment(
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
            undefined=StrictUndefined,
        )

    def load(self, name: str) -> str:
        """
        Loads a report prompt from a file in the "prompts" directory relative
        to the caller's file and caches it.
        """
        # Get the caller's frame info - skip PromptManager frames
        # in case this is the call in .render()
        stack = inspect.stack()
        caller_frame = None
        for frame_info in stack[1:]:  # Skip current frame (load)
            frame = frame_info.frame
            # Check if this frame is not in PromptManager
            module_name = frame.f_globals.get("__name__", "")
            if module_name != "arcagi3.prompts.manager" or frame_info.function not in (
                "load",
                "render",
            ):
                caller_frame = frame_info
                break

        if caller_frame is None:
            # Fallback to stack[1] if we couldn't find a non-PromptManager frame
            caller_frame = stack[1]

        caller_filepath = caller_frame.filename
        caller_directory = Path(caller_filepath).parent

        # Construct the file path relative to the caller's file
        filepath = caller_directory / "prompts" / f"{name}"

        with self.__lock:
            # Try .prompt extension first, then no extension
            candidates = [filepath.with_suffix(".prompt"), filepath]
            for candidate in candidates:
                if candidate in self.__cache:
                    return self.__cache[candidate]
                if candidate.exists():
                    text = candidate.read_text(encoding="utf-8")
                    self.__cache[candidate] = text
                    return text
            raise FileNotFoundError(
                f"Prompt '{name}' not found. Tried: {[str(p) for p in candidates]}"
            )

    def render(self, name: str, vars: Optional[PromptVars] = None) -> str:
        """
        Renders the named prompt with the given variables using jinja2 templating.
        Loads and caches the prompt if it is not already cached.

        Templates support full jinja2 syntax including:
        - Variables: {{ var }}
        - Conditionals: {% if condition %}...{% endif %}
        - Loops: {% for item in items %}...{% endfor %}
        - Filters: {{ var|upper }}

        If the template references variables not passed in, a ValueError will be raised.
        Extra variables that are not used in the template are allowed (useful for conditionals).
        """
        template_text = self.load(name)

        # Get the file path for template caching
        stack = inspect.stack()
        caller_frame = None
        for frame_info in stack[1:]:  # Skip current frame (render)
            frame = frame_info.frame
            module_name = frame.f_globals.get("__name__", "")
            if module_name != "arcagi3.prompts.manager" or frame_info.function not in (
                "load",
                "render",
            ):
                caller_frame = frame_info
                break

        if caller_frame is None:
            caller_frame = stack[1]

        caller_filepath = caller_frame.filename
        caller_directory = Path(caller_filepath).parent
        filepath = caller_directory / "prompts" / f"{name}"

        # Find the actual file path (with or without .prompt extension)
        with self.__lock:
            candidates = [filepath.with_suffix(".prompt"), filepath]
            template_path = None
            for candidate in candidates:
                if candidate.exists():
                    template_path = candidate
                    break

            if template_path is None:
                raise FileNotFoundError(
                    f"Prompt '{name}' not found. Tried: {[str(p) for p in candidates]}"
                )

            # Get or create jinja2 template
            if template_path not in self.__template_cache:
                self.__template_cache[template_path] = self._jinja_env.from_string(template_text)

            template = self.__template_cache[template_path]

        if not vars:
            vars = {}

        # Render with jinja2 - this will raise UndefinedError if a required variable is missing
        try:
            return template.render(**vars)
        except UndefinedError as e:
            # Extract variable name from jinja2 error message
            error_msg = str(e)
            # Error format: "'variable_name' is undefined"
            if "'" in error_msg:
                var_name = error_msg.split("'")[1]
                raise ValueError(f"Missing variable(s) for template: {var_name}")
            raise ValueError(f"Template rendering error: {error_msg}")
