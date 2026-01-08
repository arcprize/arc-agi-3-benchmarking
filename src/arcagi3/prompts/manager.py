from __future__ import annotations

import inspect
import re
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, Mapping, Optional, Set, Union


PromptVars = Dict[str, Any]
PromptCallable = Callable[[PromptVars], str]
PromptSource = Union[str, PromptCallable]


class PromptManager:
    """
    Minimal prompt loader/renderer.

    **No built-in prompt registry**: prompts are discovered relative to the *caller*.

    If a file at `/foo/bar/file.py` does:
      `PromptManager().load("myprompt")`
    this will load, in order:
      - `/foo/bar/prompts/myprompt.prompt`
      - `/foo/bar/prompts/myprompt`
    """

    # {{ var }} placeholder syntax
    __re = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")

    __lock = Lock()
    __cache: Dict[Path, str] = {}

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
            module_name = frame.f_globals.get('__name__', '')
            if module_name != 'arcagi3.prompts.manager' or frame_info.function not in ('load', 'render'):
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
        Renders the named prompt with the given variables. Loads and caches
        the prompt if it is not already cached.

        If the template has variables not passed in, or you pass in variables
        not within the prompt, a ValueError will be raised.
        """
        template_text = self.load(name)

        if not vars:
            vars = {}

        # Find all template variables in the template (as a set)
        template_vars = set(m.group(1) for m in self.__re.finditer(template_text))
        passed_vars = set(vars.keys())

        # Check: any template var missing in vars?
        missing_vars = template_vars - passed_vars
        if missing_vars:
            raise ValueError(f"Missing variable(s) for template: {', '.join(sorted(missing_vars))}")

        # Check: any passed var not in template?
        extra_vars = passed_vars - template_vars
        if extra_vars:
            raise ValueError(f"Extra variable(s) supplied that do not exist in template: {', '.join(sorted(extra_vars))}")

        def replacer(match):
            var_name = match.group(1)
            return str(vars[var_name])

        return self.__re.sub(replacer, template_text)