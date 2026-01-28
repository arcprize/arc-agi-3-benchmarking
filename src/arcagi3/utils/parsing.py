"""
Utility functions for parsing LLM responses.
"""
import json
import logging
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)


def extract_json_from_response(response_text: str) -> Dict[str, Any]:
    """
    Extract JSON from various possible formats in the response.

    Handles:
    - Bare JSON { ... }
    - Fenced JSON ```json ... ```
    - Generic fence ``` ... ```
    - Wrapper text

    Args:
        response_text: The raw text response from the LLM

    Returns:
        Parsed JSON dictionary

    Raises:
        ValueError: If JSON cannot be extracted or parsed
    """
    if not response_text or not response_text.strip():
        raise ValueError("Empty response text")

    # Try fenced ```json ... ``` blocks (with better regex for multiline)
    fence = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.S | re.I | re.M)
    if fence:
        json_str = fence.group(1).strip()
    else:
        # Try any ``` ... ``` fence
        fence = re.search(r"```[a-z]*\s*(\{.*?\})\s*```", response_text, re.S | re.M)
        if fence:
            json_str = fence.group(1).strip()
        else:
            # Fallback: find the first '{' and match balanced braces
            start = response_text.find("{")
            if start == -1:
                raise ValueError(
                    f"No JSON object detected in response. Response was: {response_text[:200]}"
                )

            # Find matching closing brace, skipping strings to avoid false matches
            brace_count = 0
            end = start
            in_string = False
            escape_next = False

            for i in range(start, len(response_text)):
                char = response_text[i]

                if escape_next:
                    escape_next = False
                    continue

                if char == "\\":
                    escape_next = True
                    continue

                if char == '"' and not in_string:
                    in_string = True
                elif char == '"' and in_string:
                    in_string = False
                elif not in_string:
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            end = i
                            break

            if brace_count != 0:
                # If we couldn't find balanced braces, the JSON might be truncated
                # Try to get what we have and let json.loads fail with a better error
                logger.warning(
                    f"Unbalanced braces in JSON (count: {brace_count}). JSON might be truncated."
                )
                json_str = response_text[start:].strip()
                # Try to close the JSON
                json_str = json_str.rstrip() + "}"
            else:
                json_str = response_text[start : end + 1].strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Clean control characters and try again
        try:
            import unicodedata

            cleaned = "".join(
                char if unicodedata.category(char)[0] != "C" or char in "\n\r\t" else " "
                for char in json_str
            )
            return json.loads(cleaned)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse JSON: {e}. JSON string was: {json_str[:200]}")
