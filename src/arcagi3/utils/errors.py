from __future__ import annotations

import re
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


def _classify_exception(exc: Exception) -> Tuple[str, str, str, List[str]]:
    message = str(exc) or exc.__class__.__name__
    message_lower = message.lower()
    env_match = re.search(r"\b[A-Z][A-Z0-9_]*_API_KEY\b", message)
    env_key = env_match.group(0) if env_match else "API key"

    if isinstance(exc, FileNotFoundError):
        return (
            "MissingFile",
            "Required file not found.",
            "A file or path referenced by the run could not be located.",
            [
                "Verify the path exists and is readable.",
                "Check for typos in file names or arguments.",
            ],
        )

    if isinstance(exc, TimeoutError):
        return (
            "Timeout",
            "Operation timed out.",
            "A network or API call did not complete within the expected time.",
            ["Retry the run.", "Check your network connection or provider status."],
        )

    if isinstance(exc, ConnectionError):
        return (
            "NetworkError",
            "Network connection failed.",
            "The runner could not reach a required service.",
            [
                "Check network access and firewall settings.",
                "Retry after confirming the service is reachable.",
            ],
        )

    if ("api key" in message_lower or "api_key" in message_lower) and (
        "missing" in message_lower or "not set" in message_lower
    ):
        return (
            "MissingApiKey",
            "API key is missing.",
            "The required API key environment variable is not set.",
            [
                f"Set `{env_key}` in your environment or `.env` file.",
                "Run `--check` to verify credentials.",
            ],
        )

    if (
        "unauthorized" in message_lower
        or "401" in message_lower
        or "invalid api key" in message_lower
    ):
        return (
            "InvalidApiKey",
            "API key appears invalid.",
            "The configured API key was rejected by the provider.",
            ["Verify the key value and permissions.", "Re-run `--check` after updating the key."],
        )

    if "checkpoint" in message_lower and (
        "not found" in message_lower or "missing" in message_lower
    ):
        return (
            "CheckpointMissing",
            "Checkpoint not found.",
            "The requested checkpoint could not be located on disk.",
            [
                "Use `--list-checkpoints` to see available IDs.",
                "Confirm the checkpoint directory exists.",
            ],
        )

    if isinstance(exc, ValueError):
        return (
            "InvalidInput",
            "Invalid input or configuration.",
            "A provided argument or configuration value is not valid for this run.",
            [
                "Double-check CLI args and config names.",
                "Use `--list-games` or `--list-agents` if unsure.",
            ],
        )

    return (
        "RuntimeError",
        "Unexpected runtime error.",
        "An unexpected error occurred while running the benchmark.",
        ["Review the stack trace for details.", "Retry with `--verbose` for more logs."],
    )


def build_error_payload(
    exc: Exception,
    *,
    context: Optional[Dict[str, Any]] = None,
    trace: Optional[str] = None,
) -> Dict[str, Any]:
    error_type, summary, explanation, suggested_fixes = _classify_exception(exc)
    payload = {
        "error_type": error_type,
        "exception_type": exc.__class__.__name__,
        "message": str(exc),
        "summary": summary,
        "explanation": explanation,
        "suggested_fixes": suggested_fixes,
        "context": context or {},
        "traceback": trace or traceback.format_exc(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return payload


def format_user_message(payload: Dict[str, Any]) -> str:
    lines = [
        f"Error: {payload.get('summary')}",
        payload.get("explanation", ""),
    ]
    fixes = payload.get("suggested_fixes") or []
    if fixes:
        lines.append("Suggested fixes:")
        lines.extend([f"- {fix}" for fix in fixes])
    return "\n".join(line for line in lines if line)
