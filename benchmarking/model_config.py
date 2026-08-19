from pathlib import Path
from typing import Any

import yaml

MODEL_CONFIG_PATH = Path(__file__).resolve().parent / "model_configs.yaml"
REQUIRED_CONFIG_SECTIONS = ("runtime", "client", "request")
SUPPORTED_RUNTIME_PAIRS = frozenset(
    {
        ("anthropic-python", "messages"),
        ("google-genai", "generate_content"),
        ("openai-python", "chat_completions"),
        ("openai-python", "responses"),
    }
)
DEFAULT_RUNTIME_STATE = "manual_rolling"
SERVER_RUNTIME_STATE = "previous_response_id"
ENCRYPTED_REPLAY_RUNTIME_STATE = "encrypted_replay"
# Backwards-compatible alias: most call sites and tests reference the default.
SUPPORTED_RUNTIME_STATE = DEFAULT_RUNTIME_STATE
SUPPORTED_RUNTIME_STATES = frozenset(
    {
        DEFAULT_RUNTIME_STATE,
        SERVER_RUNTIME_STATE,
        ENCRYPTED_REPLAY_RUNTIME_STATE,
    }
)
# Response-ID chaining is OpenAI-only. Encrypted replay additionally supports
# Anthropic's stateless beta Messages compaction API.
SERVER_STATE_RUNTIME_PAIRS = frozenset({("openai-python", "responses")})
ENCRYPTED_REPLAY_RUNTIME_PAIRS = frozenset(
    {
        ("openai-python", "responses"),
        ("anthropic-python", "messages"),
    }
)
ANTHROPIC_COMPACTION_BETA = "compact-2026-01-12"
ANTHROPIC_COMPACTION_EDIT = "compact_20260112"
ANTHROPIC_MIN_COMPACTION_TOKENS = 50_000
ANTHROPIC_OPENAI_COMPAT_CLIENT_FIELDS = frozenset({"base_url"})
ANTHROPIC_OPENAI_COMPAT_REQUEST_FIELDS = frozenset(
    {
        "extra_body",
        "max_completion_tokens",
        "max_output_tokens",
    }
)


def _read_raw_model_configs() -> list[dict[str, Any]]:
    if not MODEL_CONFIG_PATH.exists():
        raise ValueError(f"Model config file not found: {MODEL_CONFIG_PATH}")

    try:
        configs = yaml.safe_load(MODEL_CONFIG_PATH.read_text()) or []
    except OSError as e:
        raise ValueError(
            f"Failed to read model config file {MODEL_CONFIG_PATH}: {e}"
        ) from e

    if not isinstance(configs, list):
        raise ValueError(f"Model config file is invalid: {MODEL_CONFIG_PATH}")

    return configs


def _format_supported_runtime_pairs() -> str:
    return ", ".join(
        f"(sdk={sdk!r}, api={api!r})" for sdk, api in sorted(SUPPORTED_RUNTIME_PAIRS)
    )


def _validate_anthropic_messages_config(config_id: str, entry: dict[str, Any]) -> None:
    client = entry["client"]
    request = entry["request"]

    invalid_client_fields = sorted(
        ANTHROPIC_OPENAI_COMPAT_CLIENT_FIELDS.intersection(client)
    )
    if invalid_client_fields:
        fields = ", ".join(invalid_client_fields)
        raise ValueError(
            f"Model config '{config_id}' uses OpenAI-compatible client field(s) "
            f"for native Anthropic runtime: {fields}."
        )

    invalid_request_fields = sorted(
        ANTHROPIC_OPENAI_COMPAT_REQUEST_FIELDS.intersection(request)
    )
    if invalid_request_fields:
        fields = ", ".join(invalid_request_fields)
        raise ValueError(
            f"Model config '{config_id}' uses OpenAI-compatible request field(s) "
            f"for native Anthropic runtime: {fields}."
        )


def _validate_openai_encrypted_replay_config(
    config_id: str,
    entry: dict[str, Any],
) -> None:
    """Enforce OpenAI request invariants for stateless encrypted replay."""
    request = entry["request"]
    if request.get("store") is not False:
        raise ValueError(
            f"Model config '{config_id}' uses runtime.state="
            f"{ENCRYPTED_REPLAY_RUNTIME_STATE!r} and must set request.store=false."
        )
    if request.get("background") is True:
        raise ValueError(
            f"Model config '{config_id}' uses runtime.state="
            f"{ENCRYPTED_REPLAY_RUNTIME_STATE!r} and cannot enable "
            f"request.background."
        )

    incompatible_fields = sorted(
        field for field in ("conversation", "previous_response_id") if field in request
    )
    if incompatible_fields:
        fields = ", ".join(incompatible_fields)
        raise ValueError(
            f"Model config '{config_id}' uses runtime.state="
            f"{ENCRYPTED_REPLAY_RUNTIME_STATE!r} with incompatible request "
            f"field(s): {fields}."
        )


def _validate_anthropic_encrypted_replay_config(
    config_id: str,
    entry: dict[str, Any],
) -> None:
    """Require Anthropic's stateless compaction and continuation contract."""
    request = entry["request"]
    betas = request.get("betas")
    if not isinstance(betas, list) or ANTHROPIC_COMPACTION_BETA not in betas:
        raise ValueError(
            f"Model config '{config_id}' uses runtime.state="
            f"{ENCRYPTED_REPLAY_RUNTIME_STATE!r} on Anthropic and must include "
            f"request.betas={ANTHROPIC_COMPACTION_BETA!r}."
        )

    context_management = request.get("context_management")
    edits = (
        context_management.get("edits")
        if isinstance(context_management, dict)
        else None
    )
    compact_edits = [
        edit
        for edit in edits or []
        if isinstance(edit, dict) and edit.get("type") == ANTHROPIC_COMPACTION_EDIT
    ]
    if (
        not isinstance(edits, list)
        or len(edits) != 1
        or len(compact_edits) != 1
    ):
        raise ValueError(
            f"Model config '{config_id}' uses Anthropic encrypted replay and "
            f"must define exactly one request.context_management edit, with "
            f"type={ANTHROPIC_COMPACTION_EDIT!r}."
        )

    compact_edit = compact_edits[0]
    if compact_edit.get("pause_after_compaction") is not False:
        raise ValueError(
            f"Model config '{config_id}' uses Anthropic encrypted replay and "
            f"must set pause_after_compaction=false."
        )
    if "instructions" in compact_edit:
        raise ValueError(
            f"Model config '{config_id}' uses Anthropic encrypted replay and "
            f"cannot override the provider's compaction instructions."
        )

    trigger = compact_edit.get("trigger")
    trigger_value = trigger.get("value") if isinstance(trigger, dict) else None
    if (
        not isinstance(trigger, dict)
        or trigger.get("type") != "input_tokens"
        or isinstance(trigger_value, bool)
        or not isinstance(trigger_value, int)
        or trigger_value < ANTHROPIC_MIN_COMPACTION_TOKENS
    ):
        raise ValueError(
            f"Model config '{config_id}' uses Anthropic encrypted replay and "
            f"must define an input_tokens compaction trigger of at least "
            f"{ANTHROPIC_MIN_COMPACTION_TOKENS}."
        )

    thinking = request.get("thinking")
    if not isinstance(thinking, dict) or thinking.get("type") != "adaptive":
        raise ValueError(
            f"Model config '{config_id}' uses Anthropic encrypted replay and "
            f"must enable adaptive thinking."
        )
    if thinking.get("display") != "summarized":
        raise ValueError(
            f"Model config '{config_id}' uses Anthropic encrypted replay and "
            f"must request summarized thinking alongside encrypted signatures."
        )


def _validate_encrypted_replay_config(
    config_id: str,
    entry: dict[str, Any],
    runtime_pair: tuple[str, str],
) -> None:
    if runtime_pair == ("openai-python", "responses"):
        _validate_openai_encrypted_replay_config(config_id, entry)
        return
    _validate_anthropic_encrypted_replay_config(config_id, entry)


def _validate_model_config_entry(
    entry: Any, index: int, seen_ids: set[str]
) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError(
            f"Model config entry #{index} in {MODEL_CONFIG_PATH} must be a mapping."
        )

    raw_config_id = entry.get("id")
    if not isinstance(raw_config_id, str) or not raw_config_id.strip():
        legacy_name = entry.get("name")
        if isinstance(legacy_name, str) and legacy_name.strip():
            raise ValueError(
                f"Model config entry #{index} uses legacy field 'name'. "
                f"Rename it to 'id'."
            )
        raise ValueError(f"Model config entry #{index} is missing required 'id'.")

    config_id = raw_config_id.strip()
    if config_id in seen_ids:
        raise ValueError(
            f"Duplicate model config id '{config_id}' found in {MODEL_CONFIG_PATH}."
        )
    seen_ids.add(config_id)

    for section in REQUIRED_CONFIG_SECTIONS:
        if not isinstance(entry.get(section), dict):
            raise ValueError(
                f"Model config '{config_id}' is missing required section '{section}'."
            )

    agent = entry.get("agent", {})
    if agent is not None and not isinstance(agent, dict):
        raise ValueError(
            f"Model config '{config_id}' section 'agent' must be a mapping if present."
        )
    if isinstance(agent, dict) and "analysis_mode" in agent:
        if not isinstance(agent["analysis_mode"], bool):
            raise ValueError(
                f"Model config '{config_id}' agent.analysis_mode must be a boolean."
            )
    if isinstance(agent, dict) and "MAX_RUNTIME_SECONDS" in agent:
        max_runtime = agent["MAX_RUNTIME_SECONDS"]
        if (
            isinstance(max_runtime, bool)
            or not isinstance(max_runtime, (int, float))
            or max_runtime <= 0
        ):
            raise ValueError(
                f"Model config '{config_id}' agent.MAX_RUNTIME_SECONDS must be "
                f"a positive number (seconds)."
            )

    runtime = entry["runtime"]
    if not isinstance(runtime.get("sdk"), str) or not runtime["sdk"].strip():
        raise ValueError(f"Model config '{config_id}' is missing runtime.sdk.")
    if not isinstance(runtime.get("api"), str) or not runtime["api"].strip():
        raise ValueError(f"Model config '{config_id}' is missing runtime.api.")
    runtime_pair = (runtime["sdk"], runtime["api"])
    if runtime_pair not in SUPPORTED_RUNTIME_PAIRS:
        raise ValueError(
            f"Model config '{config_id}' uses unsupported runtime "
            f"(sdk={runtime['sdk']!r}, api={runtime['api']!r}). "
            f"Supported runtimes: {_format_supported_runtime_pairs()}."
        )
    runtime_state = runtime.get("state")
    if runtime_state not in SUPPORTED_RUNTIME_STATES:
        supported = ", ".join(repr(s) for s in sorted(SUPPORTED_RUNTIME_STATES))
        raise ValueError(
            f"Model config '{config_id}' uses runtime.state={runtime_state!r}, "
            f"but only {supported} are supported."
        )
    if (
        runtime_state == SERVER_RUNTIME_STATE
        and runtime_pair not in SERVER_STATE_RUNTIME_PAIRS
    ):
        raise ValueError(
            f"Model config '{config_id}' uses runtime.state={runtime_state!r}, "
            f"which is only supported on the OpenAI Responses runtime "
            f"(sdk='openai-python', api='responses')."
        )
    if (
        runtime_state == ENCRYPTED_REPLAY_RUNTIME_STATE
        and runtime_pair not in ENCRYPTED_REPLAY_RUNTIME_PAIRS
    ):
        raise ValueError(
            f"Model config '{config_id}' uses runtime.state={runtime_state!r}, "
            f"which is only supported on OpenAI Responses or Anthropic "
            f"Messages runtimes."
        )
    if runtime_state == ENCRYPTED_REPLAY_RUNTIME_STATE:
        _validate_encrypted_replay_config(config_id, entry, runtime_pair)
    if runtime_pair == ("anthropic-python", "messages"):
        _validate_anthropic_messages_config(config_id, entry)

    return entry


def load_model_configs() -> list[dict[str, Any]]:
    seen_ids: set[str] = set()
    return [
        _validate_model_config_entry(entry, index, seen_ids)
        for index, entry in enumerate(_read_raw_model_configs(), start=1)
    ]


def list_model_config_ids() -> list[str]:
    return [entry["id"] for entry in load_model_configs()]


def get_model_config(config_id: str) -> dict[str, Any]:
    for entry in load_model_configs():
        if entry["id"] == config_id:
            return entry

    available_configs = ", ".join(sorted(list_model_config_ids()))
    raise ValueError(
        f"Model config '{config_id}' not found in {MODEL_CONFIG_PATH}. "
        f"Available configs: {available_configs}"
    )
