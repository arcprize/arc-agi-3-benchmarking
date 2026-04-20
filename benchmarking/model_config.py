from pathlib import Path
from typing import Any

import yaml

MODEL_CONFIG_PATH = Path(__file__).resolve().parent / "model_configs.yaml"
REQUIRED_CONFIG_SECTIONS = ("runtime", "client", "request")
SUPPORTED_RUNTIME_PAIRS = frozenset(
    {
        ("anthropic-python", "messages"),
        ("openai-python", "chat_completions"),
        ("openai-python", "responses"),
    }
)
SUPPORTED_RUNTIME_STATE = "manual_rolling"
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
        f"(sdk={sdk!r}, api={api!r})"
        for sdk, api in sorted(SUPPORTED_RUNTIME_PAIRS)
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


def _validate_model_config_entry(entry: Any, index: int, seen_ids: set[str]) -> dict[str, Any]:
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
        raise ValueError(
            f"Model config entry #{index} is missing required 'id'."
        )

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
    if runtime.get("state") != SUPPORTED_RUNTIME_STATE:
        raise ValueError(
            f"Model config '{config_id}' uses runtime.state={runtime.get('state')!r}, "
            f"but only '{SUPPORTED_RUNTIME_STATE}' is supported."
        )
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
