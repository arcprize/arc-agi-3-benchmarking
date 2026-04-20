from __future__ import annotations

import os
from typing import Any

from anthropic import Anthropic as AnthropicClient
from openai import OpenAI as OpenAIClient

DEFAULT_ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
DEFAULT_OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENAI_API_KEY_ENV = "OPENROUTER_API_KEY"


def _read_required_api_key(
    *,
    client_config: dict[str, Any],
    config_id: str,
    default_api_key_env: str,
) -> str:
    api_key_env = str(
        client_config.get("api_key_env", default_api_key_env)
    ).strip()
    if not api_key_env:
        raise ValueError(
            f"Model config '{config_id}' is missing client.api_key_env."
        )

    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise ValueError(
            f"No {api_key_env} set. "
            f"The selected model config '{config_id}' requires "
            f"the {api_key_env} environment variable to be set in your .env file."
        )
    return api_key


def build_model_runtime_client(
    *,
    runtime_config: dict[str, Any],
    client_config: dict[str, Any],
    config_id: str,
) -> Any:
    sdk = runtime_config.get("sdk")
    if sdk == "openai-python":
        api_key = _read_required_api_key(
            client_config=client_config,
            config_id=config_id,
            default_api_key_env=DEFAULT_OPENAI_API_KEY_ENV,
        )
        return OpenAIClient(
            base_url=client_config.get("base_url", DEFAULT_OPENAI_BASE_URL),
            api_key=api_key,
        )

    if sdk == "anthropic-python":
        api_key = _read_required_api_key(
            client_config=client_config,
            config_id=config_id,
            default_api_key_env=DEFAULT_ANTHROPIC_API_KEY_ENV,
        )
        return AnthropicClient(api_key=api_key)

    raise ValueError(
        f"Model config '{config_id}' uses unsupported runtime "
        f"(sdk={sdk!r})."
    )
