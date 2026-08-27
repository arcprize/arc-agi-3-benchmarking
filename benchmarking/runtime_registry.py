"""Local registry for stable runtime adapter identifiers."""

from __future__ import annotations

from typing import Any

from .openai_runtime import OpenAIEncryptedReplayRuntimeAdapter
from .runtime_state import (
    CONTINUOUS_CONVERSATION_RUNTIME_STATE,
    AdapterDescriptor,
    StatefulRuntimeAdapter,
)

OPENAI_RESPONSES_ADAPTER_ID = "openai.responses.v1"

_LEGACY_RUNTIME_ADAPTER_IDS = {
    ("openai-python", "chat_completions"): "openai.chat_completions.v1",
    ("openai-python", "responses"): OPENAI_RESPONSES_ADAPTER_ID,
    ("anthropic-python", "messages"): "anthropic.messages.v1",
    ("google-genai", "generate_content"): "google.generate_content.v1",
}

ADAPTER_DESCRIPTORS = {
    "openai.chat_completions.v1": AdapterDescriptor(
        adapter_id="openai.chat_completions.v1",
        provider="openai-compatible",
        api_surface="chat_completions",
        implementation_path="benchmarking/runtime_adapters.py",
        version="1",
        approval_status="unreviewed",
    ),
    OPENAI_RESPONSES_ADAPTER_ID: AdapterDescriptor(
        adapter_id=OPENAI_RESPONSES_ADAPTER_ID,
        provider="openai",
        api_surface="responses",
        implementation_path="benchmarking/openai_runtime.py",
        version="1",
        approval_status="provider_reference",
    ),
    "anthropic.messages.v1": AdapterDescriptor(
        adapter_id="anthropic.messages.v1",
        provider="anthropic",
        api_surface="messages",
        implementation_path="benchmarking/runtime_adapters.py",
        version="1",
        approval_status="unreviewed",
    ),
    "google.generate_content.v1": AdapterDescriptor(
        adapter_id="google.generate_content.v1",
        provider="google",
        api_surface="generate_content",
        implementation_path="benchmarking/runtime_adapters.py",
        version="1",
        approval_status="unreviewed",
    ),
}


def resolve_adapter_id(runtime_config: dict[str, Any], config_id: str) -> str:
    sdk = runtime_config.get("sdk")
    api = runtime_config.get("api")
    runtime_pair = (sdk, api) if isinstance(sdk, str) and isinstance(api, str) else None
    derived = (
        _LEGACY_RUNTIME_ADAPTER_IDS.get(runtime_pair)
        if runtime_pair is not None
        else None
    )
    explicit = runtime_config.get("adapter_id")
    if explicit is None:
        if derived is None:
            raise ValueError(f"Model config '{config_id}' has no registered adapter.")
        return derived
    if not isinstance(explicit, str) or explicit not in ADAPTER_DESCRIPTORS:
        raise ValueError(
            f"Model config '{config_id}' uses unknown runtime.adapter_id={explicit!r}."
        )
    if derived is not None and explicit != derived:
        raise ValueError(
            f"Model config '{config_id}' uses runtime.adapter_id={explicit!r}, "
            f"which does not match sdk={sdk!r}, api={api!r}."
        )
    return explicit


def build_stateful_runtime_adapter(
    *,
    model_adapter: Any,
    runtime_config: dict[str, Any],
    config_id: str,
) -> StatefulRuntimeAdapter:
    adapter_id = resolve_adapter_id(runtime_config, config_id)
    descriptor = ADAPTER_DESCRIPTORS[adapter_id]
    strategy = runtime_config.get("state")
    if strategy == CONTINUOUS_CONVERSATION_RUNTIME_STATE:
        if adapter_id != OPENAI_RESPONSES_ADAPTER_ID:
            raise ValueError(
                f"Model config '{config_id}' uses continuous_conversation with "
                f"adapter_id={adapter_id!r}; only {OPENAI_RESPONSES_ADAPTER_ID!r} "
                "supports it."
            )
        return OpenAIEncryptedReplayRuntimeAdapter(
            model_adapter=model_adapter, descriptor=descriptor
        )
    raise ValueError(
        f"Model config '{config_id}' cannot use the stateful turn contract with "
        f"runtime.state={strategy!r}; only "
        f"{CONTINUOUS_CONVERSATION_RUNTIME_STATE!r} is opt-in."
    )
