"""Local registry for stable runtime adapter identifiers."""

from __future__ import annotations

from typing import Any

from .anthropic_runtime import AnthropicContinuousConversationRuntimeAdapter
from .google_runtime import GoogleContinuousConversationRuntimeAdapter
from .openai_runtime import OpenAIContinuousConversationRuntimeAdapter
from .runtime_state import (
    CONTINUOUS_CONVERSATION_RUNTIME_STATE,
    AdapterDescriptor,
    StatefulRuntimeAdapter,
)
from .xai_runtime import XAIContinuousConversationRuntimeAdapter

OPENAI_RESPONSES_ADAPTER_ID = "openai.responses.v1"
ANTHROPIC_MESSAGES_ADAPTER_ID = "anthropic.messages.v1"
GOOGLE_INTERACTIONS_ADAPTER_ID = "google.interactions.v1"
XAI_RESPONSES_ADAPTER_ID = "xai.responses.v1"

_LEGACY_RUNTIME_ADAPTER_IDS = {
    ("openai-python", "chat_completions"): "openai.chat_completions.v1",
    ("openai-python", "responses"): OPENAI_RESPONSES_ADAPTER_ID,
    ("anthropic-python", "messages"): "anthropic.messages.v1",
    ("google-genai", "generate_content"): "google.generate_content.v1",
    ("google-genai", "interactions"): GOOGLE_INTERACTIONS_ADAPTER_ID,
}

ADAPTER_DESCRIPTORS = {
    XAI_RESPONSES_ADAPTER_ID: AdapterDescriptor(
        adapter_id=XAI_RESPONSES_ADAPTER_ID,
        provider="xai",
        api_surface="responses",
        implementation_path="benchmarking/xai_runtime.py",
        version="1",
        approval_status="unreviewed",
    ),
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
        implementation_path="benchmarking/anthropic_runtime.py",
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
    GOOGLE_INTERACTIONS_ADAPTER_ID: AdapterDescriptor(
        adapter_id=GOOGLE_INTERACTIONS_ADAPTER_ID,
        provider="google",
        api_surface="interactions",
        implementation_path="benchmarking/google_runtime.py",
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
    if explicit == XAI_RESPONSES_ADAPTER_ID:
        if runtime_pair != ("openai-python", "responses") or runtime_config.get(
            "state"
        ) != CONTINUOUS_CONVERSATION_RUNTIME_STATE:
            raise ValueError(
                "xai.responses.v1 requires openai-python/responses with "
                "continuous_conversation state."
            )
        return explicit
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
        if adapter_id == ANTHROPIC_MESSAGES_ADAPTER_ID:
            return AnthropicContinuousConversationRuntimeAdapter(
                model_adapter=model_adapter,
                descriptor=descriptor,
                compaction=runtime_config.get("compaction"),
            )
        if adapter_id == XAI_RESPONSES_ADAPTER_ID:
            return XAIContinuousConversationRuntimeAdapter(
                model_adapter=model_adapter,
                descriptor=descriptor,
                compaction=runtime_config.get("compaction"),
            )
        if adapter_id == OPENAI_RESPONSES_ADAPTER_ID:
            return OpenAIContinuousConversationRuntimeAdapter(
                model_adapter=model_adapter, descriptor=descriptor
            )
        if adapter_id == GOOGLE_INTERACTIONS_ADAPTER_ID:
            return GoogleContinuousConversationRuntimeAdapter(
                model_adapter=model_adapter, descriptor=descriptor
            )
        raise ValueError(
            f"Model config '{config_id}' uses continuous_conversation with "
            f"unsupported adapter_id={adapter_id!r}."
        )
    raise ValueError(
        f"Model config '{config_id}' cannot use the stateful turn contract with "
        f"runtime.state={strategy!r}; only "
        f"{CONTINUOUS_CONVERSATION_RUNTIME_STATE!r} is opt-in."
    )
