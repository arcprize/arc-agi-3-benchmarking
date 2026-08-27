"""Provider-neutral state and turn contracts for model runtimes."""

from __future__ import annotations

import json
import os
from typing import Any, Protocol

from pydantic import BaseModel, Field, model_validator

from .runtime_models import Message, ModelResponse

RUNTIME_STATE_SCHEMA_VERSION = 1
DEFAULT_RUNTIME_STATE = "manual_rolling"
SERVER_RUNTIME_STATE = "previous_response_id"
CONTINUOUS_CONVERSATION_RUNTIME_STATE = "continuous_conversation"
SUPPORTED_RUNTIME_STATES = frozenset(
    {
        DEFAULT_RUNTIME_STATE,
        SERVER_RUNTIME_STATE,
        CONTINUOUS_CONVERSATION_RUNTIME_STATE,
    }
)


class RuntimeState(BaseModel):
    """Versioned, JSON-serializable envelope for provider-owned turn state."""

    schema_version: int = RUNTIME_STATE_SCHEMA_VERSION
    adapter_id: str
    strategy: str
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_envelope(self) -> RuntimeState:
        if self.schema_version != RUNTIME_STATE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported runtime state schema_version={self.schema_version}; "
                f"expected {RUNTIME_STATE_SCHEMA_VERSION}."
            )
        if not self.adapter_id:
            raise ValueError("Runtime state adapter_id must not be empty.")
        if self.strategy not in SUPPORTED_RUNTIME_STATES:
            raise ValueError(f"Unsupported runtime state strategy={self.strategy!r}.")
        try:
            json.dumps(self.payload)
        except (TypeError, ValueError) as exc:
            raise ValueError("Runtime state payload must be JSON-serializable.") from exc
        return self

    def validate_for(self, *, adapter_id: str, strategy: str) -> None:
        if self.adapter_id != adapter_id:
            raise ValueError(
                f"Runtime state adapter mismatch: state={self.adapter_id!r}, "
                f"selected={adapter_id!r}."
            )
        if self.strategy != strategy:
            raise ValueError(
                f"Runtime state strategy mismatch: state={self.strategy!r}, "
                f"selected={strategy!r}."
            )


class ModelTurnRequest(BaseModel):
    """One model turn plus the last accepted runtime state."""

    system_prompt: str
    new_messages: list[Message]
    request_config: dict[str, Any]
    previous_state: RuntimeState
    max_context_length: int = 100_000
    estimated_chars_per_token: float = 1.0
    include_reasoning_summary_in_transcript: bool = False


class StateTransitionTelemetry(BaseModel):
    input_items_sent: int = 0
    compaction_items_returned: int = 0
    history_items_before_prune: int | None = None
    history_items_after_prune: int | None = None
    sanitized_items: list[dict[str, Any]] = Field(default_factory=list)


class ModelTurnResult(BaseModel):
    """Normalized response and provisional state for an attempted turn."""

    response: ModelResponse
    state: RuntimeState
    sanitized_request: dict[str, Any]
    transition: StateTransitionTelemetry
    action_state: dict[str, Any] | None = None


class AdapterDescriptor(BaseModel):
    adapter_id: str
    provider: str
    api_surface: str
    implementation_path: str
    version: str
    approval_status: str


class StatefulRuntimeAdapter(Protocol):
    descriptor: AdapterDescriptor
    strategy: str
    provides_continuous_conversation: bool

    def initial_state(self) -> RuntimeState: ...

    def buffer_inputs(
        self, state: RuntimeState, messages: list[Message]
    ) -> RuntimeState: ...

    def invoke_turn(self, request: ModelTurnRequest) -> ModelTurnResult: ...


def replace_runtime_payload(
    state: RuntimeState, payload: dict[str, Any]
) -> RuntimeState:
    """Replace provider payload while re-running envelope validation."""

    return RuntimeState.model_validate(
        {**state.model_dump(exclude={"payload"}), "payload": payload}
    )


def runtime_payload_items(state: RuntimeState, key: str) -> list[dict[str, Any]]:
    value = state.payload.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, dict) for item in value
    ):
        raise ValueError(f"Runtime state payload.{key} must be a list of mappings.")
    return list(value)


def sanitize_settings(value: Any) -> Any:
    """Remove secrets and encrypted bodies from provenance and recordings."""

    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            lowered = key.lower()
            if lowered == "encrypted_content":
                continue
            if lowered in {
                "api_key",
                "authorization",
                "access_token",
                "bearer_token",
            }:
                sanitized[key] = "[redacted]"
            else:
                sanitized[key] = sanitize_settings(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_settings(item) for item in value]
    return value


def harness_commit_sha() -> str | None:
    value = os.environ.get("ARC_HARNESS_COMMIT_SHA", "").strip()
    return value or None


def source_permalink(
    *, repository: str, implementation_path: str, commit_sha: str | None
) -> str | None:
    if not commit_sha:
        return None
    return (
        f"{repository.rstrip('/')}/blob/{commit_sha}/"
        f"{implementation_path.lstrip('/')}"
    )
