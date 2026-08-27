"""Provider-neutral state and turn contracts for model runtimes."""

from __future__ import annotations

import json
import math
import os
from typing import Any, Protocol

from pydantic import BaseModel, Field, model_validator

from .runtime_models import Message, ModelRequest, ModelResponse

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
    strategy: str
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
    reasoning_record_field: str = "reasoning"
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


def _assistant_content(request: ModelTurnRequest, response: ModelResponse) -> str:
    if not request.include_reasoning_summary_in_transcript or not response.reasoning_text:
        return response.output_text
    return (
        "<reasoning_summary>\n"
        f"{response.reasoning_text}\n"
        "</reasoning_summary>\n\n"
        f"{response.output_text}\n"
    )


def _trim_messages(
    messages: list[dict[str, Any]],
    *,
    max_context_length: int,
    chars_per_token: float,
) -> list[dict[str, Any]]:
    """Apply the harness's existing oldest-turn token/window policy."""

    trimmed = list(messages)
    while math.ceil(
        sum(len(str(item.get("content", ""))) for item in trimmed)
        / chars_per_token
    ) > max_context_length:
        user_index = next(
            (index for index, item in enumerate(trimmed) if item.get("role") == "user"),
            None,
        )
        if user_index is None:
            break
        end = user_index + 1
        if end < len(trimmed) and trimmed[end].get("role") == "assistant":
            end += 1
        if len(trimmed) - (end - user_index) < 1:
            break
        trimmed = trimmed[:user_index] + trimmed[end:]
    return trimmed


class ManualRollingRuntimeAdapter:
    strategy = DEFAULT_RUNTIME_STATE
    provides_continuous_conversation = False

    def __init__(
        self, *, model_adapter: Any, descriptor: AdapterDescriptor
    ) -> None:
        self._model_adapter = model_adapter
        self.descriptor = descriptor

    def initial_state(self) -> RuntimeState:
        return RuntimeState(
            adapter_id=self.descriptor.adapter_id,
            strategy=self.strategy,
            payload={"messages": []},
        )

    def buffer_inputs(
        self, state: RuntimeState, messages: list[Message]
    ) -> RuntimeState:
        state.validate_for(
            adapter_id=self.descriptor.adapter_id, strategy=self.strategy
        )
        history = runtime_payload_items(state, "messages")
        history.extend(message.model_dump() for message in messages)
        return replace_runtime_payload(state, {"messages": history})

    def invoke_turn(self, request: ModelTurnRequest) -> ModelTurnResult:
        request.previous_state.validate_for(
            adapter_id=self.descriptor.adapter_id, strategy=self.strategy
        )
        history = runtime_payload_items(request.previous_state, "messages")
        history.extend(message.model_dump() for message in request.new_messages)
        history = _trim_messages(
            history,
            max_context_length=request.max_context_length,
            chars_per_token=request.estimated_chars_per_token,
        )
        normalized_messages = [
            Message(role="system", content=request.system_prompt),
            *[Message.model_validate(item) for item in history],
        ]
        response = self._model_adapter.invoke(
            ModelRequest(
                messages=normalized_messages,
                request_config=dict(request.request_config),
            )
        )
        next_history = [
            *history,
            {
                "role": "assistant",
                "content": _assistant_content(request, response),
            },
        ]
        next_state = replace_runtime_payload(
            request.previous_state, {"messages": next_history}
        )
        sent = [message.model_dump() for message in normalized_messages]
        return ModelTurnResult(
            response=response,
            state=next_state,
            sanitized_request={"messages": sent},
            transition=StateTransitionTelemetry(
                strategy=self.strategy,
                input_items_sent=len(sent),
            ),
        )


class PreviousResponseIdRuntimeAdapter:
    strategy = SERVER_RUNTIME_STATE
    provides_continuous_conversation = False

    def __init__(
        self, *, model_adapter: Any, descriptor: AdapterDescriptor
    ) -> None:
        self._model_adapter = model_adapter
        self.descriptor = descriptor

    def initial_state(self) -> RuntimeState:
        return RuntimeState(
            adapter_id=self.descriptor.adapter_id,
            strategy=self.strategy,
            payload={"response_id": None, "pending_inputs": []},
        )

    def buffer_inputs(
        self, state: RuntimeState, messages: list[Message]
    ) -> RuntimeState:
        state.validate_for(
            adapter_id=self.descriptor.adapter_id, strategy=self.strategy
        )
        payload = dict(state.payload)
        pending = runtime_payload_items(state, "pending_inputs")
        pending.extend(message.model_dump() for message in messages)
        payload["pending_inputs"] = pending
        return replace_runtime_payload(state, payload)

    def invoke_turn(self, request: ModelTurnRequest) -> ModelTurnResult:
        request.previous_state.validate_for(
            adapter_id=self.descriptor.adapter_id, strategy=self.strategy
        )
        payload = request.previous_state.payload
        response_id = payload.get("response_id")
        if response_id is not None and not isinstance(response_id, str):
            raise ValueError("Runtime state payload.response_id must be a string or null.")
        pending = runtime_payload_items(request.previous_state, "pending_inputs")
        pending.extend(message.model_dump() for message in request.new_messages)
        request_config = dict(request.request_config)
        messages = [Message.model_validate(item) for item in pending]
        if response_id is None:
            messages.insert(0, Message(role="system", content=request.system_prompt))
        else:
            request_config["previous_response_id"] = response_id
        response = self._model_adapter.invoke(
            ModelRequest(messages=messages, request_config=request_config)
        )
        next_state = replace_runtime_payload(
            request.previous_state,
            {
                "response_id": response.response_id,
                "pending_inputs": [],
            },
        )
        sent = [message.model_dump() for message in messages]
        return ModelTurnResult(
            response=response,
            state=next_state,
            sanitized_request={"messages": sent},
            transition=StateTransitionTelemetry(
                strategy=self.strategy,
                input_items_sent=len(sent),
            ),
        )


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
