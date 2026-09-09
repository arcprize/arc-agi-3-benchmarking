"""Google-specific continuous-conversation state strategy."""

from __future__ import annotations

from typing import Any

from .runtime_models import Message, ModelRequest, ModelResponse
from .runtime_state import (
    CONTINUOUS_CONVERSATION_RUNTIME_STATE,
    AdapterDescriptor,
    ModelTurnRequest,
    ModelTurnResult,
    RuntimeState,
    StateTransitionTelemetry,
    replace_runtime_payload,
    runtime_payload_items,
    sanitize_settings,
)


def _response_steps(response: ModelResponse) -> list[Any]:
    raw_response = response.raw_response
    if isinstance(raw_response, dict):
        return list(raw_response.get("steps", []) or [])
    return list(getattr(raw_response, "steps", []) or [])


def serialize_interaction_steps(response: ModelResponse) -> list[dict[str, Any]]:
    """Serialize every model-generated step for exact stateless replay."""
    serialized: list[dict[str, Any]] = []
    for step in _response_steps(response):
        if hasattr(step, "model_dump"):
            value = step.model_dump(mode="json")
        elif isinstance(step, dict):
            value = dict(step)
        elif hasattr(step, "__dict__"):
            value = dict(vars(step))
        else:
            raise TypeError(
                "Google continuous conversation steps must be mappings or "
                "support model_dump()."
            )
        if not isinstance(value, dict):
            raise TypeError(
                "Google continuous conversation steps must serialize to mappings."
            )
        serialized.append(value)
    if not serialized:
        raise RuntimeError(
            "Google continuous conversation response did not contain reusable steps."
        )
    return serialized


def message_to_interaction_step(message: Message) -> dict[str, Any]:
    if message.role == "user":
        return {
            "type": "user_input",
            "content": [{"type": "text", "text": message.content}],
        }
    if message.role == "assistant":
        return {
            "type": "model_output",
            "content": [{"type": "text", "text": message.content}],
        }
    raise ValueError(
        "Google continuous conversation system messages must be passed separately."
    )


def sanitized_step_descriptor(step: dict[str, Any]) -> dict[str, Any]:
    descriptor: dict[str, Any] = {"type": step.get("type", "unknown")}
    if step.get("id") is not None:
        descriptor["id"] = step["id"]
    return descriptor


def validate_google_continuous_conversation_request(
    request_config: dict[str, Any],
) -> None:
    if request_config.get("store") is not False:
        raise ValueError("Google continuous conversation requires store=false.")
    if request_config.get("background") is True:
        raise ValueError(
            "Google continuous conversation does not support background mode."
        )
    if "previous_interaction_id" in request_config:
        raise ValueError(
            "Google continuous conversation does not support "
            "previous_interaction_id."
        )
    generation_config = request_config.get("generation_config")
    if not isinstance(generation_config, dict):
        raise ValueError(
            "Google continuous conversation requires generation_config."
        )
    if generation_config.get("thinking_summaries") != "auto":
        raise ValueError(
            "Google continuous conversation requires "
            "generation_config.thinking_summaries=auto."
        )


class GoogleContinuousConversationRuntimeAdapter:
    strategy = CONTINUOUS_CONVERSATION_RUNTIME_STATE
    provides_continuous_conversation = True

    def __init__(self, *, model_adapter: Any, descriptor: AdapterDescriptor) -> None:
        self._model_adapter = model_adapter
        self.descriptor = descriptor

    def initial_state(self) -> RuntimeState:
        return RuntimeState(
            adapter_id=self.descriptor.adapter_id,
            strategy=self.strategy,
            payload={"steps": []},
        )

    def buffer_inputs(
        self, state: RuntimeState, messages: list[Message]
    ) -> RuntimeState:
        state.validate_for(
            adapter_id=self.descriptor.adapter_id, strategy=self.strategy
        )
        steps = runtime_payload_items(state, "steps")
        steps.extend(message_to_interaction_step(message) for message in messages)
        return replace_runtime_payload(state, {"steps": steps})

    def invoke_turn(self, request: ModelTurnRequest) -> ModelTurnResult:
        request.previous_state.validate_for(
            adapter_id=self.descriptor.adapter_id, strategy=self.strategy
        )
        validate_google_continuous_conversation_request(request.request_config)
        input_steps = runtime_payload_items(request.previous_state, "steps")
        input_steps.extend(
            message_to_interaction_step(message) for message in request.new_messages
        )
        model_request = ModelRequest(
            messages=[
                Message(role="system", content=request.system_prompt),
                *request.new_messages,
            ],
            request_config=dict(request.request_config),
            native_input=input_steps,
        )
        response = self._model_adapter.invoke(model_request)
        output_steps = serialize_interaction_steps(response)
        next_steps = [*input_steps, *output_steps]
        descriptors = [sanitized_step_descriptor(step) for step in input_steps]
        transition = StateTransitionTelemetry(
            input_items_sent=len(input_steps),
            history_items_before_prune=len(next_steps),
            history_items_after_prune=len(next_steps),
            sanitized_items=descriptors,
        )
        return ModelTurnResult(
            response=response,
            state=replace_runtime_payload(
                request.previous_state, {"steps": next_steps}
            ),
            sanitized_request={
                "instructions_present": True,
                "input_items": descriptors,
                "settings": sanitize_settings(request.request_config),
            },
            transition=transition,
            action_state={
                "input_items_sent": len(input_steps),
                "history_items_before_prune": len(next_steps),
                "history_items_after_prune": len(next_steps),
            },
        )
