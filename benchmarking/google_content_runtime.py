"""Native generateContent history, including signatures, across accepted turns."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .runtime_models import Message, ModelRequest, ModelResponse
from .runtime_state import (
    CONTINUOUS_CONVERSATION_RUNTIME_STATE,
    AdapterDescriptor,
    CompactionUnwindResult,
    ModelTurnRequest,
    ModelTurnResult,
    RuntimeState,
    StateTransitionTelemetry,
    append_accepted_turn,
    replace_runtime_payload,
    restore_unwound_runtime_state_items,
    runtime_payload_items,
    sanitize_settings,
    unwind_runtime_state_items,
)


def validate_content_request(config: dict[str, Any]) -> None:
    for field in ("store", "previous_interaction_id", "background"):
        if field in config:
            raise ValueError(f"generateContent native replay does not accept {field}.")
    if config.get("thinking_config", {}).get("include_thoughts") is not True:
        raise ValueError(
            "generateContent native replay requires thinking_config.include_thoughts=true."
        )


def message_to_content(message: Message) -> dict[str, Any]:
    if message.role not in ("user", "assistant"):
        raise ValueError("System instructions must be passed separately.")
    return {
        "role": "model" if message.role == "assistant" else "user",
        "parts": [{"text": message.content}],
    }


def response_contents(response: ModelResponse) -> list[dict[str, Any]]:
    raw = response.raw_response
    if not isinstance(raw, dict):
        raise ValueError("generateContent response must retain native content.")
    content = deepcopy(raw["candidates"][0]["content"])
    content["role"] = "model"
    return [content]


def content_descriptor(content: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "content",
        "role": content.get("role"),
        "part_count": len(content.get("parts", [])),
    }


def readable_content_messages(
    *, system_prompt: str, input_steps: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for content in input_steps:
        text = "\n".join(
            p["text"]
            for p in content.get("parts", [])
            if isinstance(p.get("text"), str)
        )
        messages.append(
            {
                "role": "assistant" if content.get("role") == "model" else "user",
                "content": text,
            }
        )
    return messages


class GoogleContentConversationRuntimeAdapter:
    strategy = CONTINUOUS_CONVERSATION_RUNTIME_STATE
    provides_continuous_conversation = True

    def __init__(self, *, model_adapter: Any, descriptor: AdapterDescriptor) -> None:
        self._model_adapter = model_adapter
        self.descriptor = descriptor

    def initial_state(self) -> RuntimeState:
        return RuntimeState(
            adapter_id=self.descriptor.adapter_id,
            strategy=self.strategy,
            payload={"contents": []},
        )

    def buffer_inputs(
        self, state: RuntimeState, messages: list[Message]
    ) -> RuntimeState:
        state.validate_for(
            adapter_id=self.descriptor.adapter_id, strategy=self.strategy
        )
        steps = runtime_payload_items(state, "contents")
        steps.extend(message_to_content(message) for message in messages)
        return replace_runtime_payload(state, {"contents": steps})

    def invoke_turn(self, request: ModelTurnRequest) -> ModelTurnResult:
        request.previous_state.validate_for(
            adapter_id=self.descriptor.adapter_id, strategy=self.strategy
        )
        validate_content_request(request.request_config)
        previous_steps = runtime_payload_items(request.previous_state, "contents")
        turn_start = len(previous_steps)
        input_steps = list(previous_steps)
        input_steps.extend(
            message_to_content(message) for message in request.new_messages
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
        output_steps = response_contents(response)
        next_steps = [*input_steps, *output_steps]
        descriptors = [content_descriptor(step) for step in input_steps]
        transition = StateTransitionTelemetry(
            input_items_sent=len(input_steps),
            history_items_before_prune=len(next_steps),
            history_items_after_prune=len(next_steps),
            sanitized_items=descriptors,
        )
        return ModelTurnResult(
            response=response,
            state=append_accepted_turn(
                state=request.previous_state,
                payload={"contents": next_steps},
                start_item=turn_start,
                end_item=len(next_steps),
                request_messages=request.new_messages,
                response=response,
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
            readable_request_messages=readable_content_messages(
                system_prompt=request.system_prompt,
                input_steps=input_steps,
            ),
        )

    def unwind_latest_accepted_turn(
        self, state: RuntimeState
    ) -> CompactionUnwindResult | None:
        state.validate_for(
            adapter_id=self.descriptor.adapter_id, strategy=self.strategy
        )
        return unwind_runtime_state_items(state, payload_key="contents")

    def rebuild_after_compaction(
        self,
        summary_message: Message,
        retained_turns: list[CompactionUnwindResult],
    ) -> RuntimeState:
        summary_state = self.buffer_inputs(
            self.initial_state(),
            [summary_message],
        )
        return restore_unwound_runtime_state_items(
            summary_state,
            payload_key="contents",
            retained_turns=retained_turns,
        )
