"""OpenAI-specific continuous-conversation state strategy."""

from __future__ import annotations

from typing import Any

from .runtime_models import (
    Message,
    ModelRequest,
    ModelResponse,
    extract_responses_reasoning_summary,
)
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


def _response_output(response: ModelResponse) -> list[Any]:
    raw_response = response.raw_response
    if isinstance(raw_response, dict):
        return list(raw_response.get("output", []) or [])
    return list(getattr(raw_response, "output", []) or [])


def serialize_response_output_items(response: ModelResponse) -> list[dict[str, Any]]:
    """Serialize every native output item, dropping only rejected SDK fields."""

    serialized: list[dict[str, Any]] = []
    for item in _response_output(response):
        if hasattr(item, "model_dump"):
            value = item.model_dump(mode="json")
        elif isinstance(item, dict):
            value = dict(item)
        elif hasattr(item, "__dict__"):
            value = dict(vars(item))
        else:
            raise TypeError(
                "OpenAI continuous conversation output items must be mappings or "
                "support model_dump()."
            )
        if not isinstance(value, dict):
            raise TypeError(
                "OpenAI continuous conversation output items must serialize to "
                "mappings."
            )
        if value.get("type") == "reasoning":
            value.pop("status", None)
        elif value.get("type") == "compaction":
            value.pop("created_by", None)
        serialized.append(value)
    if not serialized:
        raise RuntimeError(
            "OpenAI continuous conversation response did not contain reusable "
            "output items."
        )
    return serialized


def prune_after_latest_compaction(
    input_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for index in range(len(input_items) - 1, -1, -1):
        if input_items[index].get("type") == "compaction":
            return input_items[index:]
    return input_items


def sanitized_item_descriptor(item: dict[str, Any]) -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "type": item.get("type", "message"),
    }
    for key in ("id", "role"):
        if item.get(key) is not None:
            descriptor[key] = item[key]
    return descriptor


def validate_continuous_conversation_request(
    request_config: dict[str, Any],
) -> None:
    """Enforce continuous-conversation invariants at the provider boundary."""

    if request_config.get("store") is not False:
        raise ValueError("OpenAI continuous conversation requires store=false.")
    if request_config.get("background") is True:
        raise ValueError(
            "OpenAI continuous conversation does not support background mode."
        )
    incompatible = sorted(
        field
        for field in ("conversation", "previous_response_id")
        if field in request_config
    )
    if incompatible:
        raise ValueError(
            "OpenAI continuous conversation does not support request field(s): "
            f"{', '.join(incompatible)}."
        )
    include = request_config.get("include")
    if not isinstance(include, list) or "reasoning.encrypted_content" not in include:
        raise ValueError(
            "OpenAI continuous conversation must include reasoning.encrypted_content."
        )
    reasoning = request_config.get("reasoning")
    if not isinstance(reasoning, dict):
        raise ValueError("OpenAI continuous conversation requires reasoning settings.")
    if reasoning.get("context") != "auto":
        raise ValueError(
            "OpenAI continuous conversation requires reasoning.context=auto."
        )
    if reasoning.get("summary") != "auto":
        raise ValueError(
            "OpenAI continuous conversation requires reasoning.summary=auto."
        )


class OpenAIContinuousConversationRuntimeAdapter:
    strategy = CONTINUOUS_CONVERSATION_RUNTIME_STATE
    provides_continuous_conversation = True

    def __init__(self, *, model_adapter: Any, descriptor: AdapterDescriptor) -> None:
        self._model_adapter = model_adapter
        self.descriptor = descriptor

    def initial_state(self) -> RuntimeState:
        return RuntimeState(
            adapter_id=self.descriptor.adapter_id,
            strategy=self.strategy,
            payload={"input_items": []},
        )

    def buffer_inputs(
        self, state: RuntimeState, messages: list[Message]
    ) -> RuntimeState:
        state.validate_for(
            adapter_id=self.descriptor.adapter_id, strategy=self.strategy
        )
        items = runtime_payload_items(state, "input_items")
        items.extend(message.model_dump() for message in messages)
        return replace_runtime_payload(state, {"input_items": items})

    def invoke_turn(self, request: ModelTurnRequest) -> ModelTurnResult:
        request.previous_state.validate_for(
            adapter_id=self.descriptor.adapter_id, strategy=self.strategy
        )
        validate_continuous_conversation_request(request.request_config)
        input_items = runtime_payload_items(request.previous_state, "input_items")
        input_items.extend(message.model_dump() for message in request.new_messages)
        model_request = ModelRequest(
            messages=[
                Message(role="system", content=request.system_prompt),
                *request.new_messages,
            ],
            request_config=dict(request.request_config),
            native_input=input_items,
        )
        response = self._model_adapter.invoke(model_request)
        response = response.model_copy(
            update={
                "reasoning_text": extract_responses_reasoning_summary(
                    response.raw_response
                )
            }
        )
        output_items = serialize_response_output_items(response)
        all_items = [*input_items, *output_items]
        history_before = len(all_items)
        next_items = prune_after_latest_compaction(all_items)
        compaction_count = sum(
            item.get("type") == "compaction" for item in output_items
        )
        descriptors = [sanitized_item_descriptor(item) for item in input_items]
        return ModelTurnResult(
            response=response,
            state=replace_runtime_payload(
                request.previous_state, {"input_items": next_items}
            ),
            sanitized_request={
                "instructions_present": True,
                "input_items": descriptors,
                "settings": sanitize_settings(request.request_config),
            },
            transition=StateTransitionTelemetry(
                input_items_sent=len(input_items),
                compaction_items_returned=compaction_count,
                history_items_before_prune=history_before,
                history_items_after_prune=len(next_items),
                sanitized_items=descriptors,
            ),
            action_state={
                "input_items_sent": len(input_items),
                "compaction_items_returned": compaction_count,
                "history_items_before_prune": history_before,
                "history_items_after_prune": len(next_items),
            },
        )
