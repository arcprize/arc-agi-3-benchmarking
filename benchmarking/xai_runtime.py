"""xAI Responses replay and transactional native context compaction."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .exceptions import EmptyResponseError
from .openai_runtime import sanitized_item_descriptor
from .runtime_models import (
    Message,
    ModelRequest,
    ModelResponse,
    NormalizedUsage,
    _normalize_responses_usage,
    normalize_responses_response,
)
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
    runtime_payload_items,
    sanitize_settings,
    unwind_runtime_state_items,
)


class XAICompactionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["native"] = "native"
    trigger_tokens: int = Field(default=175_000, gt=0, strict=True)


def validate_continuous_conversation_request(config: dict[str, Any]) -> None:
    if config.get("store") is not False:
        raise ValueError("xAI continuous conversation requires store=false.")
    if config.get("stream", False) is not False:
        raise ValueError("xAI continuous conversation does not support streaming.")
    if config.get("background", False) is not False:
        raise ValueError(
            "xAI continuous conversation does not support background mode."
        )
    incompatible = sorted(
        {
            "input",
            "instructions",
            "previous_response_id",
            "conversation",
            "context_management",
            "compact_threshold",
            "extra_body",
            "extra_query",
            "tools",
            "tool_choice",
        }.intersection(config)
    )
    if incompatible:
        raise ValueError(
            "xAI continuous conversation does not support request field(s): "
            + ", ".join(incompatible)
            + ". Configure native compaction in runtime.compaction."
        )
    if config.get("truncation", "disabled") != "disabled":
        raise ValueError("xAI continuous conversation cannot truncate native history.")
    include = config.get("include")
    if not isinstance(include, list) or "reasoning.encrypted_content" not in include:
        raise ValueError(
            "xAI continuous conversation must include reasoning.encrypted_content."
        )
    reasoning = config.get("reasoning", {})
    if not isinstance(reasoning, dict) or "context" in reasoning:
        raise ValueError("xAI does not support OpenAI reasoning.context settings.")


def native_mapping(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude_unset=True, warnings=False)
    if not isinstance(value, dict):
        raise ValueError("xAI native response must be a mapping.")
    return deepcopy(value)


def _invalid_response(message: str, raw: dict[str, Any]) -> EmptyResponseError:
    diagnostic = sanitize_settings(raw)
    if diagnostic.get("error"):
        diagnostic["error"] = {"present": True}
    return EmptyResponseError(
        message,
        response=diagnostic,
        usage=normalize_xai_usage(raw.get("usage")),
    )


def normalize_xai_usage(value: Any) -> NormalizedUsage:
    normalized = _normalize_responses_usage(value)
    if isinstance(value, dict):
        for key in ("cost", "cost_details"):
            if value.get(key) is not None:
                normalized[key] = value[key]
    cost_ticks = (
        value.get("cost_in_usd_ticks")
        if isinstance(value, dict)
        else getattr(value, "cost_in_usd_ticks", None)
    )
    if isinstance(cost_ticks, int) and not isinstance(cost_ticks, bool):
        normalized["cost"] = cost_ticks / 10_000_000_000
    return NormalizedUsage(**normalized)


def normalize_xai_response(value: Any, *, compaction: bool = False) -> ModelResponse:
    raw = native_mapping(value)
    output = raw.get("output")
    if (
        not isinstance(output, list)
        or not output
        or not all(isinstance(item, dict) for item in output)
    ):
        raise _invalid_response("xAI returned invalid native output items.", raw)
    if compaction:
        valid = (
            raw.get("object") == "response.compaction"
            and raw.get("status", "completed") == "completed"
            and not raw.get("error")
            and len(output) == 1
            and output[0].get("type") == "compaction"
            and isinstance(output[0].get("encrypted_content"), str)
            and bool(output[0]["encrypted_content"].strip())
        )
        if not valid:
            raise _invalid_response(
                "xAI returned invalid native compaction output.", raw
            )
        return ModelResponse(
            output_text="",
            usage=normalize_xai_usage(raw.get("usage")),
            raw_response=raw,
            response_status="completed",
        )
    if raw.get("status") != "completed" or raw.get("error"):
        raise _invalid_response("xAI action response did not complete.", raw)
    for item in output:
        if item.get("status", "completed") != "completed":
            raise _invalid_response("xAI returned an incomplete output item.", raw)
        if item.get("type") == "reasoning":
            if (
                not isinstance(item.get("encrypted_content"), str)
                or not item["encrypted_content"].strip()
            ):
                raise _invalid_response(
                    "xAI reasoning is missing encrypted replay state.", raw
                )
        elif item.get("type") == "message":
            content = item.get("content")
            if (
                item.get("role") != "assistant"
                or not isinstance(content, list)
                or not content
                or any(
                    not isinstance(part, dict)
                    or part.get("type") != "output_text"
                    or not isinstance(part.get("text"), str)
                    for part in content
                )
            ):
                raise _invalid_response(
                    "xAI returned invalid or refused action content.", raw
                )
        else:
            raise _invalid_response(
                "xAI returned unsupported action output items.", raw
            )
    normalization_input = {
        key: value for key, value in raw.items() if key != "output_text"
    }
    try:
        response = normalize_responses_response(normalization_input)
    except EmptyResponseError:
        raise _invalid_response("xAI returned no visible action text.", raw) from None
    if not response.output_text.strip():
        raise _invalid_response("xAI returned no visible action text.", raw)
    return response.model_copy(
        update={
            "response_status": "completed",
            "usage": normalize_xai_usage(raw.get("usage")),
            "raw_response": raw,
        }
    )


class XAIResponsesAdapter:
    def __init__(self, client: Any) -> None:
        self._client = client

    def invoke(self, request: ModelRequest) -> ModelResponse:
        validate_continuous_conversation_request(request.request_config)
        items = (
            request.native_input
            if request.native_input is not None
            else [message.model_dump() for message in request.messages]
        )
        raw = self._client.responses.create(
            input=deepcopy(items), **deepcopy(request.request_config)
        )
        return normalize_xai_response(raw)

    def compact(
        self, *, model: str, input_items: list[dict[str, Any]]
    ) -> ModelResponse:
        raw = self._client.responses.compact(model=model, input=deepcopy(input_items))
        return normalize_xai_response(raw, compaction=True)


def readable_messages(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages = []
    for item in items:
        if item.get("type") == "compaction":
            messages.append(
                {"role": "assistant", "content": "[xAI native compacted history]"}
            )
        elif item.get("role") in {"system", "user", "assistant"}:
            content = item.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    part["text"]
                    for part in content
                    if part.get("type") in {"input_text", "output_text"}
                    and isinstance(part.get("text"), str)
                )
            messages.append({"role": item["role"], "content": content})
    return messages


class XAIContinuousConversationRuntimeAdapter:
    strategy = CONTINUOUS_CONVERSATION_RUNTIME_STATE
    provides_continuous_conversation = True

    def __init__(
        self,
        *,
        model_adapter: Any,
        descriptor: AdapterDescriptor,
        compaction: dict[str, Any] | None = None,
    ) -> None:
        self._model_adapter = model_adapter
        self.descriptor = descriptor
        self.compaction_policy = (
            XAICompactionPolicy.model_validate(compaction)
            if compaction is not None
            else None
        )

    def initial_state(self) -> RuntimeState:
        return RuntimeState(
            adapter_id=self.descriptor.adapter_id,
            strategy=self.strategy,
            payload={"input_items": [], "context_tokens": 0},
        )

    def _items(self, state: RuntimeState) -> list[dict[str, Any]]:
        state.validate_for(
            adapter_id=self.descriptor.adapter_id, strategy=self.strategy
        )
        return deepcopy(runtime_payload_items(state, "input_items"))

    def buffer_inputs(
        self, state: RuntimeState, messages: list[Message]
    ) -> RuntimeState:
        items = self._items(state)
        items.extend(message.model_dump() for message in messages)
        return replace_runtime_payload(state, {**state.payload, "input_items": items})

    def invoke_turn(self, request: ModelTurnRequest) -> ModelTurnResult:
        validate_continuous_conversation_request(request.request_config)
        state = request.previous_state
        history = self._items(state)
        model = request.request_config["model"]
        for key, value in (("model", model), ("system_prompt", request.system_prompt)):
            if key in state.payload and state.payload[key] != value:
                raise ValueError(f"xAI continuous conversation cannot change {key}.")
        if "system_prompt" not in state.payload:
            history.insert(0, {"role": "system", "content": request.system_prompt})
        completed_end = state.accepted_turns[-1].end_item if state.accepted_turns else 0
        history_count = len(history)
        compaction_usage = NormalizedUsage()
        compaction_count = 0
        compacting = False
        try:
            if (
                self.compaction_policy is not None
                and completed_end
                and state.payload.get("context_tokens", 0)
                >= self.compaction_policy.trigger_tokens
            ):
                compacting = True
                compacted = self._model_adapter.compact(
                    model=model, input_items=history[:completed_end]
                )
                compaction_usage = compacted.usage
                compacting = False
                compaction_count = 1
                history = [
                    *native_mapping(compacted.raw_response)["output"],
                    *history[completed_end:],
                ]
            items = [
                *history,
                *(message.model_dump() for message in request.new_messages),
            ]
            response = self._model_adapter.invoke(
                ModelRequest(
                    messages=request.new_messages,
                    native_input=items,
                    request_config=deepcopy(request.request_config),
                )
            )
        except EmptyResponseError as exc:
            usage = (
                exc.usage
                if isinstance(exc.usage, NormalizedUsage)
                else NormalizedUsage()
            )
            raise EmptyResponseError(
                "xAI compaction or action request did not complete.",
                response=sanitize_settings(exc.response),
                usage=compaction_usage + usage,
                native_compaction_usage=(
                    compaction_usage + usage if compacting else compaction_usage
                )
                if compacting or compaction_count
                else None,
            ) from None
        except Exception as exc:
            raise EmptyResponseError(
                f"xAI request failed ({type(exc).__name__}).",
                response={
                    "provider_error": {
                        "type": type(exc).__name__,
                        "status_code": getattr(exc, "status_code", None),
                    }
                },
                usage=compaction_usage,
                native_compaction_usage=compaction_usage if compaction_count else None,
            ) from None
        next_items = [*items, *native_mapping(response.raw_response)["output"]]
        candidate = replace_runtime_payload(
            state,
            {
                "input_items": next_items,
                "model": model,
                "system_prompt": request.system_prompt,
                "context_tokens": response.usage.total_tokens,
            },
            accepted_turns=[] if compaction_count else state.accepted_turns,
        )
        response = response.model_copy(
            update={"usage": compaction_usage + response.usage}
        )
        next_state = append_accepted_turn(
            state=candidate,
            payload=candidate.payload,
            start_item=len(history),
            end_item=len(next_items),
            request_messages=request.new_messages,
            response=response,
        )
        descriptors = [sanitized_item_descriptor(item) for item in items]
        counts = {
            "input_items_sent": len(items),
            "compaction_items_returned": compaction_count,
            "history_items_before_prune": history_count
            + len(request.new_messages)
            + len(next_items)
            - len(items),
            "history_items_after_prune": len(next_items),
        }
        action_state: dict[str, Any] = dict(counts)
        if compaction_count:
            action_state["native_compaction"] = {
                "strategy": "native",
                "usage": compaction_usage.model_dump(),
            }
        return ModelTurnResult(
            response=response,
            state=next_state,
            sanitized_request={
                "input_items": descriptors,
                "settings": sanitize_settings(request.request_config),
            },
            transition=StateTransitionTelemetry(**counts, sanitized_items=descriptors),
            action_state=action_state,
            readable_request_messages=readable_messages(items),
        )

    def unwind_latest_accepted_turn(
        self, state: RuntimeState
    ) -> CompactionUnwindResult | None:
        self._items(state)
        return unwind_runtime_state_items(state, payload_key="input_items")

    def rebuild_after_compaction(
        self, summary_message: Message, retained_turns: list[CompactionUnwindResult]
    ) -> RuntimeState:
        raise ValueError("xAI requires native compaction, not harness text summaries.")
