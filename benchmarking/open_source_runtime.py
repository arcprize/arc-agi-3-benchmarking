"""Native reasoning continuity for OpenAI-compatible open-weight endpoints."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from openai import APIConnectionError, APIError

from .exceptions import (
    ContextOverflowError,
    InvalidProviderResponseError,
    TransientProviderError,
)
from .runtime_models import Message, ModelRequest, ModelResponse, NormalizedUsage
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

OPEN_SOURCE_ADAPTER_ID = "open_source.chat_completions.v1"
REASONING_REPLAY_MODES = frozenset(
    {"reasoning_content", "reasoning", "reasoning_aliases"}
)
_CONTEXT_MARKERS = (
    "context_length_exceeded",
    "context length",
    "context window",
    "maximum context",
    "maximum number of tokens",
    "too many tokens",
    "maximum input tokens",
    "input token count exceeds",
    "max_seq_len",
)


def validate_reasoning_replay(value: Any) -> str:
    if not isinstance(value, str) or value not in REASONING_REPLAY_MODES:
        raise ValueError(
            "runtime.reasoning_replay must be reasoning_content, reasoning, or "
            "reasoning_aliases."
        )
    return value


def validate_open_source_request(request: dict[str, Any]) -> None:
    extra_body = request.get("extra_body", {})
    if not isinstance(extra_body, dict):
        raise ValueError("Open-source request.extra_body must be a mapping.")
    forbidden = {
        "messages",
        "input",
        "instructions",
        "previous_response_id",
        "conversation",
        "previous_interaction_id",
        "context_management",
        "compaction",
        "background",
        "tools",
        "tool_choice",
        "functions",
        "function_call",
        "response_format",
        "stop",
        "max_output_tokens",
        "guided_json",
        "guided_regex",
        "guided_choice",
        "guided_grammar",
        "structured_outputs",
    }
    invalid = forbidden.intersection(request) | forbidden.intersection(extra_body)
    if invalid:
        raise ValueError(
            "Open-source continuous conversation requires plain-text actions and "
            f"harness-owned history; unsupported request fields: {', '.join(sorted(invalid))}."
        )
    for field in (
        "model",
        "stream",
        "stream_options",
        "n",
        "max_tokens",
        "max_completion_tokens",
    ):
        if field in extra_body:
            raise ValueError(f"Set request.{field} directly, not in extra_body.")
    if type(request.get("n", 1)) is not int or request.get("n", 1) != 1:
        raise ValueError("Open-source continuous conversation requires n=1.")
    if "stream" in request and not isinstance(request["stream"], bool):
        raise ValueError("request.stream must be a boolean.")
    stream_options = request.get("stream_options")
    if "stream_options" in request and not isinstance(stream_options, dict):
        raise ValueError("request.stream_options must be a mapping.")
    if (
        request.get("stream")
        and isinstance(stream_options, dict)
        and "include_usage" in stream_options
        and stream_options["include_usage"] is not True
    ):
        raise ValueError(
            "Open-source streaming requests require stream_options.include_usage=true."
        )
    if request.get("store") not in (None, False) or extra_body.get("store") not in (
        None,
        False,
    ):
        raise ValueError("Open-source continuous conversation cannot enable store.")
    limits = [
        field for field in ("max_tokens", "max_completion_tokens") if field in request
    ]
    if (
        len(limits) != 1
        or type(request[limits[0]]) is not int
        or request[limits[0]] <= 0
    ):
        raise ValueError(
            "Set exactly one positive request.max_tokens or max_completion_tokens."
        )


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return deepcopy(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump(mode="json", exclude_unset=True))
    raise TypeError("Expected a Chat Completions mapping or SDK response.")


def _usage(value: dict[str, Any] | None) -> NormalizedUsage:
    usage = value or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    input_tokens = usage.get("prompt_tokens") or 0
    output_tokens = usage.get("completion_tokens") or 0
    return NormalizedUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=usage.get("total_tokens") or input_tokens + output_tokens,
        reasoning_tokens=completion_details.get("reasoning_tokens") or 0,
        cached_tokens=prompt_details.get(
            "cached_tokens", usage.get("prompt_cache_hit_tokens")
        )
        or 0,
        cache_write_tokens=prompt_details.get("cache_write_tokens") or 0,
        cost=usage.get("cost") or 0.0,
        cost_details=usage.get("cost_details") or {},
    )


def _diagnostic_usage(value: Any) -> NormalizedUsage:
    if not isinstance(value, dict):
        return NormalizedUsage()
    try:
        return _usage(value)
    except (TypeError, ValueError):
        return NormalizedUsage()


def _invalid(message: str, response: dict[str, Any]) -> InvalidProviderResponseError:
    diagnostic = {key: response[key] for key in ("id", "usage") if key in response}
    diagnostic["choices"] = [
        {
            "finish_reason": choice.get("finish_reason"),
            "message": {
                key: value
                for key, value in (choice.get("message") or {}).items()
                if key
                in {"role", "content", "reasoning_content", "reasoning", "refusal"}
                and isinstance(value, str)
            },
        }
        for choice in response.get("choices") or []
    ]
    return InvalidProviderResponseError(
        message,
        response=sanitize_settings(diagnostic),
        usage=_diagnostic_usage(response.get("usage")),
    )


def _required_usage(response: dict[str, Any]) -> NormalizedUsage:
    raw_usage = response.get("usage")
    if not isinstance(raw_usage, dict):
        raise _invalid("Chat completion did not return valid token usage.", response)
    try:
        usage = _usage(raw_usage)
    except (TypeError, ValueError) as exc:
        raise _invalid(
            "Chat completion did not return valid token usage.", response
        ) from exc
    if min(usage.input_tokens, usage.output_tokens, usage.total_tokens) <= 0:
        raise _invalid("Chat completion did not return valid token usage.", response)
    return usage


def _classified_provider_error(exc: APIError) -> Exception | None:
    if isinstance(exc, APIConnectionError):
        return TransientProviderError("Chat Completions connection failed.")
    details = f"{exc.message} {exc.body}".lower()
    status_code = getattr(exc, "status_code", None)
    if status_code in (None, 400, 413, 422) and any(
        marker in details for marker in _CONTEXT_MARKERS
    ):
        return ContextOverflowError("Chat Completions context capacity exceeded.")
    if status_code in {408, 409, 429} or (
        isinstance(status_code, int) and status_code >= 500
    ):
        return TransientProviderError(f"Chat Completions HTTP {status_code}.")
    return None


def normalize_open_source_response(raw: dict[str, Any]) -> ModelResponse:
    choices = raw.get("choices") or []
    if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
        raise _invalid("Chat completion did not finish with one stopped choice.", raw)
    message = choices[0].get("message") or {}
    if (
        message.get("refusal")
        or message.get("tool_calls")
        or message.get("function_call")
    ):
        raise _invalid(
            "Chat completion returned a refusal or unsupported tool call.", raw
        )
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise _invalid("Chat completion returned no final answer text.", raw)
    reasoning_fields = [
        message.get(field) for field in ("reasoning_content", "reasoning")
    ]
    if any(
        value is not None and not isinstance(value, str) for value in reasoning_fields
    ):
        raise _invalid("Chat completion returned non-text reasoning.", raw)
    reasoning_values = [value for value in reasoning_fields if value]
    if len(set(reasoning_values)) > 1:
        raise _invalid("Chat completion returned conflicting reasoning fields.", raw)
    if any(
        message.get(field)
        for field in (
            "reasoning_details",
            "encrypted_content",
            "signature",
            "thought_signature",
        )
    ):
        raise _invalid(
            "Structured or opaque reasoning_details are not supported by this adapter.",
            raw,
        )
    return ModelResponse(
        output_text=content,
        reasoning_text=reasoning_values[0]
        if reasoning_values
        else next((value for value in reasoning_fields if value is not None), None),
        usage=_required_usage(raw),
        raw_response=raw,
        response_status="completed",
        response_id=raw.get("id"),
    )


class OpenSourceChatCompletionsAdapter:
    """Transport that retains native reasoning and rejects unfinished outputs."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def invoke(self, request: ModelRequest) -> ModelResponse:
        validate_open_source_request(request.request_config)
        kwargs = deepcopy(request.request_config)
        kwargs["messages"] = (
            deepcopy(request.native_input)
            if request.native_input is not None
            else [message.model_dump() for message in request.messages]
        )
        if kwargs.get("stream"):
            kwargs["stream_options"] = {
                **(kwargs.get("stream_options") or {}),
                "include_usage": True,
            }
        try:
            raw = self._client.chat.completions.create(**kwargs)
            if kwargs.get("stream"):
                return self._consume_stream(raw)
            return normalize_open_source_response(_mapping(raw))
        except APIError as exc:
            classified = _classified_provider_error(exc)
            if classified is not None:
                raise classified from exc
            raise

    @staticmethod
    def _consume_stream(stream: Any) -> ModelResponse:
        message: dict[str, Any] = {"role": "assistant", "content": ""}
        choice: dict[str, Any] = {"message": message, "finish_reason": None}
        response: dict[str, Any] = {"choices": [choice]}
        try:
            for event in stream:
                chunk = _mapping(event)
                if chunk.get("usage") is not None:
                    response["usage"] = chunk["usage"]
                if chunk.get("id"):
                    response["id"] = chunk["id"]
                for part in chunk.get("choices") or []:
                    if part.get("index", 0) != 0 or choice["finish_reason"] is not None:
                        raise _invalid(
                            "Unexpected choice or data after stream completion.",
                            response,
                        )
                    delta = part.get("delta") or {}
                    for field in (
                        "content",
                        "reasoning_content",
                        "reasoning",
                        "refusal",
                    ):
                        fragment = delta.get(field)
                        if fragment is not None:
                            if not isinstance(fragment, str):
                                raise _invalid(
                                    "Non-text Chat Completions stream delta.", response
                                )
                            message[field] = message.get(field, "") + fragment
                    for field in (
                        "tool_calls",
                        "function_call",
                        "reasoning_details",
                        "encrypted_content",
                        "signature",
                        "thought_signature",
                    ):
                        if delta.get(field):
                            message[field] = delta[field]
                    if part.get("finish_reason") is not None:
                        choice["finish_reason"] = part["finish_reason"]
        except InvalidProviderResponseError:
            raise
        except APIError as exc:
            classified = _classified_provider_error(exc)
            if classified is not None:
                raise classified from exc
            raise _invalid(
                "Chat Completions stream returned a provider error.", response
            ) from exc
        except Exception as exc:
            raise _invalid(
                "Chat Completions stream interrupted before completion.", response
            ) from exc
        finally:
            stream.close()
        return normalize_open_source_response(response)


class OpenSourceContinuousConversationRuntimeAdapter:
    strategy = CONTINUOUS_CONVERSATION_RUNTIME_STATE
    provides_continuous_conversation = True

    def __init__(
        self,
        *,
        model_adapter: Any,
        descriptor: AdapterDescriptor,
        reasoning_replay: str = "reasoning_content",
    ) -> None:
        self._model_adapter = model_adapter
        self.descriptor = descriptor
        self.reasoning_replay = validate_reasoning_replay(reasoning_replay)

    def initial_state(self) -> RuntimeState:
        return RuntimeState(
            adapter_id=self.descriptor.adapter_id,
            strategy=self.strategy,
            payload={
                "messages": [],
                "pending_messages": [],
                "reasoning_replay": self.reasoning_replay,
            },
        )

    def _payload(self, state: RuntimeState) -> dict[str, Any]:
        state.validate_for(
            adapter_id=self.descriptor.adapter_id, strategy=self.strategy
        )
        if state.payload.get("reasoning_replay") != self.reasoning_replay:
            raise ValueError(
                "Runtime state reasoning_replay does not match the selected adapter."
            )
        runtime_payload_items(state, "messages")
        runtime_payload_items(state, "pending_messages")
        return deepcopy(state.payload)

    def buffer_inputs(
        self, state: RuntimeState, messages: list[Message]
    ) -> RuntimeState:
        payload = self._payload(state)
        if any(message.role != "user" for message in messages):
            raise ValueError(
                "Open-source continuous conversation accepts user inputs only."
            )
        payload["pending_messages"].extend(message.model_dump() for message in messages)
        return replace_runtime_payload(state, payload)

    def split_pending_inputs(
        self, state: RuntimeState
    ) -> tuple[RuntimeState, list[Message]]:
        payload = self._payload(state)
        pending = [
            Message.model_validate(message) for message in payload["pending_messages"]
        ]
        payload["pending_messages"] = []
        return replace_runtime_payload(state, payload), pending

    def _replay_message(self, message: dict[str, Any]) -> dict[str, Any]:
        replay = {"role": message["role"], "content": message["content"]}
        reasoning = message.get("reasoning_content")
        if reasoning is not None:
            if self.reasoning_replay == "reasoning_aliases":
                replay["reasoning_content"] = reasoning
                replay["reasoning"] = reasoning
            else:
                replay[self.reasoning_replay] = reasoning
        return replay

    def invoke_turn(self, request: ModelTurnRequest) -> ModelTurnResult:
        validate_open_source_request(request.request_config)
        payload = self._payload(
            self.buffer_inputs(request.previous_state, request.new_messages)
        )
        history = payload["messages"]
        turn_start = len(history)
        pending = payload["pending_messages"]
        input_messages = [*history, *pending]
        wire_messages = [
            {"role": "system", "content": request.system_prompt},
            *(self._replay_message(message) for message in input_messages),
        ]
        response = self._model_adapter.invoke(
            ModelRequest(
                messages=[
                    Message(role="system", content=request.system_prompt),
                    *request.new_messages,
                ],
                request_config=deepcopy(request.request_config),
                native_input=wire_messages,
            )
        )
        if response.response_status != "completed":
            raise InvalidProviderResponseError(
                "Open-source response was not completed.", usage=response.usage
            )
        assistant: dict[str, Any] = {
            "role": "assistant",
            "content": response.output_text,
        }
        if response.reasoning_text is not None:
            assistant["reasoning_content"] = response.reasoning_text
        payload["messages"] = [*input_messages, assistant]
        payload["pending_messages"] = []
        descriptors = [
            {
                "role": message["role"],
                "reasoning_present": bool(message.get("reasoning_content")),
            }
            for message in input_messages
        ]
        transition = StateTransitionTelemetry(
            input_items_sent=len(input_messages),
            history_items_before_prune=len(payload["messages"]),
            history_items_after_prune=len(payload["messages"]),
            sanitized_items=descriptors,
        )
        return ModelTurnResult(
            response=response,
            state=append_accepted_turn(
                state=request.previous_state,
                payload=payload,
                start_item=turn_start,
                end_item=len(payload["messages"]),
                request_messages=[
                    Message.model_validate(message) for message in pending
                ],
                response=response,
            ),
            sanitized_request={
                "input_items": descriptors,
                "reasoning_replay": self.reasoning_replay,
                "settings": sanitize_settings(request.request_config),
            },
            transition=transition,
            action_state={
                "reasoning_replay": self.reasoning_replay,
                "input_items_sent": len(input_messages),
            },
            readable_request_messages=sanitize_settings(wire_messages),
        )

    def unwind_latest_accepted_turn(
        self, state: RuntimeState
    ) -> CompactionUnwindResult | None:
        self._payload(state)
        return unwind_runtime_state_items(state, payload_key="messages")

    def rebuild_after_compaction(
        self, summary_message: Message, retained_turns: list[CompactionUnwindResult]
    ) -> RuntimeState:
        state = self.initial_state()
        state.payload["messages"] = [summary_message.model_dump()]
        return restore_unwound_runtime_state_items(
            state, payload_key="messages", retained_turns=retained_turns
        )
