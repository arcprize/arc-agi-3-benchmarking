"""DeepSeek thinking-mode continuity through native tool calls."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from openai import APIConnectionError, APIStatusError

from .compaction import SUMMARY_SYSTEM_PROMPT
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

DEEPSEEK_ADAPTER_ID = "deepseek.chat_completions.v1"
ACTION_TOOL_NAME = "submit_action"
SUMMARY_TOOL_NAME = "return_summary"
ACTION_TOOL_RESULT = (
    "The action was accepted. The resulting game state will be provided in the "
    "next user message."
)
ACTION_TOOL = {
    "type": "function",
    "function": {
        "name": ACTION_TOOL_NAME,
        "description": (
            "Submit exactly one available ARC action. Omit x and y for simple "
            "actions; include both integer coordinates for a coordinate action."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action_type": {
                    "type": "string",
                    "description": "One action exactly as listed under Available actions.",
                },
                "x": {"type": "integer", "minimum": 0, "maximum": 63},
                "y": {"type": "integer", "minimum": 0, "maximum": 63},
            },
            "required": ["action_type"],
            "additionalProperties": False,
        },
    },
}
SUMMARY_TOOL = {
    "type": "function",
    "function": {
        "name": SUMMARY_TOOL_NAME,
        "description": "Return the requested continuation summary.",
        "parameters": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": False,
        },
    },
}
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


def validate_deepseek_request(request: dict[str, Any]) -> None:
    extra_body = request.get("extra_body", {})
    if not isinstance(extra_body, dict):
        raise ValueError("DeepSeek request.extra_body must be a mapping.")
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
            "DeepSeek continuous conversation owns tools and history; unsupported "
            f"request fields: {', '.join(sorted(invalid))}."
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
        raise ValueError("DeepSeek continuous conversation requires n=1.")
    if "stream" in request and not isinstance(request["stream"], bool):
        raise ValueError("request.stream must be a boolean.")
    if "stream_options" in request and not isinstance(request["stream_options"], dict):
        raise ValueError("request.stream_options must be a mapping.")
    if request.get("store") not in (None, False) or extra_body.get("store") not in (
        None,
        False,
    ):
        raise ValueError("DeepSeek continuous conversation cannot enable store.")
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


def _invalid(message: str, response: dict[str, Any]) -> InvalidProviderResponseError:
    diagnostic = {key: response[key] for key in ("id", "usage") if key in response}
    diagnostic["choices"] = [
        {
            "finish_reason": choice.get("finish_reason"),
            "message": {
                key: value
                for key, value in (choice.get("message") or {}).items()
                if key in {"role", "content", "reasoning_content", "refusal"}
                and isinstance(value, str)
            },
        }
        for choice in response.get("choices") or []
    ]
    return InvalidProviderResponseError(
        message,
        response=sanitize_settings(diagnostic),
        usage=_usage(response.get("usage")),
    )


def _tool_call_output(message: dict[str, Any], *, expected_tool_name: str) -> str:
    tool_calls = message.get("tool_calls")
    if tool_calls is None:
        raise ValueError("Chat completion did not return a tool call.")
    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
        raise ValueError("Chat completion must return exactly one tool call.")
    tool_call = tool_calls[0]
    if not isinstance(tool_call, dict):
        raise ValueError("Chat completion returned a malformed tool call.")
    function = tool_call.get("function")
    if (
        not isinstance(tool_call.get("id"), str)
        or not tool_call["id"]
        or tool_call.get("type") != "function"
        or not isinstance(function, dict)
        or function.get("name") != expected_tool_name
        or not isinstance(function.get("arguments"), str)
    ):
        raise ValueError("Chat completion returned an unexpected tool call.")
    try:
        arguments = json.loads(function["arguments"])
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Chat completion returned invalid tool arguments.") from exc
    if not isinstance(arguments, dict):
        raise ValueError("Chat completion tool arguments must be an object.")
    if expected_tool_name == ACTION_TOOL_NAME:
        invalid_fields = set(arguments).difference({"action_type", "x", "y"})
        if invalid_fields:
            raise ValueError("Action tool call contains unexpected arguments.")
        action_type = arguments.get("action_type")
        if not isinstance(action_type, str) or not action_type:
            raise ValueError("Action tool call is missing action_type.")
        has_x = "x" in arguments
        has_y = "y" in arguments
        if has_x != has_y:
            raise ValueError("Action tool call must include both x and y.")
        if has_x and any(
            type(arguments[field]) is not int or not 0 <= arguments[field] <= 63
            for field in ("x", "y")
        ):
            raise ValueError("Action tool coordinates must be integers from 0 to 63.")
        return json.dumps({"actions": [arguments]}, separators=(",", ":"))
    if set(arguments) != {"summary"} or not isinstance(arguments["summary"], str):
        raise ValueError("Summary tool call must contain one string summary.")
    if not arguments["summary"].strip():
        raise ValueError("Summary tool call returned an empty summary.")
    return arguments["summary"]


def normalize_deepseek_response(
    raw: dict[str, Any],
    *,
    expected_tool_name: str = ACTION_TOOL_NAME,
) -> ModelResponse:
    choices = raw.get("choices") or []
    if len(choices) != 1:
        raise _invalid("Chat completion did not return exactly one choice.", raw)
    finish_reason = choices[0].get("finish_reason")
    message = choices[0].get("message") or {}
    if message.get("refusal") or message.get("function_call"):
        raise _invalid(
            "Chat completion returned a refusal or legacy function call.", raw
        )
    reasoning = message.get("reasoning_content")
    if reasoning is not None and not isinstance(reasoning, str):
        raise _invalid("Chat completion returned non-text reasoning.", raw)
    if any(
        message.get(field)
        for field in (
            "reasoning",
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
    if finish_reason != "tool_calls":
        raise _invalid("DeepSeek did not finish with a tool call.", raw)
    try:
        output_text = _tool_call_output(message, expected_tool_name=expected_tool_name)
    except ValueError as exc:
        raise _invalid(str(exc), raw) from exc
    return ModelResponse(
        output_text=output_text,
        reasoning_text=reasoning,
        usage=_usage(raw.get("usage")),
        raw_response=raw,
        response_status="completed",
        response_id=raw.get("id"),
    )


class DeepSeekChatCompletionsAdapter:
    """Transport enforcing DeepSeek's thinking-mode tool protocol."""

    def __init__(self, client: Any) -> None:
        self._client = client

    @staticmethod
    def _tool_for_request(request: ModelRequest) -> dict[str, Any]:
        if (
            request.messages
            and request.messages[0].role == "system"
            and request.messages[0].content.startswith(SUMMARY_SYSTEM_PROMPT)
        ):
            return SUMMARY_TOOL
        return ACTION_TOOL

    def invoke(self, request: ModelRequest) -> ModelResponse:
        validate_deepseek_request(request.request_config)
        kwargs = deepcopy(request.request_config)
        kwargs["messages"] = (
            deepcopy(request.native_input)
            if request.native_input is not None
            else [message.model_dump() for message in request.messages]
        )
        tool = self._tool_for_request(request)
        kwargs["tools"] = [deepcopy(tool)]
        expected_tool_name = tool["function"]["name"]
        if kwargs.get("stream"):
            kwargs["stream_options"] = {
                "include_usage": True,
                **(kwargs.get("stream_options") or {}),
            }
        try:
            raw = self._client.chat.completions.create(**kwargs)
            if kwargs.get("stream"):
                return self._consume_stream(
                    raw,
                    expected_tool_name=expected_tool_name,
                )
            return normalize_deepseek_response(
                _mapping(raw),
                expected_tool_name=expected_tool_name,
            )
        except APIConnectionError as exc:
            raise TransientProviderError("Chat Completions connection failed.") from exc
        except APIStatusError as exc:
            details = f"{exc.message} {exc.body}".lower()
            if exc.status_code in {400, 413, 422} and any(
                marker in details for marker in _CONTEXT_MARKERS
            ):
                raise ContextOverflowError(
                    "Chat Completions context capacity exceeded."
                ) from exc
            if exc.status_code in {408, 409, 429} or exc.status_code >= 500:
                raise TransientProviderError(
                    f"Chat Completions HTTP {exc.status_code}."
                ) from exc
            raise

    @staticmethod
    def _consume_stream(
        stream: Any,
        *,
        expected_tool_name: str = ACTION_TOOL_NAME,
    ) -> ModelResponse:
        message: dict[str, Any] = {"role": "assistant", "content": ""}
        choice: dict[str, Any] = {"message": message, "finish_reason": None}
        response: dict[str, Any] = {"choices": [choice]}
        tool_calls: dict[int, dict[str, Any]] = {}
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
                        "refusal",
                    ):
                        fragment = delta.get(field)
                        if fragment is not None:
                            if not isinstance(fragment, str):
                                raise _invalid(
                                    "Non-text Chat Completions stream delta.", response
                                )
                            message[field] = message.get(field, "") + fragment
                    tool_deltas = delta.get("tool_calls") or []
                    for tool_delta in tool_deltas:
                        if not isinstance(tool_delta, dict):
                            raise _invalid(
                                "Malformed tool-call stream delta.", response
                            )
                        index = tool_delta.get("index")
                        if type(index) is not int or index < 0:
                            raise _invalid("Invalid tool-call stream index.", response)
                        target = tool_calls.setdefault(
                            index,
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        for field in ("id", "type"):
                            fragment = tool_delta.get(field)
                            if fragment is not None:
                                if not isinstance(fragment, str):
                                    raise _invalid(
                                        "Non-text tool-call stream delta.", response
                                    )
                                target[field] = fragment
                        function_delta = tool_delta.get("function") or {}
                        if not isinstance(function_delta, dict):
                            raise _invalid("Malformed function stream delta.", response)
                        name = function_delta.get("name")
                        if name is not None:
                            if not isinstance(name, str):
                                raise _invalid(
                                    "Non-text tool name stream delta.", response
                                )
                            target["function"]["name"] += name
                        arguments = function_delta.get("arguments")
                        if arguments is not None:
                            if not isinstance(arguments, str):
                                raise _invalid(
                                    "Non-text tool arguments stream delta.", response
                                )
                            target["function"]["arguments"] += arguments
                    for field in (
                        "function_call",
                        "reasoning",
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
        except Exception as exc:
            raise _invalid(
                "Chat Completions stream interrupted before completion.", response
            ) from exc
        finally:
            stream.close()
        if tool_calls:
            message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
        return normalize_deepseek_response(
            response,
            expected_tool_name=expected_tool_name,
        )


class DeepSeekContinuousConversationRuntimeAdapter:
    strategy = CONTINUOUS_CONVERSATION_RUNTIME_STATE
    provides_continuous_conversation = True

    def __init__(
        self,
        *,
        model_adapter: Any,
        descriptor: AdapterDescriptor,
    ) -> None:
        self._model_adapter = model_adapter
        self.descriptor = descriptor

    def initial_state(self) -> RuntimeState:
        return RuntimeState(
            adapter_id=self.descriptor.adapter_id,
            strategy=self.strategy,
            payload={
                "messages": [],
                "pending_messages": [],
            },
        )

    def _payload(self, state: RuntimeState) -> dict[str, Any]:
        state.validate_for(
            adapter_id=self.descriptor.adapter_id, strategy=self.strategy
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
                "DeepSeek continuous conversation accepts user inputs only."
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
        if message["role"] == "tool":
            replay["tool_call_id"] = message["tool_call_id"]
            return replay
        if message.get("tool_calls") is not None:
            replay["tool_calls"] = deepcopy(message["tool_calls"])
        if "reasoning_content" in message:
            replay["reasoning_content"] = message["reasoning_content"]
        return replay

    def invoke_turn(self, request: ModelTurnRequest) -> ModelTurnResult:
        validate_deepseek_request(request.request_config)
        payload = self._payload(
            self.buffer_inputs(request.previous_state, request.new_messages)
        )
        history = payload["messages"]
        turn_start = len(history)
        pending = payload["pending_messages"]
        input_messages = [*history, *pending]
        system_prompt = request.system_prompt
        if system_prompt.startswith(SUMMARY_SYSTEM_PROMPT):
            system_prompt += (
                " Return the summary by calling return_summary exactly once."
            )
        else:
            system_prompt += (
                " Submit the selected action by calling submit_action exactly once. "
                "Do not return the action only as final text."
            )
        wire_messages = [
            {"role": "system", "content": system_prompt},
            *(self._replay_message(message) for message in input_messages),
        ]
        response = self._model_adapter.invoke(
            ModelRequest(
                messages=[
                    Message(role="system", content=system_prompt),
                    *request.new_messages,
                ],
                request_config=deepcopy(request.request_config),
                native_input=wire_messages,
            )
        )
        if response.response_status != "completed":
            raise InvalidProviderResponseError(
                "DeepSeek response was not completed.", usage=response.usage
            )
        raw = (
            _mapping(response.raw_response) if response.raw_response is not None else {}
        )
        raw_choices = raw.get("choices") or []
        raw_message = raw_choices[0].get("message") if raw_choices else None
        if not isinstance(raw_message, dict) or not raw_message.get("tool_calls"):
            raise InvalidProviderResponseError(
                "DeepSeek response omitted the validated tool call.",
                usage=response.usage,
            )
        tool_call = deepcopy(raw_message["tool_calls"][0])
        assistant: dict[str, Any] = {
            "role": "assistant",
            "content": raw_message.get("content"),
            "tool_calls": [tool_call],
        }
        if "reasoning_content" in raw_message:
            assistant["reasoning_content"] = raw_message["reasoning_content"]
        tool_result = {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": ACTION_TOOL_RESULT,
        }
        accepted_messages = [assistant, tool_result]
        payload["messages"] = [*input_messages, *accepted_messages]
        payload["pending_messages"] = []
        descriptors = [
            {
                "role": message["role"],
                "reasoning_present": bool(message.get("reasoning_content")),
                "tool_call_present": bool(message.get("tool_calls")),
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
                "settings": sanitize_settings(request.request_config),
            },
            transition=transition,
            action_state={
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
