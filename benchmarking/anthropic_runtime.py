"""Anthropic-native conversation replay, compaction, and response handling."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .exceptions import EmptyResponseError, InvalidProviderResponseError
from .runtime_models import Message, ModelRequest, ModelResponse, NormalizedUsage
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

COMPACTION_BETA = "compact-2026-01-12"
COMPACTION_TYPE = "compact_20260112"


def native_mapping(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude_unset=True)
    elif not isinstance(value, dict) and hasattr(value, "__dict__"):
        value = vars(value)
    if not isinstance(value, dict):
        raise TypeError("Anthropic native values must serialize to mappings.")
    return deepcopy(value)


def serialize_replay_content(content: list[Any]) -> list[dict[str, Any]]:
    blocks = [native_mapping(block) for block in content]
    for block in blocks:
        if block.get("type") == "text":
            block.pop("parsed_output", None)
        elif block.get("type") == "compaction":
            if block.get("encrypted_content") is None:
                block.pop("encrypted_content", None)
    return blocks


def validate_continuous_conversation_request(
    request_config: dict[str, Any],
) -> None:
    incompatible = sorted(
        set(request_config).intersection(
            {
                "store",
                "background",
                "conversation",
                "previous_response_id",
                "include",
                "reasoning",
                "extra_body",
                "max_output_tokens",
                "max_completion_tokens",
                "fallbacks",
                "tools",
                "tool_choice",
                "mcp_servers",
                "container",
                "messages",
                "system",
            }
        )
    )
    if incompatible:
        raise ValueError(
            "Anthropic continuous conversation does not support request field(s): "
            + ", ".join(incompatible)
            + "."
        )
    model = request_config.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Anthropic continuous conversation requires a model.")
    max_tokens = request_config.get("max_tokens")
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError(
            "Anthropic continuous conversation requires positive max_tokens."
        )
    thinking = request_config.get("thinking")
    if not isinstance(thinking, dict) or thinking.get("type") != "adaptive":
        raise ValueError(
            "Anthropic continuous conversation requires adaptive thinking."
        )
    if thinking.get("display") != "summarized":
        raise ValueError(
            "Anthropic continuous conversation requires thinking.display='summarized'."
        )
    betas = request_config.get("betas", [])
    if not isinstance(betas, list) or not all(isinstance(beta, str) for beta in betas):
        raise ValueError("Anthropic request.betas must be a list of strings.")
    if "context_management" not in request_config:
        return
    context = request_config["context_management"]
    edits = context.get("edits") if isinstance(context, dict) else None
    if (
        not isinstance(edits, list)
        or len(edits) != 1
        or not isinstance(edits[0], dict)
        or edits[0].get("type") != COMPACTION_TYPE
    ):
        raise ValueError(
            "Anthropic compaction requires exactly one compact_20260112 edit."
        )
    if COMPACTION_BETA not in betas:
        raise ValueError("Anthropic compaction requires beta compact-2026-01-12.")
    edit = edits[0]
    if edit.get("pause_after_compaction", False) is not False:
        raise ValueError("Anthropic compaction requires pause_after_compaction=false.")
    trigger = edit.get("trigger")
    if trigger is not None and (
        not isinstance(trigger, dict)
        or trigger.get("type") != "input_tokens"
        or type(trigger.get("value")) is not int
        or trigger["value"] < 50_000
    ):
        raise ValueError(
            "Anthropic compaction requires an input_tokens trigger of at least 50000."
        )


def normalize_native_usage(value: Any) -> NormalizedUsage:
    if value is None:
        return NormalizedUsage()
    usage = native_mapping(value)
    iterations = usage.get("iterations")
    entries = iterations if isinstance(iterations, list) and iterations else [usage]
    total = NormalizedUsage()
    compaction_thinking_tokens = 0
    for entry in entries:
        item = native_mapping(entry)
        cached_tokens = item.get("cache_read_input_tokens") or 0
        cache_write_tokens = item.get("cache_creation_input_tokens") or 0
        input_tokens = (
            (item.get("input_tokens") or 0) + cached_tokens + cache_write_tokens
        )
        output_tokens = item.get("output_tokens") or 0
        details = native_mapping(item.get("output_tokens_details") or {})
        thinking_tokens = details.get("thinking_tokens") or 0
        if item.get("type") == "compaction":
            compaction_thinking_tokens += thinking_tokens
        total += NormalizedUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            reasoning_tokens=thinking_tokens,
            cached_tokens=cached_tokens,
            cache_write_tokens=cache_write_tokens,
        )
    details = native_mapping(usage.get("output_tokens_details") or {})
    if iterations and details.get("thinking_tokens") is not None:
        total = total.model_copy(
            update={
                "reasoning_tokens": details["thinking_tokens"]
                + compaction_thinking_tokens
            }
        )
    return total


def normalize_native_response(
    value: Any, request_config: dict[str, Any]
) -> ModelResponse:
    raw = native_mapping(value)
    usage = normalize_native_usage(raw.get("usage"))
    stop_reason = raw.get("stop_reason")
    completed = stop_reason == "end_turn" or (
        stop_reason == "stop_sequence"
        and raw.get("stop_sequence") in (request_config.get("stop_sequences") or [])
    )
    if not completed:
        raise InvalidProviderResponseError(
            "Anthropic response did not complete an action turn.",
            response=sanitize_settings(raw),
            usage=usage,
        )
    blocks = [native_mapping(block) for block in raw.get("content", []) or []]
    if any(
        block.get("type") in {"tool_use", "server_tool_use", "mcp_tool_use"}
        for block in blocks
    ):
        raise InvalidProviderResponseError(
            "Anthropic action turns cannot contain tool calls.",
            response=sanitize_settings(raw),
            usage=usage,
        )
    text = "".join(
        block.get("text", "") for block in blocks if block.get("type") == "text"
    )
    if not text.strip():
        raise EmptyResponseError(
            "Anthropic response contained no action text.",
            response=sanitize_settings(raw),
            usage=usage,
        )
    reasoning = "\n".join(
        block["thinking"]
        for block in blocks
        if block.get("type") == "thinking" and block.get("thinking")
    )
    raw["content"] = blocks
    return ModelResponse(
        output_text=text,
        reasoning_text=reasoning or None,
        usage=usage,
        raw_response=raw,
    )


def prune_after_latest_compaction(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for message_index in range(len(messages) - 1, -1, -1):
        message = messages[message_index]
        blocks = message.get("content")
        if message.get("role") != "assistant" or not isinstance(blocks, list):
            continue
        for block_index in range(len(blocks) - 1, -1, -1):
            block = blocks[block_index]
            summary = block.get("content")
            if (
                block.get("type") == "compaction"
                and isinstance(summary, str)
                and summary.strip()
            ):
                return [
                    {**message, "content": blocks[block_index:]},
                    *messages[message_index + 1 :],
                ]
    return messages


def readable_messages(
    system_prompt: str, messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    readable: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            parts = []
            for block in content:
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "thinking" and block.get("thinking"):
                    parts.append(
                        f"<reasoning_summary>\n{block['thinking']}\n</reasoning_summary>"
                    )
                elif block.get("type") == "compaction" and block.get("content"):
                    parts.append(
                        f"<compaction_summary>\n{block['content']}\n</compaction_summary>"
                    )
            content = "\n".join(parts)
        readable.append({"role": message["role"], "content": content})
    return readable


class AnthropicContinuousConversationRuntimeAdapter:
    strategy = CONTINUOUS_CONVERSATION_RUNTIME_STATE
    provides_continuous_conversation = True

    def __init__(self, *, model_adapter: Any, descriptor: AdapterDescriptor) -> None:
        self._model_adapter = model_adapter
        self.descriptor = descriptor

    def initial_state(self) -> RuntimeState:
        return RuntimeState(
            adapter_id=self.descriptor.adapter_id,
            strategy=self.strategy,
            payload={"messages": []},
        )

    def _messages(self, state: RuntimeState) -> list[dict[str, Any]]:
        state.validate_for(
            adapter_id=self.descriptor.adapter_id, strategy=self.strategy
        )
        messages = deepcopy(runtime_payload_items(state, "messages"))
        for message in messages:
            content = message.get("content")
            if message.get("role") not in {"user", "assistant"} or not (
                isinstance(content, str)
                or (
                    isinstance(content, list)
                    and all(isinstance(block, dict) for block in content)
                )
            ):
                raise ValueError(
                    "Anthropic state must contain native user/assistant messages."
                )
        return messages

    def buffer_inputs(
        self, state: RuntimeState, messages: list[Message]
    ) -> RuntimeState:
        native = self._messages(state)
        if any(message.role != "user" for message in messages):
            raise ValueError(
                "Anthropic continuous conversation accepts user inputs only."
            )
        native.extend(message.model_dump() for message in messages)
        return replace_runtime_payload(
            state, {**deepcopy(state.payload), "messages": native}
        )

    def invoke_turn(self, request: ModelTurnRequest) -> ModelTurnResult:
        validate_continuous_conversation_request(request.request_config)
        state = self.buffer_inputs(request.previous_state, request.new_messages)
        model = request.request_config["model"]
        for key, value in (("model", model), ("system_prompt", request.system_prompt)):
            if key in state.payload and state.payload[key] != value:
                raise ValueError(
                    f"Anthropic continuous conversation cannot change {key}."
                )
        messages = self._messages(state)
        try:
            response = self._model_adapter.invoke(
                ModelRequest(
                    messages=[Message(role="system", content=request.system_prompt)],
                    request_config=deepcopy(request.request_config),
                    native_input=deepcopy(messages),
                )
            )
        except EmptyResponseError:
            raise
        except Exception as exc:
            raise InvalidProviderResponseError(
                f"Anthropic request failed ({type(exc).__name__})."
            ) from None
        response = normalize_native_response(
            response.raw_response, request.request_config
        )
        raw = native_mapping(response.raw_response)
        output = {
            "role": "assistant",
            "content": serialize_replay_content(raw["content"]),
        }
        all_messages = [*messages, output]
        next_messages = prune_after_latest_compaction(all_messages)
        compaction_count = sum(
            block.get("type") == "compaction" for block in raw["content"]
        )
        descriptors = [
            {
                "role": message["role"],
                "content_types": (
                    [block.get("type", "unknown") for block in message["content"]]
                    if isinstance(message["content"], list)
                    else ["text"]
                ),
            }
            for message in messages
        ]
        counts = {
            "input_items_sent": len(messages),
            "compaction_items_returned": compaction_count,
            "history_items_before_prune": len(all_messages),
            "history_items_after_prune": len(next_messages),
        }
        return ModelTurnResult(
            response=response,
            state=replace_runtime_payload(
                state,
                {
                    "messages": next_messages,
                    "model": model,
                    "system_prompt": request.system_prompt,
                },
            ),
            sanitized_request={
                "input_items": descriptors,
                "settings": sanitize_settings(request.request_config),
            },
            readable_request_messages=readable_messages(
                request.system_prompt, messages
            ),
            transition=StateTransitionTelemetry(**counts, sanitized_items=descriptors),
            action_state={**counts, "stop_reason": raw.get("stop_reason")},
        )
