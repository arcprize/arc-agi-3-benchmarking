"""Anthropic-native conversation replay, compaction, and response handling."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Literal

from anthropic import APIError
from pydantic import BaseModel, ConfigDict, Field

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

COMPACTION_BETA = "compact-2026-09-04"


class AnthropicCompactionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["native"]
    trigger_tokens: int = Field(strict=True, gt=0)
    summary_max_output_tokens: int = Field(default=8_192, strict=True, gt=0)


def safe_provider_error_metadata(exc: Exception) -> dict[str, Any]:
    metadata: dict[str, Any] = {"exception_class": type(exc).__name__}
    if not isinstance(exc, APIError):
        return metadata

    body = exc.body if isinstance(exc.body, dict) else {}
    error = body.get("error", body)
    error_type = error.get("type") if isinstance(error, dict) else None
    if isinstance(error_type, str) and re.fullmatch(
        r"[a-z][a-z0-9_]{0,79}", error_type
    ):
        metadata["provider_error_type"] = error_type
    status = getattr(exc, "status_code", None)
    if type(status) is int and 100 <= status <= 599:
        metadata["http_status"] = status
    request_id = getattr(exc, "request_id", None) or body.get("request_id")
    if isinstance(request_id, str) and re.fullmatch(
        r"[A-Za-z0-9_-]{1,200}", request_id
    ):
        metadata["request_id"] = request_id
    return metadata


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
    *,
    compaction_request: bool = False,
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
                "context_management",
                "pause_after_compaction",
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
    if "compact-2026-01-12" in betas:
        raise ValueError(
            "Anthropic threshold compaction is unsupported; use native on-demand compaction."
        )
    output_config = request_config.get("output_config", {})
    if not isinstance(output_config, dict):
        raise ValueError("Anthropic output_config must be a mapping.")
    if compaction_request:
        if request_config.get("compaction") != {"type": "summarize"}:
            raise ValueError("Anthropic compaction requires type='summarize'.")
        if COMPACTION_BETA not in betas:
            raise ValueError(f"Anthropic compaction requires beta {COMPACTION_BETA}.")
        if "stop_sequences" in request_config or "format" in output_config:
            raise ValueError(
                "Anthropic compaction cannot constrain the summary format."
            )
    elif "compaction" in request_config:
        raise ValueError(
            "Configure native compaction in runtime.compaction, not request."
        )
    if "remaining" in (output_config.get("task_budget") or {}):
        raise ValueError(
            "Anthropic native replay does not support task_budget.remaining."
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
    content = raw.get("content", []) or []
    try:
        if not isinstance(content, list):
            raise TypeError("Native content must be a list.")
        blocks = [native_mapping(block) for block in content]
    except (TypeError, ValueError):
        raise InvalidProviderResponseError(
            "Anthropic response contained malformed native content.",
            response=sanitize_settings(raw),
            usage=usage,
        ) from None
    stop_reason = raw.get("stop_reason")
    if "compaction" in request_config:
        if not (
            stop_reason == "compaction"
            and len(blocks) == 1
            and blocks[0].get("type") == "compaction"
            and isinstance(blocks[0].get("content"), str)
            and blocks[0]["content"].strip()
            and isinstance(blocks[0].get("signature"), str)
            and blocks[0]["signature"].strip()
        ):
            raise InvalidProviderResponseError(
                "Anthropic compaction did not return a completed, nonempty signed summary.",
                response=sanitize_settings(raw),
                usage=usage,
            )
        raw["content"] = blocks
        return ModelResponse(
            output_text=blocks[0]["content"], usage=usage, raw_response=raw
        )
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
    if any(
        block.get("type")
        in {"tool_use", "server_tool_use", "mcp_tool_use", "compaction"}
        for block in blocks
    ):
        raise InvalidProviderResponseError(
            "Anthropic action turns cannot contain tool calls or inline compaction.",
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

    def __init__(
        self,
        *,
        model_adapter: Any,
        descriptor: AdapterDescriptor,
        compaction: dict[str, Any] | None = None,
    ) -> None:
        self._model_adapter = model_adapter
        self.descriptor = descriptor
        self._compaction = (
            AnthropicCompactionPolicy.model_validate(compaction)
            if compaction is not None
            else None
        )

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
        compactions = [
            (message_index, block_index, block)
            for message_index, message in enumerate(messages)
            if isinstance(message["content"], list)
            for block_index, block in enumerate(message["content"])
            if block.get("type") == "compaction"
        ]
        if compactions and (
            len(compactions) != 1
            or compactions[0][:2] != (0, 0)
            or not isinstance(compactions[0][2].get("signature"), str)
            or not compactions[0][2]["signature"].strip()
        ):
            raise ValueError(
                "Anthropic replay requires exactly one signed compaction block first."
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
        state = request.previous_state
        history = self._messages(state)
        model = request.request_config["model"]
        for key, value in (("model", model), ("system_prompt", request.system_prompt)):
            if key in state.payload and state.payload[key] != value:
                raise ValueError(
                    f"Anthropic continuous conversation cannot change {key}."
                )
        has_summary = any(
            isinstance(message["content"], list)
            and any(block.get("type") == "compaction" for block in message["content"])
            for message in history
        )
        if (
            self._compaction is not None or has_summary
        ) and COMPACTION_BETA not in request.request_config.get("betas", []):
            raise ValueError(
                f"Anthropic native compaction requires beta {COMPACTION_BETA}."
            )
        history_count = len(history)
        completed_end = next(
            (
                index + 1
                for index in range(len(history) - 1, -1, -1)
                if history[index]["role"] == "assistant"
            ),
            0,
        )
        compaction_usage = NormalizedUsage()
        compaction_count = 0
        trigger_tokens = state.payload.get("context_tokens", 0)
        try:
            if (
                self._compaction is not None
                and completed_end
                and trigger_tokens >= self._compaction.trigger_tokens
            ):
                config = deepcopy(request.request_config)
                config["compaction"] = {"type": "summarize"}
                config["max_tokens"] = self._compaction.summary_max_output_tokens
                config.pop("stop_sequences", None)
                config.get("output_config", {}).pop("format", None)
                compacted = self._invoke(
                    request.system_prompt, history[:completed_end], config
                )
                compaction_usage = compacted.usage
                history = [
                    {
                        "role": "assistant",
                        "content": serialize_replay_content(
                            native_mapping(compacted.raw_response)["content"]
                        ),
                    },
                    *history[completed_end:],
                ]
                compaction_count = 1
            messages = self._messages(
                self.buffer_inputs(
                    replace_runtime_payload(
                        state, {**state.payload, "messages": history}
                    ),
                    request.new_messages,
                )
            )
            response = self._invoke(
                request.system_prompt, messages, request.request_config
            )
        except EmptyResponseError as exc:
            usage = (
                exc.usage
                if isinstance(exc.usage, NormalizedUsage)
                else NormalizedUsage()
            )
            raise InvalidProviderResponseError(
                "Anthropic compaction or action request did not complete.",
                response=sanitize_settings(exc.response),
                usage=compaction_usage + usage,
            ) from None
        except Exception as exc:
            raise InvalidProviderResponseError(
                f"Anthropic request failed ({type(exc).__name__}).",
                response={"provider_error": safe_provider_error_metadata(exc)},
                usage=compaction_usage,
            ) from None
        context_tokens = response.usage.total_tokens
        response = response.model_copy(
            update={"usage": compaction_usage + response.usage}
        )
        raw = native_mapping(response.raw_response)
        output = {
            "role": "assistant",
            "content": serialize_replay_content(raw["content"]),
        }
        next_messages = [*messages, output]
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
            "history_items_before_prune": history_count + len(request.new_messages) + 1,
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
                    "context_tokens": context_tokens,
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
            action_state={
                **counts,
                "stop_reason": raw.get("stop_reason"),
                **(
                    {
                        "native_compaction": {
                            "trigger_tokens": trigger_tokens,
                            "history_items_to_compact": completed_end,
                            "usage": compaction_usage.model_dump(),
                        }
                    }
                    if compaction_count
                    else {}
                ),
            },
        )

    def _invoke(
        self, system_prompt: str, messages: list[dict[str, Any]], config: dict[str, Any]
    ) -> ModelResponse:
        response = self._model_adapter.invoke(
            ModelRequest(
                messages=[Message(role="system", content=system_prompt)],
                request_config=deepcopy(config),
                native_input=deepcopy(messages),
            )
        )
        return normalize_native_response(response.raw_response, config)
