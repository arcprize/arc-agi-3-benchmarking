"""Assemble text-only Chat Completions streams for the standard harness."""

from __future__ import annotations

from typing import Any

from openai.types.chat import ChatCompletion

from .exceptions import EmptyResponseError
from .runtime_models import (
    ModelResponse,
    NormalizedUsage,
    _normalize_chat_usage,
    normalize_chat_completion_response,
)


def consume_chat_stream(stream: Any) -> ModelResponse:
    content: list[str] = []
    reasoning: dict[str, list[str]] = {"reasoning": [], "reasoning_content": []}
    usage: dict[str, Any] = {}
    sdk_usage: Any = None
    metadata: dict[str, Any] = {}
    finish_reason = None
    unsupported = False

    def invalid(message: str) -> EmptyResponseError:
        return EmptyResponseError(
            message,
            response={"finish_reason": finish_reason, "usage": usage},
            usage=NormalizedUsage(**_normalize_chat_usage(sdk_usage)),
        )

    try:
        for event in stream:
            chunk = event.model_dump(mode="json", exclude_unset=True, warnings=False)
            if chunk.get("usage") is not None:
                usage = chunk["usage"]
                sdk_usage = event.usage
            for key in ("id", "model", "created", "system_fingerprint"):
                if key in chunk:
                    metadata[key] = chunk[key]
            if chunk.get("error"):
                raise invalid("Chat Completions stream returned a provider error.")
            for choice in chunk.get("choices", []):
                if choice.get("index") != 0 or finish_reason is not None:
                    raise invalid("Unexpected choice after or outside stream completion.")
                delta = choice.get("delta") or {}
                if delta.get("tool_calls") or delta.get("function_call") or delta.get("refusal"):
                    unsupported = True
                for key, parts in (("content", content), *reasoning.items()):
                    value = delta.get(key)
                    if value is not None:
                        if not isinstance(value, str):
                            raise invalid("Chat Completions returned a non-text delta.")
                        parts.append(value)
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
        if finish_reason != "stop" or unsupported:
            raise invalid("Chat Completions stream did not complete a text response.")
        if not isinstance(usage, dict) or any(
            not isinstance(usage.get(key), int)
            or isinstance(usage[key], bool)
            or usage[key] < 0
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        ):
            raise invalid("Chat Completions stream ended without valid token usage.")
        text = "".join(content)
        if not text.strip():
            raise invalid("Chat Completions stream returned no visible action text.")
        raw = ChatCompletion.model_validate({
            **metadata,
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "finish_reason": finish_reason,
                "message": {
                    "role": "assistant", "content": text,
                    **{key: "".join(parts) for key, parts in reasoning.items() if parts},
                },
            }],
            "usage": usage,
        })
        return normalize_chat_completion_response(raw)
    except EmptyResponseError:
        raise
    except Exception as exc:
        raise invalid(f"Chat Completions stream failed ({type(exc).__name__}).") from None
    finally:
        stream.close()
