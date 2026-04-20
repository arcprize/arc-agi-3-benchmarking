from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .exceptions import EmptyResponseError
from .models import (
    ActionMetadata,
    CostDetails,
    InputTokensDetails,
    OutputTokensDetails,
    ResponseUsage,
    calculate_cost,
)


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ModelRequest(BaseModel):
    messages: list[Message]
    request_config: dict[str, Any]


class NormalizedUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    cost: float = 0.0
    cost_details: dict[str, float] = Field(default_factory=dict)

    def __add__(self, other: NormalizedUsage) -> NormalizedUsage:
        merged_cost_details: dict[str, float] = {}
        for key in set(
            list(self.cost_details.keys()) + list(other.cost_details.keys())
        ):
            merged_cost_details[key] = self.cost_details.get(
                key, 0.0
            ) + other.cost_details.get(key, 0.0)
        return NormalizedUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cost=self.cost + other.cost,
            cost_details=merged_cost_details,
        )


class ModelResponse(BaseModel):
    output_text: str
    reasoning_text: str | None = None
    usage: NormalizedUsage
    raw_response: Any | None = None


def _value_from_response_object(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _normalize_chat_usage(usage: Any) -> dict[str, Any]:
    if not usage:
        return {}

    normalized_usage_kwargs: dict[str, Any] = {
        "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }
    if getattr(usage, "completion_tokens_details", None):
        normalized_usage_kwargs["reasoning_tokens"] = (
            usage.completion_tokens_details.reasoning_tokens or 0
        )
    if getattr(usage, "prompt_tokens_details", None):
        normalized_usage_kwargs["cached_tokens"] = (
            usage.prompt_tokens_details.cached_tokens or 0
        )
        normalized_usage_kwargs["cache_write_tokens"] = (
            getattr(usage.prompt_tokens_details, "cache_write_tokens", 0) or 0
        )
    extras = getattr(usage, "model_extra", {}) or {}
    if "cost" in extras:
        normalized_usage_kwargs["cost"] = extras["cost"]
    if "cost_details" in extras:
        normalized_usage_kwargs["cost_details"] = extras["cost_details"]
    return normalized_usage_kwargs


def _normalize_responses_usage(usage: Any) -> dict[str, Any]:
    if not usage:
        return {}

    normalized_usage_kwargs: dict[str, Any] = {
        "input_tokens": _value_from_response_object(usage, "input_tokens", 0) or 0,
        "output_tokens": _value_from_response_object(usage, "output_tokens", 0) or 0,
        "total_tokens": _value_from_response_object(usage, "total_tokens", 0) or 0,
    }
    output_tokens_details = _value_from_response_object(
        usage,
        "output_tokens_details",
    )
    if output_tokens_details:
        normalized_usage_kwargs["reasoning_tokens"] = (
            _value_from_response_object(output_tokens_details, "reasoning_tokens", 0)
            or 0
        )
    input_tokens_details = _value_from_response_object(
        usage,
        "input_tokens_details",
    )
    if input_tokens_details:
        normalized_usage_kwargs["cached_tokens"] = (
            _value_from_response_object(input_tokens_details, "cached_tokens", 0)
            or 0
        )
        normalized_usage_kwargs["cache_write_tokens"] = (
            _value_from_response_object(
                input_tokens_details,
                "cache_write_tokens",
                0,
            )
            or 0
        )
    extras = _value_from_response_object(usage, "model_extra", {}) or {}
    if "cost" in extras:
        normalized_usage_kwargs["cost"] = extras["cost"]
    if "cost_details" in extras:
        normalized_usage_kwargs["cost_details"] = extras["cost_details"]
    return normalized_usage_kwargs


def _normalize_anthropic_messages_usage(usage: Any) -> dict[str, Any]:
    if not usage:
        return {}

    input_tokens = _value_from_response_object(usage, "input_tokens", 0) or 0
    output_tokens = _value_from_response_object(usage, "output_tokens", 0) or 0
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cached_tokens": (
            _value_from_response_object(usage, "cache_read_input_tokens", 0) or 0
        ),
        "cache_write_tokens": (
            _value_from_response_object(usage, "cache_creation_input_tokens", 0) or 0
        ),
    }


def _extract_responses_output_text(response: Any) -> str:
    helper_text = _value_from_response_object(response, "output_text")
    if helper_text is not None:
        return helper_text

    output_items = _value_from_response_object(response, "output", []) or []
    if not output_items:
        raise EmptyResponseError(
            "API returned 200 with empty output.",
            response=response,
        )

    text_parts: list[str] = []
    found_message_item = False
    for item in output_items:
        if _value_from_response_object(item, "type") != "message":
            continue
        role = _value_from_response_object(item, "role")
        if role not in (None, "assistant"):
            continue
        found_message_item = True
        for content_item in _value_from_response_object(item, "content", []) or []:
            if _value_from_response_object(content_item, "type") != "output_text":
                continue
            text_parts.append(_value_from_response_object(content_item, "text", "") or "")

    if not found_message_item:
        raise EmptyResponseError(
            "API returned 200 with empty output.",
            response=response,
        )

    return "".join(text_parts)


def _extract_anthropic_messages_output_text(response: Any) -> str:
    content_blocks = _value_from_response_object(response, "content", []) or []
    if not content_blocks:
        raise EmptyResponseError(
            "API returned 200 with empty output.",
            response=response,
        )

    text_parts: list[str] = []
    for content_block in content_blocks:
        if _value_from_response_object(content_block, "type") != "text":
            continue
        text_parts.append(_value_from_response_object(content_block, "text", "") or "")

    if not text_parts:
        raise EmptyResponseError(
            "API returned 200 with empty output.",
            response=response,
        )

    return "".join(text_parts)


def _extract_reasoning_text_fragment(item: Any) -> str | None:
    if isinstance(item, str):
        return item
    return (
        _value_from_response_object(item, "text")
        or _value_from_response_object(item, "summary_text")
    )


def _extract_responses_reasoning_text(response: Any) -> str | None:
    output_items = _value_from_response_object(response, "output", []) or []
    reasoning_parts: list[str] = []

    for item in output_items:
        if _value_from_response_object(item, "type") != "reasoning":
            continue

        for summary_item in _value_from_response_object(item, "summary", []) or []:
            fragment = _extract_reasoning_text_fragment(summary_item)
            if fragment:
                reasoning_parts.append(fragment)

        for content_item in _value_from_response_object(item, "content", []) or []:
            fragment = _extract_reasoning_text_fragment(content_item)
            if fragment:
                reasoning_parts.append(fragment)

    if not reasoning_parts:
        return None
    return "\n".join(reasoning_parts)


def normalize_chat_completion_response(response: Any) -> ModelResponse:
    if not getattr(response, "choices", None):
        raise EmptyResponseError(
            "API returned 200 with empty choices.",
            response=response,
        )

    message = response.choices[0].message
    usage = getattr(response, "usage", None)

    return ModelResponse(
        output_text=message.content or "",
        reasoning_text=getattr(message, "reasoning", None)
        or getattr(message, "reasoning_content", None),
        usage=NormalizedUsage(**_normalize_chat_usage(usage)),
        raw_response=response,
    )


def normalize_responses_response(response: Any) -> ModelResponse:
    return ModelResponse(
        output_text=_extract_responses_output_text(response),
        reasoning_text=_extract_responses_reasoning_text(response),
        usage=NormalizedUsage(
            **_normalize_responses_usage(
                _value_from_response_object(response, "usage"),
            )
        ),
        raw_response=response,
    )


def normalize_anthropic_messages_response(response: Any) -> ModelResponse:
    return ModelResponse(
        output_text=_extract_anthropic_messages_output_text(response),
        reasoning_text=None,
        usage=NormalizedUsage(
            **_normalize_anthropic_messages_usage(
                _value_from_response_object(response, "usage"),
            )
        ),
        raw_response=response,
    )


def action_metadata_from_model_response(
    model_response: ModelResponse,
    pricing: dict[str, float],
) -> ActionMetadata:
    input_cost = calculate_cost(model_response.usage.input_tokens, pricing.get("input", 0.0))
    output_cost = calculate_cost(
        model_response.usage.output_tokens, pricing.get("output", 0.0)
    )
    return ActionMetadata(
        output=model_response.output_text,
        reasoning=model_response.reasoning_text,
        usage=ResponseUsage(
            input_tokens=model_response.usage.input_tokens,
            input_tokens_details=InputTokensDetails(
                cached_tokens=model_response.usage.cached_tokens,
            ),
            output_tokens=model_response.usage.output_tokens,
            output_tokens_details=OutputTokensDetails(
                reasoning_tokens=model_response.usage.reasoning_tokens,
            ),
            total_tokens=model_response.usage.total_tokens,
        ),
        cost=CostDetails(
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=input_cost + output_cost,
        ),
    )
