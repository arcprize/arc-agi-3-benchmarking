"""Inception Chat Completions with lossless encrypted reasoning replay."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .open_source_runtime import (
    OpenSourceChatCompletionsAdapter,
    OpenSourceContinuousConversationRuntimeAdapter,
    _invalid,
    normalize_open_source_response,
)
from .runtime_models import ModelResponse
from .runtime_state import AdapterDescriptor

INCEPTION_ADAPTER_ID = "inception.chat_completions.v1"
_DETAIL_FIELDS = frozenset({"type", "format", "index", "id", "data"})


def _validate_detail(detail: Any, *, partial: bool = False) -> None:
    if not isinstance(detail, dict) or set(detail) - _DETAIL_FIELDS:
        raise ValueError("Unsupported Inception reasoning detail fields.")
    index = detail.get("index")
    if type(index) is not int or index < 0:
        raise ValueError("Inception reasoning details require a nonnegative index.")
    for key, expected in (("type", "reasoning.encrypted"), ("format", "inception-v1")):
        if (not partial or key in detail) and detail.get(key) != expected:
            raise ValueError("Unsupported Inception reasoning detail type or format.")
    if "id" in detail and not isinstance(detail["id"], str):
        raise ValueError("Inception reasoning detail id must be a string.")
    if not partial or "data" in detail:
        if not isinstance(detail.get("data"), str) or (
            not partial and not detail["data"]
        ):
            raise ValueError("Inception encrypted reasoning must contain string data.")


def _validated_details(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Inception reasoning_details must be a list.")
    seen = set()
    for detail in value:
        _validate_detail(detail)
        if detail["index"] in seen:
            raise ValueError("Duplicate Inception reasoning detail index.")
        seen.add(detail["index"])
    return deepcopy(value)


def normalize_inception_response(raw: dict[str, Any]) -> ModelResponse:
    # Keep the common completion/usage gate, but handle Inception's opaque field
    # explicitly rather than letting the text-only adapter discard or reject it.
    projected = deepcopy(raw)
    try:
        for choice in projected.get("choices") or []:
            message = choice.get("message") or {}
            _validated_details(message.pop("reasoning_details", None))
    except ValueError as exc:
        raise _invalid(str(exc), raw) from exc
    # Some Inception surfaces use the top-level usage aliases.
    usage = raw.get("usage") or {}
    for container, field, alias in (
        ("completion_tokens_details", "reasoning_tokens", "reasoning_tokens"),
        ("prompt_tokens_details", "cached_tokens", "cached_input_tokens"),
    ):
        if (
            isinstance(usage, dict)
            and alias in usage
            and ((usage.get(container) or {}).get(field) is None)
        ):
            value = usage[alias]
            if type(value) is not int or value < 0:
                raise _invalid("Inception returned invalid token usage.", raw)
            projected["usage"][container] = {
                **(usage.get(container) or {}),
                field: value,
            }
    response = normalize_open_source_response(projected)
    response.raw_response = deepcopy(raw)
    return response


class InceptionChatCompletionsAdapter(OpenSourceChatCompletionsAdapter):
    """Collect encrypted blocks by index without interpreting their ciphertext."""

    _normalize_response = staticmethod(normalize_inception_response)

    @staticmethod
    def _merge_structured_delta(
        message: dict[str, Any], delta: dict[str, Any], response: dict[str, Any]
    ) -> None:
        OpenSourceChatCompletionsAdapter._merge_structured_delta(
            message,
            {key: value for key, value in delta.items() if key != "reasoning_details"},
            response,
        )
        details = delta.get("reasoning_details")
        if details is None:
            return
        try:
            if not isinstance(details, list):
                raise ValueError("Inception reasoning_details delta must be a list.")
            assembled = message.setdefault("reasoning_details", [])
            for fragment in details:
                _validate_detail(fragment, partial=True)
                existing = next(
                    (item for item in assembled if item["index"] == fragment["index"]),
                    None,
                )
                if existing is None:
                    assembled.append(deepcopy(fragment))
                    continue
                for key, value in fragment.items():
                    if key == "data":
                        existing[key] = existing.get(key, "") + value
                    elif key in existing and existing[key] != value:
                        raise ValueError(
                            "Conflicting Inception reasoning detail metadata."
                        )
                    else:
                        existing[key] = value
        except ValueError as exc:
            raise _invalid(str(exc), response) from exc


class InceptionContinuousConversationRuntimeAdapter(
    OpenSourceContinuousConversationRuntimeAdapter
):
    """Use shared turn/compaction transactions with Inception-native replay."""

    def __init__(self, *, model_adapter: Any, descriptor: AdapterDescriptor) -> None:
        self._model_adapter = model_adapter
        self.descriptor = descriptor
        # Fixed protocol identity stored in runtime state, not a user setting.
        self.reasoning_replay = "reasoning_details"

    def _replay_message(self, message: dict[str, Any]) -> dict[str, Any]:
        replay = {"role": message["role"], "content": message["content"]}
        if message.get("reasoning_content") is not None:
            replay["reasoning_content"] = message["reasoning_content"]
        if "reasoning_details" in message:
            replay["reasoning_details"] = _validated_details(
                message["reasoning_details"]
            )
        return replay

    def _assistant_message(self, response: ModelResponse) -> dict[str, Any]:
        assistant = super()._assistant_message(response)
        raw_message = response.raw_response["choices"][0]["message"]
        if "reasoning_details" in raw_message:
            assistant["reasoning_details"] = _validated_details(
                raw_message["reasoning_details"]
            )
        return assistant
