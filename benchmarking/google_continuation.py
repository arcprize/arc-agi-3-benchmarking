"""Automatic generateContent decode continuation within a single model turn."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import Any

from google.genai import types

from .exceptions import InvalidProviderResponseError
from .runtime_models import (
    ModelResponse,
    NormalizedUsage,
    normalize_google_genai_response,
)

logger = logging.getLogger(__name__)


def generate_with_continuation(
    client: Any, call_kwargs: dict[str, Any]
) -> ModelResponse:
    """Keep slices in one turn and preserve raw parts, signatures, and usage."""
    config = call_kwargs["config"].model_copy(deep=True)
    config.should_return_http_response = True
    options = config.http_options or types.HttpOptions()
    if options.timeout is None:
        options.timeout = 3_600_000
    # Avoid hidden replays of an expensive slice; the harness owns retries.
    options.retry_options = types.HttpRetryOptions(attempts=1)
    extra = deepcopy(options.extra_body or {})
    if any(
        key in extra for key in ("contents", "continuationToken", "generationConfig")
    ):
        raise ValueError(
            "Gemini continuation manages contents, token, and generationConfig."
        )
    options.extra_body = extra
    config.http_options = options
    base_contents = [
        content.model_dump(mode="json", by_alias=True, exclude_none=True)
        for content in call_kwargs["contents"]
    ]
    parts: list[dict[str, Any]] = []
    slices: list[dict[str, Any]] = []
    seen_tokens: set[str] = set()
    usage = NormalizedUsage()
    while True:
        try:
            response = client.models.generate_content(
                **{**call_kwargs, "config": config.model_copy(deep=True)}
            )
            raw = json.loads(response.sdk_http_response.body)
        except Exception as exc:
            if not slices:
                raise
            raise InvalidProviderResponseError(
                "Gemini continuation request failed after completed slices.",
                response={"slices": slices},
                usage=usage,
            ) from exc
        if not isinstance(raw, dict):
            raise InvalidProviderResponseError(
                "Gemini continuation response is not an object.",
                response={"slices": slices},
                usage=usage,
            )
        slices.append(raw)
        counts = raw.get("usageMetadata") or {}
        prompt = counts.get("promptTokenCount") or 0
        output = counts.get("candidatesTokenCount") or 0
        thoughts = counts.get("thoughtsTokenCount") or 0
        usage += NormalizedUsage(
            input_tokens=prompt,
            output_tokens=output + thoughts,
            reasoning_tokens=thoughts,
            total_tokens=prompt + output + thoughts,
            cached_tokens=counts.get("cachedContentTokenCount") or 0,
        )

        def invalid(message: str) -> InvalidProviderResponseError:
            return InvalidProviderResponseError(
                message, response={"slices": slices}, usage=usage
            )

        candidates = raw.get("candidates") or []
        if not candidates:
            raise invalid("Gemini continuation response has no candidates.")
        candidate = candidates[0]
        parts.extend((candidate.get("content") or {}).get("parts") or [])
        reason = candidate.get("finishReason")
        logger.info("Gemini decode slice %d finished with %s", len(slices), reason)
        if reason == "CONTINUATION":
            if not config.max_output_tokens or config.max_output_tokens <= 0:
                raise invalid(
                    "Gemini returned CONTINUATION without explicit positive max_output_tokens."
                )
            token = candidate.get("continuationToken")
            if not isinstance(token, str) or not token:
                raise invalid("Gemini CONTINUATION response has no continuation token.")
            if token in seen_tokens:
                raise invalid("Gemini returned a repeated continuation token.")
            seen_tokens.add(token)
            config.http_options.extra_body = {
                **extra,
                "continuationToken": token,
                "contents": base_contents
                + [{"role": "model", "parts": deepcopy(parts)}],
            }
            continue
        if reason not in ("STOP", "MAX_TOKENS"):
            raise invalid(f"Unexpected Gemini continuation finish reason: {reason}")
        if reason == "MAX_TOKENS":
            logger.warning("Gemini exhausted its cumulative output budget.")
        # The normalizer accepts mappings and keeps thought text separate from actions.
        aggregate = {
            "candidates": [{"content": {"parts": parts}, "finish_reason": reason}],
            "usage_metadata": {
                "prompt_token_count": usage.input_tokens,
                "candidates_token_count": usage.output_tokens - usage.reasoning_tokens,
                "thoughts_token_count": usage.reasoning_tokens,
                "cached_content_token_count": usage.cached_tokens,
            },
            "slices": slices,
        }
        return normalize_google_genai_response(aggregate)
