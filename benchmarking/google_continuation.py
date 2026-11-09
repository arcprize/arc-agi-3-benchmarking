"""Automatic generateContent decode continuation within a single model turn."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from threading import current_thread
from typing import Any

import httpx
from google.genai import types

from .exceptions import InvalidProviderResponseError
from .runtime_models import (
    ModelResponse,
    NormalizedUsage,
    normalize_google_genai_response,
)

logger = logging.getLogger(__name__)
MAX_STREAM_RECOVERY_RETRIES = 3


def generate_with_continuation(
    client: Any,
    call_kwargs: dict[str, Any],
    *,
    native_contents: list[Any] | None = None,
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
    if native_contents is not None:
        base_contents = deepcopy(native_contents)
        options.extra_body = {**extra, "contents": base_contents}
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
                "contents": base_contents,
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


def stream_with_continuation(
    client: Any,
    call_kwargs: dict[str, Any],
    *,
    native_contents: list[Any] | None = None,
) -> ModelResponse:
    """Stream one turn, accounting for cumulative usage in each decode hop."""
    config = call_kwargs["config"].model_copy(deep=True)
    # The harness owns retries and usage accounting. The SDK must not silently
    # reconnect or merge continuation hops on our behalf.
    config.automatic_continuation = False
    options = config.http_options or types.HttpOptions()
    if options.timeout is None:
        options.timeout = 3_600_000
    options.retry_options = types.HttpRetryOptions(attempts=1)
    extra = deepcopy(options.extra_body or {})
    if any(
        key in extra for key in ("contents", "continuationToken", "generationConfig")
    ):
        raise ValueError(
            "Gemini continuation manages contents, token, and generationConfig."
        )
    if config.continuation_token is not None:
        raise ValueError("Gemini continuation manages the continuation token.")
    base_contents = [
        content.model_dump(mode="json", by_alias=True, exclude_none=True)
        for content in call_kwargs["contents"]
    ]
    if native_contents is not None:
        base_contents = deepcopy(native_contents)
        options.extra_body = {**extra, "contents": base_contents}
    else:
        options.extra_body = extra
    config.http_options = options

    parts: list[dict[str, Any]] = []
    slices: list[dict[str, Any]] = []
    seen_tokens: set[bytes] = set()
    usage = NormalizedUsage()
    recovery_retries = 0

    while True:
        hop_parts: list[dict[str, Any]] = []
        pending_parts: list[dict[str, Any]] = []
        hop_usage = NormalizedUsage()
        hop_usage_metadata: dict[str, Any] = {}
        checkpoint_usage = NormalizedUsage()
        checkpoint_usage_metadata: dict[str, Any] = {}
        reason: str | None = None
        token: bytes | None = None
        saw_chunk = False
        try:
            for chunk in client.models.generate_content_stream(
                **{**call_kwargs, "config": config.model_copy(deep=True)}
            ):
                saw_chunk = True
                if chunk.usage_metadata is not None:
                    counts = chunk.usage_metadata
                    hop_usage_metadata = counts.model_dump(
                        mode="json", by_alias=True, exclude_none=True
                    )
                    prompt = counts.prompt_token_count or 0
                    output = counts.candidates_token_count or 0
                    thoughts = counts.thoughts_token_count or 0
                    hop_usage = NormalizedUsage(
                        input_tokens=prompt,
                        output_tokens=output + thoughts,
                        reasoning_tokens=thoughts,
                        total_tokens=(
                            counts.total_token_count
                            if counts.total_token_count is not None
                            else prompt + output + thoughts
                        ),
                        cached_tokens=counts.cached_content_token_count or 0,
                    )
                if not chunk.candidates:
                    continue
                candidate = chunk.candidates[0]
                if candidate.content and candidate.content.parts:
                    pending_parts.extend(
                        part.model_dump(mode="json", by_alias=True, exclude_none=True)
                        for part in candidate.content.parts
                    )
                if candidate.continuation_token:
                    next_token = candidate.continuation_token
                    if next_token != token:
                        if (
                            next_token in seen_tokens
                            or next_token == config.continuation_token
                        ):
                            raise InvalidProviderResponseError(
                                "Gemini returned a repeated continuation token.",
                                response={"slices": slices},
                                usage=usage + hop_usage,
                            )
                        token = next_token
                        seen_tokens.add(token)
                        hop_parts.extend(pending_parts)
                        pending_parts.clear()
                        checkpoint_usage = hop_usage.model_copy(deep=True)
                        checkpoint_usage_metadata = deepcopy(hop_usage_metadata)
                if candidate.finish_reason is not None:
                    reason = candidate.finish_reason.value
        except Exception as exc:
            if isinstance(exc, InvalidProviderResponseError):
                raise
            if not isinstance(exc, httpx.TransportError):
                if not slices and not saw_chunk:
                    raise
                raise InvalidProviderResponseError(
                    "Gemini stream failed after producing response chunks.",
                    response={
                        "slices": slices
                        + [
                            _stream_slice(
                                hop_parts + pending_parts, reason, hop_usage_metadata
                            )
                        ]
                    },
                    usage=usage + hop_usage,
                ) from exc
            if token and not checkpoint_usage_metadata:
                raise InvalidProviderResponseError(
                    "Gemini stream checkpoint has no usage metadata.",
                    response={"slices": slices},
                    usage=usage + hop_usage,
                ) from exc
            if token:
                usage += checkpoint_usage
                parts.extend(hop_parts)
                slices.append(
                    _stream_slice(hop_parts, "CONTINUATION", checkpoint_usage_metadata)
                )
                config.continuation_token = token
            recovery_retries += 1
            if recovery_retries <= MAX_STREAM_RECOVERY_RETRIES:
                logger.warning(
                    "Gemini stream disconnected; retrying from %s checkpoint (%d/%d)",
                    "latest" if token else "initial",
                    recovery_retries,
                    MAX_STREAM_RECOVERY_RETRIES,
                )
                continue
            if not slices and not saw_chunk:
                raise
            raise InvalidProviderResponseError(
                "Gemini stream recovery retries exhausted.",
                response={"slices": slices},
                usage=usage,
            ) from exc

        if reason == "CONTINUATION" and token:
            # A terminal signal without a new token may follow an earlier
            # checkpoint. Only output covered by that token can be resumed.
            completed_parts = hop_parts
            completed_usage = checkpoint_usage
            completed_usage_metadata = checkpoint_usage_metadata
        else:
            completed_parts = hop_parts + pending_parts
            completed_usage = hop_usage
            completed_usage_metadata = hop_usage_metadata
        if saw_chunk:
            slices.append(
                _stream_slice(completed_parts, reason, completed_usage_metadata)
            )
        logger.info(
            "Gemini stream %s hop %d finished with %s; reported usage: %s",
            current_thread().name,
            len(slices),
            reason,
            bool(completed_usage_metadata),
        )
        if not completed_usage_metadata:
            raise InvalidProviderResponseError(
                "Gemini stream ended without usage metadata.",
                response={"slices": slices},
                usage=usage,
            )
        usage += completed_usage
        parts.extend(completed_parts)

        def invalid(message: str) -> InvalidProviderResponseError:
            return InvalidProviderResponseError(
                message, response={"slices": slices}, usage=usage
            )

        if reason == "CONTINUATION":
            if not config.max_output_tokens or config.max_output_tokens <= 0:
                raise invalid(
                    "Gemini returned CONTINUATION without explicit positive max_output_tokens."
                )
            if not token:
                raise invalid("Gemini CONTINUATION response has no continuation token.")
            config.continuation_token = token
            continue
        if reason not in ("STOP", "MAX_TOKENS"):
            raise invalid(f"Unexpected Gemini stream finish reason: {reason}")
        if reason == "MAX_TOKENS":
            logger.warning("Gemini exhausted its cumulative output budget.")
        aggregate = {
            "candidates": [{"content": {"parts": parts}, "finish_reason": reason}],
            "usage_metadata": {
                "prompt_token_count": usage.input_tokens,
                "candidates_token_count": usage.output_tokens - usage.reasoning_tokens,
                "thoughts_token_count": usage.reasoning_tokens,
                "cached_content_token_count": usage.cached_tokens,
                "total_token_count": usage.total_tokens,
            },
            "slices": slices,
        }
        return normalize_google_genai_response(aggregate)


def _stream_slice(
    parts: list[dict[str, Any]], reason: str | None, usage: dict[str, Any]
) -> dict[str, Any]:
    return {
        "candidates": [{"content": {"parts": parts}, "finishReason": reason}],
        "usageMetadata": usage,
    }
