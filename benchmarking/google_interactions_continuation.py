"""LOCAL EXPERIMENT: assumed Interactions decode continuation, not a confirmed API contract.

Assumes top-level continuation_token plus status=continuation or
finish_reason=CONTINUATION; sends continuation_token through extra_body.
Steps and usage are assumed incremental per slice. No inference from incomplete.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

from .exceptions import InvalidProviderResponseError
from .provider_requests import begin_provider_request
from .runtime_models import NormalizedUsage, _normalize_google_interactions_usage

logger = logging.getLogger(__name__)


def value(response: Any, key: str, default: Any = None) -> Any:
    return (
        response.get(key, default)
        if isinstance(response, dict)
        else getattr(response, key, default)
    )


def create_with_continuation(
    client: Any,
    kwargs: dict[str, Any],
    *,
    provider: str = "google",
    api_surface: str = "interactions",
) -> Any:
    payload = deepcopy(kwargs)
    steps: list[Any] = []
    slices: list[Any] = []
    seen: set[str] = set()
    usage = NormalizedUsage()
    while True:
        attempt = begin_provider_request(
            provider=provider,
            api_surface=api_surface,
            request_payload=payload,
            model=payload.get("model"),
        )
        try:
            response = client.interactions.create(**attempt.request_payload)
        except Exception as exc:
            attempt.record_exception(exc, response=getattr(exc, "response", None))
            if not slices:
                raise
            raise InvalidProviderResponseError(
                "Experimental Interactions continuation request failed.",
                response={"slices": slices},
                usage=usage,
            ) from exc
        reason = value(response, "finish_reason", value(response, "finishReason"))
        status = value(response, "status")
        continuing = reason == "CONTINUATION" or str(status).lower() == "continuation"
        # Preserve the existing SDK response and normalization on ordinary calls.
        if not slices and not continuing:
            attempt.record_success(response)
            return response
        slices.append(response)
        usage += NormalizedUsage(
            **_normalize_google_interactions_usage(value(response, "usage"))
        )
        attempt.record_success(response, usage=usage)
        steps.extend(deepcopy(value(response, "steps", []) or []))
        logger.info(
            "Experimental Gemini Interactions slice %d: status=%s finish_reason=%s",
            len(slices),
            status,
            reason,
        )
        if not continuing:
            if status != "completed":
                raise InvalidProviderResponseError(
                    "Interactions continuation did not complete.",
                    response={"slices": slices},
                    usage=usage,
                )
            return {
                "id": value(response, "id"),
                "status": status,
                "steps": steps,
                "usage": {
                    "total_input_tokens": usage.input_tokens,
                    "total_output_tokens": usage.output_tokens - usage.reasoning_tokens,
                    "total_thought_tokens": usage.reasoning_tokens,
                    "total_tokens": usage.total_tokens,
                    "total_cached_tokens": usage.cached_tokens,
                },
                "slices": slices,
            }
        token = value(
            response, "continuation_token", value(response, "continuationToken")
        )
        budget = kwargs.get("generation_config", {}).get("max_output_tokens")
        if (
            not isinstance(token, str)
            or not token
            or token in seen
            or not budget
            or budget <= 0
        ):
            raise InvalidProviderResponseError(
                "Invalid experimental Interactions continuation token or budget.",
                response={"slices": slices},
                usage=usage,
            )
        seen.add(token)
        payload["extra_body"] = {
            **kwargs.get("extra_body", {}),
            "continuation_token": token,
        }
        # Resend original input; do not introduce a new user turn or interaction ID.
