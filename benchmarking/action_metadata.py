"""Size action metadata for ARC's reasoning field."""

from __future__ import annotations

import json
from typing import Any

# ARC rejects reasoning payloads above 16,384 bytes. Keep a 384-byte margin.
MAX_ACTION_METADATA_BYTES = 16_000
TEXT_FIELDS = ("output", "reasoning", "reasoning_summary")
TRUNCATION_MARKER = "\n\n... truncated {removed_chars} characters ...\n\n"


def serialized_action_metadata_size(payload: dict[str, Any]) -> int:
    """Return the exact UTF-8 size used by ARC's reasoning validator."""
    return len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))


def _truncated_text(text: str, retained_chars: int) -> str:
    """Keep a balanced prefix and suffix and report omitted characters."""
    removed_chars = len(text) - retained_chars
    if removed_chars <= 0:
        return text

    head_chars = (retained_chars + 1) // 2
    tail_chars = retained_chars // 2
    tail = text[-tail_chars:] if tail_chars else ""
    marker = TRUNCATION_MARKER.format(removed_chars=removed_chars)
    return text[:head_chars] + marker + tail


def _payload_with_retained_cap(
    payload: dict[str, Any],
    text_fields: dict[str, str],
    retained_cap: int,
) -> dict[str, Any]:
    """Cap only fields larger than ``retained_cap`` characters."""
    candidate = payload.copy()
    for field, text in text_fields.items():
        candidate[field] = _truncated_text(text, min(len(text), retained_cap))
    return candidate


def _fit_with_shared_retained_cap(
    payload: dict[str, Any],
    text_fields: dict[str, str],
    max_bytes: int,
) -> dict[str, Any] | None:
    """Find the highest waterline that makes the complete payload fit.

    A field below the waterline remains unchanged. Longer fields are capped at
    the same retained character count, which reduces the largest field first
    and then reduces similarly sized large fields together.
    """
    field_lengths = sorted({len(text) for text in text_fields.values()})

    # Marker insertion creates a size discontinuity when a field first becomes
    # truncated. Search each interval with a stable set of truncated fields.
    for index in range(len(field_lengths) - 1, -1, -1):
        low = field_lengths[index - 1] if index else 0
        high = field_lengths[index] - 1
        smallest = _payload_with_retained_cap(payload, text_fields, low)
        if serialized_action_metadata_size(smallest) > max_bytes:
            continue

        best = smallest
        while low <= high:
            retained_cap = (low + high) // 2
            candidate = _payload_with_retained_cap(
                payload,
                text_fields,
                retained_cap,
            )
            if serialized_action_metadata_size(candidate) <= max_bytes:
                best = candidate
                low = retained_cap + 1
            else:
                high = retained_cap - 1
        return best

    return None


def fit_action_metadata_payload(
    payload: dict[str, Any],
    max_bytes: int = MAX_ACTION_METADATA_BYTES,
) -> dict[str, Any]:
    """Return metadata fitted to a byte budget by trimming shared text fields.

    The complete payload is measured on every iteration. The largest text
    fields are reduced toward the same retained character count, while smaller
    fields stay intact. The input dictionary is not modified so full response
    text can remain available to local recordings.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")

    if serialized_action_metadata_size(payload) <= max_bytes:
        return payload.copy()

    text_fields = {
        field: value
        for field in TEXT_FIELDS
        if isinstance((value := payload.get(field)), str) and value
    }
    fitted = _fit_with_shared_retained_cap(payload, text_fields, max_bytes)
    if fitted is None:
        fitted = payload.copy()

    final_size = serialized_action_metadata_size(fitted)
    if final_size > max_bytes:
        raise ValueError(
            "Action metadata cannot fit within the byte budget by truncating "
            "reasoning, reasoning summary, and output fields: "
            f"{final_size} > {max_bytes}"
        )
    assert final_size <= max_bytes
    return fitted
