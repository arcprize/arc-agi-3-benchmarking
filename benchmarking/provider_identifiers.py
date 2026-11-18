"""Best-effort provider request/response id extraction for diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

ProviderIdentifier = dict[str, str]

FIELD_KINDS = {
    "_request_id": "request_id",
    "request_id": "request_id",
    "requestId": "request_id",
    "response_id": "response_id",
    "responseId": "response_id",
    "id": "id",
}

HEADER_KINDS = {
    "x-request-id": "request_id",
    "request-id": "request_id",
    "x-goog-request-id": "request_id",
    "x-google-request-id": "request_id",
    "openai-request-id": "request_id",
    "anthropic-request-id": "request_id",
}


def _value(item: Any, name: str) -> Any:
    if isinstance(item, Mapping):
        if name in item:
            return item[name]
        lower = name.lower()
        return next(
            (
                value
                for key, value in item.items()
                if isinstance(key, str) and key.lower() == lower
            ),
            None,
        )
    return getattr(item, name, None)


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > 512 or any(ord(char) < 32 for char in value):
        return None
    return value


def _header_value(headers: Any, name: str) -> Any:
    if not hasattr(headers, "get"):
        return None
    value = headers.get(name)
    if value is not None or not isinstance(headers, Mapping):
        return value
    lower = name.lower()
    return next(
        (
            value
            for key, value in headers.items()
            if isinstance(key, str) and key.lower() == lower
        ),
        None,
    )


def merge_provider_identifiers(
    *groups: list[ProviderIdentifier],
) -> list[ProviderIdentifier]:
    merged: list[ProviderIdentifier] = []
    seen: set[tuple[str, str, str]] = set()
    for identifier in (item for group in groups for item in group):
        key = (
            identifier.get("kind", ""),
            identifier.get("value", ""),
            identifier.get("source", ""),
        )
        if "" in key or key in seen:
            continue
        seen.add(key)
        merged.append(identifier)
    return merged


def _add(
    identifiers: list[ProviderIdentifier],
    *,
    kind: str,
    value: Any,
    source: str,
    provider_name: str,
) -> None:
    if clean := _clean(value):
        identifiers[:] = merge_provider_identifiers(
            identifiers,
            [
                {
                    "kind": kind,
                    "value": clean,
                    "source": source,
                    "provider_name": provider_name,
                }
            ],
        )


def _extract_from_item(item: Any, source: str) -> list[ProviderIdentifier]:
    identifiers: list[ProviderIdentifier] = []
    if isinstance(item, Mapping):
        metadata = item.get("diagnostic_metadata")
        existing = (
            metadata.get("provider_identifiers")
            if isinstance(metadata, Mapping)
            else None
        )
        if isinstance(existing, list):
            identifiers.extend(
                identifier for identifier in existing if isinstance(identifier, dict)
            )
    for field, kind in FIELD_KINDS.items():
        _add(
            identifiers,
            kind=kind,
            value=_value(item, field),
            source=f"{source}.{field}",
            provider_name=field,
        )
    for field in ("headers", "response_headers"):
        headers = _value(item, field)
        for header, kind in HEADER_KINDS.items():
            _add(
                identifiers,
                kind=kind,
                value=_header_value(headers, header),
                source=f"{source}.{field}.{header}",
                provider_name=header,
            )
    return merge_provider_identifiers(identifiers)


def extract_provider_identifiers(response: Any) -> list[ProviderIdentifier]:
    identifiers = _extract_from_item(response, "response")
    for field in ("sdk_http_response", "http_response"):
        if nested := _value(response, field):
            identifiers = merge_provider_identifiers(
                identifiers,
                _extract_from_item(nested, f"response.{field}"),
            )
    if isinstance(response, Mapping) and isinstance(response.get("slices"), list):
        for index, item in enumerate(response["slices"]):
            identifiers = merge_provider_identifiers(
                identifiers,
                _extract_from_item(item, f"response.slices[{index}]"),
            )
    return identifiers
