"""Per-run provider API request ledger and diagnostics."""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator

from .runtime_models import NormalizedUsage

LEDGER_FILENAME = "provider_requests.jsonl"
CLIENT_REQUEST_HEADER = "X-Client-Request-Id"
_PROVIDER_REQUEST_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "provider_request_context",
    default=None,
)
_CLIENT_HEADER_PROVIDERS = frozenset(
    {"anthropic", "deepseek", "google", "openai", "xai"}
)
_REQUEST_ID_HEADERS = {
    "anthropic-request-id",
    "openai-request-id",
    "request-id",
    "x-request-id",
    "x-requestid",
}


@contextmanager
def provider_request_context(
    *,
    run_dir: str | None = None,
    step: int | None = None,
    attempt: int | None = None,
    operation: str | None = None,
) -> Iterator[dict[str, Any] | None]:
    """Attach run/step metadata to provider calls in this context."""

    current = _PROVIDER_REQUEST_CONTEXT.get()
    if current is None and run_dir is None:
        yield None
        return

    context = dict(current or {})
    if run_dir is not None:
        context["run_dir"] = run_dir
    if step is not None:
        context["step"] = step
    if attempt is not None:
        context["attempt"] = attempt
    if operation is not None:
        context["operation"] = operation
    context.setdefault("request_index", 0)
    context.setdefault("provider_request_ids", [])
    token = _PROVIDER_REQUEST_CONTEXT.set(context)
    try:
        yield context
    finally:
        _PROVIDER_REQUEST_CONTEXT.reset(token)


def current_provider_request_ids() -> list[str]:
    context = _PROVIDER_REQUEST_CONTEXT.get()
    if context is None:
        return []
    values = context.get("provider_request_ids")
    return list(values) if isinstance(values, list) else []


def json_safe(value: Any, *, _depth: int = 0) -> Any:
    if _depth > 12:
        return repr(value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {
            str(key): json_safe(item, _depth=_depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item, _depth=_depth + 1) for item in value]
    if hasattr(value, "model_dump"):
        return json_safe(
            value.model_dump(mode="json", exclude_unset=True, warnings=False),
            _depth=_depth + 1,
        )
    if hasattr(value, "__dict__"):
        return json_safe(vars(value), _depth=_depth + 1)
    return repr(value)


def _headers(value: Any) -> dict[str, str]:
    raw = getattr(value, "headers", None)
    if raw is None:
        response = getattr(value, "response", None)
        raw = getattr(response, "headers", None)
    if raw is None and isinstance(value, dict):
        raw = value.get("headers")
    if raw is None:
        return {}
    try:
        return {str(key): str(item) for key, item in raw.items()}
    except Exception:
        return {}


def _status_code(value: Any) -> int | None:
    if isinstance(value, dict):
        for key in ("http_status", "status_code", "status"):
            candidate = value.get(key)
            if type(candidate) is int and 100 <= candidate <= 599:
                return candidate
        provider_error = value.get("provider_error")
        if isinstance(provider_error, dict):
            for key in ("http_status", "status_code", "status"):
                candidate = provider_error.get(key)
                if type(candidate) is int and 100 <= candidate <= 599:
                    return candidate
    candidates = [
        getattr(value, "status_code", None),
        getattr(value, "status", None),
    ]
    response = getattr(value, "response", None)
    if response is not None:
        candidates.extend(
            [
                getattr(response, "status_code", None),
                getattr(response, "status", None),
            ]
        )
    for candidate in candidates:
        if type(candidate) is int and 100 <= candidate <= 599:
            return candidate
    return None


def _body_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    body = getattr(value, "body", None)
    if isinstance(body, dict):
        return body
    safe = json_safe(value)
    return safe if isinstance(safe, dict) else {}


def _identifier(kind: str, value: Any, source: str, provider_name: str) -> dict[str, str]:
    return {
        "kind": kind,
        "value": str(value),
        "source": source,
        "provider_name": provider_name,
    }


def extract_provider_identifiers(value: Any) -> list[dict[str, str]]:
    identifiers: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    def add(kind: str, raw_value: Any, source: str, provider_name: str) -> None:
        if not isinstance(raw_value, str) or not raw_value.strip():
            return
        item = _identifier(kind, raw_value.strip(), source, provider_name)
        key = (item["kind"], item["value"], item["provider_name"])
        if key not in seen:
            seen.add(key)
            identifiers.append(item)

    for attr in ("request_id", "_request_id"):
        add("request_id", getattr(value, attr, None), "sdk_attribute", attr)

    for name, header_value in _headers(value).items():
        lower = name.lower()
        if lower in _REQUEST_ID_HEADERS:
            add("request_id", header_value, "response_header", name)
        elif lower == CLIENT_REQUEST_HEADER.lower():
            add("client_request_id", header_value, "request_header", name)

    body = _body_mapping(value)
    add("request_id", body.get("request_id"), "response_body", "request_id")
    add("response_id", body.get("id"), "response_body", "id")

    error = body.get("error")
    if isinstance(error, dict):
        add("request_id", error.get("request_id"), "response_body", "error.request_id")
    provider_error = body.get("provider_error")
    if isinstance(provider_error, dict):
        add(
            "request_id",
            provider_error.get("request_id"),
            "response_body",
            "provider_error.request_id",
        )

    return identifiers


def _request_summary(request_payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "fields": sorted(str(key) for key in request_payload),
    }
    model = request_payload.get("model")
    if isinstance(model, str):
        summary["model"] = model
    if "messages" in request_payload:
        messages = request_payload.get("messages")
        summary["message_count"] = len(messages) if isinstance(messages, list) else None
    if "input" in request_payload:
        input_value = request_payload.get("input")
        summary["input_count"] = len(input_value) if isinstance(input_value, list) else None
    return summary


def _append_jsonl(path: str, payload: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        json.dump(payload, handle, default=str)
        handle.write("\n")


def _write_diagnostic(run_dir: str, row: dict[str, Any], payload: dict[str, Any]) -> str:
    step = row.get("step") or "unknown"
    attempt = row.get("attempt") or "unknown"
    index = row.get("request_index") or "unknown"
    filename = (
        f"provider_request_step_{step}_attempt_{attempt}_"
        f"request_{index}_{uuid.uuid4().hex}.json"
    )
    path = os.path.join(run_dir, filename)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return filename


def _usage_payload(usage: NormalizedUsage | None) -> dict[str, Any] | None:
    return usage.model_dump() if isinstance(usage, NormalizedUsage) else None


@dataclass
class ProviderRequestAttempt:
    provider: str
    api_surface: str
    request_payload: dict[str, Any]
    model: str | None
    client_request_id: str
    context: dict[str, Any] | None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_monotonic: float = field(default_factory=time.monotonic)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    recorded: bool = False

    def record_success(self, response: Any, *, usage: NormalizedUsage | None = None) -> None:
        self._record("success", response=response, usage=usage)

    def record_exception(
        self,
        exception: Exception,
        *,
        response: Any | None = None,
        usage: NormalizedUsage | None = None,
        outcome: str = "error",
    ) -> None:
        self._record(outcome, response=response, exception=exception, usage=usage)

    def _record(
        self,
        outcome: str,
        *,
        response: Any | None = None,
        exception: Exception | None = None,
        usage: NormalizedUsage | None = None,
    ) -> None:
        if self.recorded or self.context is None:
            return
        self.recorded = True
        run_dir = self.context.get("run_dir")
        if not isinstance(run_dir, str) or not run_dir:
            return
        os.makedirs(run_dir, exist_ok=True)
        status_code = _status_code(exception) if exception is not None else None
        if status_code is None and response is not None:
            status_code = _status_code(response)
        identifiers = [
            *_identifier_list(self.client_request_id),
            *extract_provider_identifiers(response),
            *extract_provider_identifiers(exception),
        ]
        row = {
            "provider_request_id": self.request_id,
            "timestamp": self.started_at.isoformat(),
            "duration_seconds": round(time.monotonic() - self.started_monotonic, 3),
            "step": self.context.get("step"),
            "attempt": self.context.get("attempt"),
            "operation": self.context.get("operation"),
            "request_index": self.context.get("request_index"),
            "provider": self.provider,
            "api_surface": self.api_surface,
            "model": self.model,
            "outcome": outcome,
            "http_status": status_code,
            "error_class": type(exception).__name__ if exception is not None else None,
            "client_request_id": self.client_request_id,
            "identifiers": _dedupe_identifiers(identifiers),
            "usage": _usage_payload(usage),
            "request": _request_summary(self.request_payload),
        }
        if status_code is not None and 400 <= status_code <= 599:
            row["diagnostic_path"] = _write_diagnostic(
                run_dir,
                row,
                {
                    "ledger": row,
                    "request": json_safe(self.request_payload),
                    "response": json_safe(response),
                    "response_headers": _headers(response),
                    "exception": json_safe(
                        {
                            "class": type(exception).__name__,
                            "message": str(exception),
                            "body": getattr(exception, "body", None),
                            "headers": _headers(exception),
                        }
                    )
                    if exception is not None
                    else None,
                },
            )
        _append_jsonl(os.path.join(run_dir, LEDGER_FILENAME), row)
        ids = self.context.setdefault("provider_request_ids", [])
        if isinstance(ids, list):
            ids.append(self.request_id)


def _identifier_list(client_request_id: str) -> list[dict[str, str]]:
    return [
        _identifier(
            "client_request_id",
            client_request_id,
            "harness",
            CLIENT_REQUEST_HEADER,
        )
    ]


def _dedupe_identifiers(values: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for value in values:
        key = (value["kind"], value["value"], value["provider_name"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _inject_client_request_header(
    *,
    provider: str,
    api_surface: str,
    payload: dict[str, Any],
    client_request_id: str,
) -> None:
    if provider not in _CLIENT_HEADER_PROVIDERS:
        return
    if provider == "google" and api_surface == "generate_content":
        config = payload.get("config")
        if config is None:
            return
        options = getattr(config, "http_options", None)
        if options is None:
            try:
                from google.genai import types

                options = types.HttpOptions()
            except Exception:
                return
        elif hasattr(options, "model_copy"):
            options = options.model_copy(deep=True)
        headers = dict(getattr(options, "headers", None) or {})
        headers.setdefault(CLIENT_REQUEST_HEADER, client_request_id)
        options.headers = headers
        config.http_options = options
        payload["config"] = config
        return
    headers = dict(payload.get("extra_headers") or {})
    headers.setdefault(CLIENT_REQUEST_HEADER, client_request_id)
    payload["extra_headers"] = headers


def begin_provider_request(
    *,
    provider: str,
    api_surface: str,
    request_payload: dict[str, Any],
    model: str | None = None,
) -> ProviderRequestAttempt:
    context = _PROVIDER_REQUEST_CONTEXT.get()
    payload = deepcopy(request_payload)
    client_request_id = f"arc3-{uuid.uuid4()}"
    if context is not None:
        context["request_index"] = int(context.get("request_index") or 0) + 1
        _inject_client_request_header(
            provider=provider,
            api_surface=api_surface,
            payload=payload,
            client_request_id=client_request_id,
        )
    return ProviderRequestAttempt(
        provider=provider,
        api_surface=api_surface,
        request_payload=payload,
        model=model or payload.get("model"),
        client_request_id=client_request_id,
        context=context,
    )
