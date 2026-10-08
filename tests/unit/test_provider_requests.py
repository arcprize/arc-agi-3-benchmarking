import json

import httpx
import pytest

from benchmarking.provider_requests import (
    CLIENT_REQUEST_HEADER,
    LEDGER_FILENAME,
    begin_provider_request,
    current_provider_request_ids,
    provider_request_context,
)
from benchmarking.runtime_models import NormalizedUsage


@pytest.mark.unit
class TestProviderRequestRecording:
    def test_records_compact_ledger_and_error_diagnostic(self, tmp_path):
        with provider_request_context(
            run_dir=str(tmp_path),
            step=7,
            attempt=2,
            operation="action",
        ):
            attempt = begin_provider_request(
                provider="openai",
                api_surface="responses",
                request_payload={
                    "model": "gpt-5.4",
                    "input": [{"role": "user", "content": "frame"}],
                },
            )
            assert attempt.request_payload["extra_headers"][CLIENT_REQUEST_HEADER]
            error = httpx.HTTPStatusError(
                "server error",
                request=httpx.Request("POST", "https://example.test/v1/responses"),
                response=httpx.Response(
                    500,
                    headers={"x-request-id": "req_provider_123"},
                    json={"error": {"message": "failed"}},
                ),
            )

            attempt.record_exception(
                error,
                usage=NormalizedUsage(input_tokens=10, output_tokens=2, total_tokens=12),
            )
            recorded_ids = current_provider_request_ids()

        ledger_lines = (tmp_path / LEDGER_FILENAME).read_text().splitlines()
        assert len(ledger_lines) == 1
        row = json.loads(ledger_lines[0])
        assert row["provider_request_id"] == recorded_ids[0]
        assert row["step"] == 7
        assert row["attempt"] == 2
        assert row["provider"] == "openai"
        assert row["api_surface"] == "responses"
        assert row["http_status"] == 500
        assert row["usage"]["total_tokens"] == 12
        assert {
            "kind": "request_id",
            "value": "req_provider_123",
            "source": "response_header",
            "provider_name": "x-request-id",
        } in row["identifiers"]
        diagnostic = json.loads((tmp_path / row["diagnostic_path"]).read_text())
        assert diagnostic["request"]["input"] == [
            {"role": "user", "content": "frame"}
        ]
        assert diagnostic["exception"]["class"] == "HTTPStatusError"

    def test_outside_context_does_not_mutate_request_or_write_ledger(self, tmp_path):
        attempt = begin_provider_request(
            provider="openai",
            api_surface="chat_completions",
            request_payload={"model": "gpt-5.4"},
        )

        attempt.record_success({"id": "chatcmpl_123"})

        assert "extra_headers" not in attempt.request_payload
        assert not (tmp_path / LEDGER_FILENAME).exists()
