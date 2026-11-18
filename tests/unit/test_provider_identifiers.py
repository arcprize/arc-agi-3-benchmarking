from types import SimpleNamespace

from benchmarking.provider_identifiers import extract_provider_identifiers


def test_extracts_provider_identifiers_from_sdk_http_response_headers():
    response = SimpleNamespace(
        sdk_http_response=SimpleNamespace(
            headers={"x-goog-request-id": "goog-request-123"}
        )
    )

    assert extract_provider_identifiers(response) == [
        {
            "kind": "request_id",
            "value": "goog-request-123",
            "source": "response.sdk_http_response.headers.x-goog-request-id",
            "provider_name": "x-goog-request-id",
        }
    ]


def test_extracts_provider_identifiers_from_response_object_fields():
    response = SimpleNamespace(_request_id="req_openai_123", id="resp_123")

    assert extract_provider_identifiers(response) == [
        {
            "kind": "request_id",
            "value": "req_openai_123",
            "source": "response._request_id",
            "provider_name": "_request_id",
        },
        {
            "kind": "id",
            "value": "resp_123",
            "source": "response.id",
            "provider_name": "id",
        },
    ]
