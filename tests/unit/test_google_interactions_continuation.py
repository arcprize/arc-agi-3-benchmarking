import json

import httpx
import pytest
from google import genai
from google.genai import types

from benchmarking.exceptions import InvalidProviderResponseError
from benchmarking.google_interactions_continuation import create_with_continuation
from benchmarking.runtime_models import normalize_google_interaction_response


def response(status, **extra):
    return {
        "id": "test-id",
        "created": "2026-10-02T12:00:00Z",
        "updated": "2026-10-02T12:00:00Z",
        "status": status,
        "steps": [],
        **extra,
    }


@pytest.mark.parametrize("missing", [False, True])
def test_experimental_wire_contract_and_accounting(missing):
    replies = iter(
        [
            response(
                "continuation",
                **({} if missing else {"continuation_token": "opaque-token"}),
                steps=[
                    {
                        "type": "thought",
                        "signature": "signed",
                        "summary": [{"type": "text", "text": "think"}],
                    }
                ],
                usage={"total_input_tokens": 5, "total_thought_tokens": 7},
            ),
            response(
                "completed",
                steps=[
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": "ACTION1"}],
                    }
                ],
                usage={"total_input_tokens": 6, "total_output_tokens": 2},
            ),
        ]
    )
    calls = []

    def handle(req):
        calls.append(json.loads(req.content))
        return httpx.Response(200, json=next(replies))

    client = genai.Client(
        api_key="test",
        http_options=types.HttpOptions(
            httpx_client=httpx.Client(transport=httpx.MockTransport(handle))
        ),
    )
    kwargs = {
        "model": "gemini-test",
        "input": "frame",
        "store": False,
        "generation_config": {"max_output_tokens": 1000000},
    }
    try:
        if missing:
            with pytest.raises(InvalidProviderResponseError) as error:
                create_with_continuation(client, kwargs)
            assert error.value.usage.total_tokens == 12
            assert len(calls) == 1
        else:
            result = normalize_google_interaction_response(
                create_with_continuation(client, kwargs)
            )
            assert result.output_text == "ACTION1"
            assert result.reasoning_text == "think"
            assert result.usage.total_tokens == 20
            assert result.usage.reasoning_tokens == 7
            assert calls[1]["continuation_token"] == "opaque-token"
            assert calls[1]["input"] == calls[0]["input"]
            assert calls[1]["generation_config"]["max_output_tokens"] == 1000000
            assert "continuation_token" not in calls[0]
    finally:
        client.close()
