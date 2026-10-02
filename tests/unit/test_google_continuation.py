"""Validate actual SDK serialization against an in-memory HTTP transport."""

import json
from copy import deepcopy

import httpx
import pytest
from google import genai
from google.genai import types

from benchmarking.exceptions import EmptyResponseError, InvalidProviderResponseError
from benchmarking.runtime_adapters import (
    GoogleGenAIGenerateContentAdapter,
)
from benchmarking.runtime_models import Message, ModelRequest


def reply(reason, parts=None, token=None, usage=None):
    candidate = {"finishReason": reason, "content": {"parts": parts or []}}
    if token is not None:
        candidate["continuationToken"] = token
    return {"candidates": [candidate], "usageMetadata": usage or {}}


def request(**overrides):
    return ModelRequest(
        messages=[
            Message(role="system", content="Play the game."),
            Message(role="user", content="frame"),
        ],
        request_config={
            "model": "models/gemini-test",
            "max_output_tokens": 1000000,
            **overrides,
        },
    )


@pytest.fixture
def make_adapter():
    clients = []

    def make(responses):
        responses = iter(responses)
        calls = []

        def handle(req):
            calls.append(req)
            response = next(responses)
            if isinstance(response, Exception):
                raise response
            return httpx.Response(200, json=response)

        client = genai.Client(
            api_key="test-key",
            http_options=types.HttpOptions(
                httpx_client=httpx.Client(transport=httpx.MockTransport(handle))
            ),
        )
        clients.append(client)
        return GoogleGenAIGenerateContentAdapter(client), calls

    yield make
    for client in clients:
        client.close()


def test_continuation_wire_replay_and_usage(make_adapter):
    thought = {"thought": True, "text": "Think", "thoughtSignature": "opaque-signature"}
    adapter, calls = make_adapter(
        [
            reply(
                "CONTINUATION",
                [thought],
                "token1",
                {
                    "promptTokenCount": 10,
                    "thoughtsTokenCount": 20,
                    "cachedContentTokenCount": 3,
                },
            ),
            reply(
                "CONTINUATION",
                [{"text": "AC"}],
                "token2",
                {"promptTokenCount": 11, "candidatesTokenCount": 1},
            ),
            reply(
                "STOP",
                [{"text": "TION1"}],
                usage={
                    "promptTokenCount": 12,
                    "candidatesTokenCount": 2,
                    "thoughtsTokenCount": None,
                },
            ),
        ]
    )
    req = request()
    original = deepcopy(req)
    result = adapter.invoke(req)
    assert result.output_text == "ACTION1"
    assert result.reasoning_text == "Think"
    assert result.usage.input_tokens == 33
    assert result.usage.output_tokens == 23
    assert result.usage.reasoning_tokens == 20
    assert result.usage.total_tokens == 56
    assert result.usage.cached_tokens == 3
    assert len(result.raw_response["slices"]) == 3
    bodies = [json.loads(call.content) for call in calls]
    assert "continuationToken" not in bodies[0]
    assert bodies[1]["continuationToken"] == "token1"
    assert bodies[2]["continuationToken"] == "token2"
    assert bodies[1]["contents"] == bodies[0]["contents"] + [
        {"role": "model", "parts": [thought]}
    ]
    assert bodies[2]["contents"][-1]["parts"] == [thought, {"text": "AC"}]
    for body, call in zip(bodies, calls):
        assert body["generationConfig"]["maxOutputTokens"] == 1000000
        assert "continuation" not in body["generationConfig"]
        assert body["systemInstruction"]["parts"][0]["text"] == "Play the game."
        assert call.extensions["timeout"]["read"] == 3600
    assert req == original


@pytest.mark.parametrize("reason", ["STOP", "MAX_TOKENS"])
def test_terminal_and_timeout_override(make_adapter, reason):
    adapter, calls = make_adapter([reply(reason, [{"text": "ACTION1"}])])
    assert (
        adapter.invoke(request(http_options={"timeout": 42000})).output_text
        == "ACTION1"
    )
    assert calls[0].extensions["timeout"]["read"] == 42


@pytest.mark.parametrize(
    "last", [reply("CONTINUATION"), reply("SAFETY"), {"candidates": []}]
)
def test_protocol_failure_preserves_observed_usage(make_adapter, last):
    adapter, calls = make_adapter(
        [
            reply(
                "CONTINUATION",
                [{"text": "ACTION1"}],
                "token",
                {"promptTokenCount": 5, "thoughtsTokenCount": 7},
            ),
            last,
        ]
    )
    with pytest.raises(InvalidProviderResponseError) as error:
        adapter.invoke(request())
    assert error.value.usage.total_tokens == 12
    assert len(error.value.response["slices"]) == 2
    assert len(calls) == 2


def test_repeated_token(make_adapter):
    adapter, calls = make_adapter([reply("CONTINUATION", token="same")] * 2)
    with pytest.raises(InvalidProviderResponseError, match="repeated"):
        adapter.invoke(request())
    assert len(calls) == 2


def test_transport_failure_preserves_usage(make_adapter):
    adapter, calls = make_adapter(
        [
            reply("CONTINUATION", token="token", usage={"promptTokenCount": 5}),
            httpx.ReadTimeout("deadline exceeded"),
        ]
    )
    with pytest.raises(InvalidProviderResponseError) as error:
        adapter.invoke(request())
    assert error.value.usage.input_tokens == 5
    assert isinstance(error.value.__cause__, httpx.ReadTimeout)
    assert len(calls) == 2


def test_empty_terminal_preserves_usage(make_adapter):
    adapter, _ = make_adapter([reply("MAX_TOKENS", usage={"thoughtsTokenCount": 10})])
    with pytest.raises(EmptyResponseError) as error:
        adapter.invoke(request())
    assert error.value.usage.reasoning_tokens == 10


@pytest.mark.parametrize("budget", [None, 0, -1])
def test_continuation_without_budget_fails_with_usage(make_adapter, budget):
    adapter, calls = make_adapter(
        [reply("CONTINUATION", token="token", usage={"thoughtsTokenCount": 10})]
    )
    with pytest.raises(
        InvalidProviderResponseError, match="explicit positive"
    ) as error:
        adapter.invoke(request(max_output_tokens=budget))
    assert error.value.usage.reasoning_tokens == 10
    assert len(calls) == 1


def test_single_response_without_explicit_budget(make_adapter):
    adapter, calls = make_adapter([reply("STOP", [{"text": "ACTION1"}])])
    assert adapter.invoke(request(max_output_tokens=None)).output_text == "ACTION1"
    assert len(calls) == 1


def test_existing_path(make_adapter):
    adapter, calls = make_adapter([reply("STOP", [{"text": "ACTION1"}])])
    assert adapter.invoke(request()).output_text == "ACTION1"
    assert len(calls) == 1
