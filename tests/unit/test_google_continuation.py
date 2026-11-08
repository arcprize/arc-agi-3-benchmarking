"""Validate actual SDK serialization against an in-memory HTTP transport."""

import base64
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


def stream_response(*events):
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        text="".join(f"data: {json.dumps(event)}\n\n" for event in events),
    )


@pytest.fixture
def make_stream_adapter():
    clients = []

    def make(responses):
        responses = iter(responses)
        calls = []

        def handle(req):
            calls.append(req)
            response = next(responses)
            if isinstance(response, Exception):
                raise response
            return response

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


def test_stream_uses_latest_cumulative_usage_and_keeps_thought_parts(
    make_stream_adapter,
):
    signature = base64.b64encode(b"opaque-signature").decode()
    thought = {"thought": True, "text": "Think", "thoughtSignature": signature}
    adapter, calls = make_stream_adapter(
        [
            stream_response(
                reply(
                    "STOP",
                    [thought],
                    usage={
                        "promptTokenCount": 10,
                        "thoughtsTokenCount": 2,
                        "totalTokenCount": 12,
                    },
                ),
                reply(
                    "STOP",
                    [{"text": "ACTION1"}],
                    usage={
                        "promptTokenCount": 10,
                        "candidatesTokenCount": 2,
                        "thoughtsTokenCount": 3,
                        "cachedContentTokenCount": 4,
                        "totalTokenCount": 16,
                    },
                ),
            )
        ]
    )
    result = adapter.invoke(request(stream=True))
    assert result.output_text == "ACTION1"
    assert result.reasoning_text == "Think"
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5
    assert result.usage.reasoning_tokens == 3
    assert result.usage.cached_tokens == 4
    assert result.usage.total_tokens == 16
    assert result.raw_response["slices"][0]["candidates"][0]["content"]["parts"][0][
        "thoughtSignature"
    ] == signature
    assert calls[0].url.path.endswith(":streamGenerateContent")
    assert calls[0].extensions["timeout"]["read"] == 3600


def test_stream_continuation_adds_hop_usage_once(make_stream_adapter):
    token = base64.b64encode(b"token1").decode()
    adapter, calls = make_stream_adapter(
        [
            stream_response(
                reply(
                    "CONTINUATION",
                    [{"text": "AC"}],
                    token,
                    {"promptTokenCount": 10, "candidatesTokenCount": 1, "totalTokenCount": 11},
                )
            ),
            stream_response(
                reply(
                    "STOP",
                    [{"text": "TION1"}],
                    usage={
                        "promptTokenCount": 11,
                        "candidatesTokenCount": 2,
                        "thoughtsTokenCount": 4,
                        "totalTokenCount": 17,
                    },
                )
            ),
        ]
    )
    result = adapter.invoke(request(stream=True))
    assert result.output_text == "ACTION1"
    assert result.usage.input_tokens == 21
    assert result.usage.output_tokens == 7
    assert result.usage.reasoning_tokens == 4
    assert result.usage.total_tokens == 28
    assert len(result.raw_response["slices"]) == 2
    bodies = [json.loads(call.content) for call in calls]
    assert "continuationToken" not in bodies[0]
    assert bodies[1]["continuationToken"] == token
    assert bodies[1]["contents"] == bodies[0]["contents"]


def test_stream_interruption_preserves_last_reported_usage(make_stream_adapter):
    class InterruptedStream(httpx.SyncByteStream):
        def __iter__(self):
            event = reply(
                "STOP",
                [{"text": "partial"}],
                usage={"promptTokenCount": 5, "thoughtsTokenCount": 7},
            )
            yield f"data: {json.dumps(event)}\n\n".encode()
            raise httpx.ReadError("stream interrupted")

    adapter, calls = make_stream_adapter(
        [
            httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=InterruptedStream(),
            )
        ]
    )
    with pytest.raises(InvalidProviderResponseError) as error:
        adapter.invoke(request(stream=True))
    assert error.value.usage.input_tokens == 5
    assert error.value.usage.output_tokens == 7
    assert error.value.usage.reasoning_tokens == 7
    assert len(error.value.response["slices"]) == 1
    assert len(calls) == 1


def test_stream_requires_provider_usage(make_stream_adapter):
    adapter, _ = make_stream_adapter(
        [stream_response(reply("STOP", [{"text": "ACTION1"}]))]
    )
    with pytest.raises(InvalidProviderResponseError, match="without usage metadata"):
        adapter.invoke(request(stream=True))


def test_stream_preserves_native_contents_on_wire(make_stream_adapter):
    adapter, calls = make_stream_adapter(
        [
            stream_response(
                reply(
                    "STOP",
                    [{"text": "ACTION1"}],
                    usage={"promptTokenCount": 5, "candidatesTokenCount": 2},
                )
            )
        ]
    )
    req = request(stream=True)
    req.native_input = [
        {
            "role": "model",
            "parts": [{"text": "prior", "thoughtSignature": "c2ln"}],
        },
        {"role": "user", "parts": [{"text": "frame"}]},
    ]
    original = deepcopy(req)
    assert adapter.invoke(req).output_text == "ACTION1"
    assert json.loads(calls[0].content)["contents"] == req.native_input
    assert req == original


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

@pytest.mark.parametrize('status', [500, 503])
def test_real_sdk_server_errors_are_transient(status):
    from benchmarking.exceptions import TransientProviderError

    def handle(req):
        return httpx.Response(status, json={
            'error': {'code': status, 'message': 'Temporary failure', 'status': 'INTERNAL'}
        })

    with genai.Client(api_key='test-key', http_options=types.HttpOptions(
        httpx_client=httpx.Client(transport=httpx.MockTransport(handle))
    )) as client:
        with pytest.raises(TransientProviderError) as caught:
            GoogleGenAIGenerateContentAdapter(client).invoke(request())
        assert caught.value.__cause__.code == status


def test_transport_timeout_is_transient(make_adapter):
    from benchmarking.exceptions import TransientProviderError
    adapter, _ = make_adapter([httpx.ReadTimeout('timeout')])
    with pytest.raises(TransientProviderError):
        adapter.invoke(request())


def test_permanent_google_error_is_not_transient():
    from google.genai.errors import ClientError

    def handle(req):
        return httpx.Response(403, json={
            'error': {'code': 403, 'message': 'Denied', 'status': 'PERMISSION_DENIED'}
        })

    with genai.Client(api_key='test-key', http_options=types.HttpOptions(
        httpx_client=httpx.Client(transport=httpx.MockTransport(handle))
    )) as client:
        with pytest.raises(ClientError):
            GoogleGenAIGenerateContentAdapter(client).invoke(request())
