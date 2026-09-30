import json

import httpx
import pytest
from openai import OpenAI

from benchmarking.exceptions import EmptyResponseError
from benchmarking.runtime_adapters import OpenAIChatCompletionsAdapter
from benchmarking.runtime_models import Message, ModelRequest


def _chunk(delta=None, *, finish=None, usage=None):
    return {
        "id": "chat-1", "created": 1, "model": "grok-4.7",
        "object": "chat.completion.chunk",
        "choices": [] if usage else [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
        "usage": usage,
    }


USAGE = {
    "prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30,
    "prompt_tokens_details": {"cached_tokens": 5},
    "completion_tokens_details": {"reasoning_tokens": 6},
}


def _invoke(events, update=None):
    calls = []

    def handle(request):
        calls.append(json.loads(request.content))
        text = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        return httpx.Response(200, content=text + "data: [DONE]\n\n",
                              headers={"content-type": "text/event-stream"})

    with OpenAI(api_key="test", base_url="https://api.x.ai/v1", max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
        result = OpenAIChatCompletionsAdapter(client).invoke(ModelRequest(
            messages=[Message(role="user", content="frame")],
            request_config={"model": "grok-4.7", "stream": True, **(update or {})},
        ))
    return result, calls


@pytest.mark.unit
class TestChatStreaming:
    def test_assembles_text_reasoning_and_usage_through_sdk(self):
        result, calls = _invoke([
            _chunk({"reasoning_content": "Think "}),
            _chunk({"reasoning_content": "carefully."}),
            _chunk({"content": "ACTION"}),
            _chunk({"content": "1"}, finish="stop"),
            _chunk(usage=USAGE),
        ])
        assert result.output_text == "ACTION1"
        assert result.reasoning_text == "Think carefully."
        assert result.usage.input_tokens == 20
        assert result.usage.output_tokens == 10
        assert result.usage.cached_tokens == 5
        assert result.usage.reasoning_tokens == 6
        assert calls[0]["stream_options"] == {"include_usage": True}

    @pytest.mark.parametrize("finish", [None, "length", "content_filter", "tool_calls"])
    def test_rejects_unfinished_or_refused_answer_and_keeps_usage(self, finish):
        with pytest.raises(EmptyResponseError) as exc:
            _invoke([_chunk({"content": "ACTION1"}, finish=finish), _chunk(usage=USAGE)])
        assert exc.value.usage.total_tokens == 30

    def test_rejects_reasoning_only_response(self):
        with pytest.raises(EmptyResponseError, match="no visible action"):
            _invoke([_chunk({"reasoning_content": "thinking"}, finish="stop"), _chunk(usage=USAGE)])

    def test_requires_usage(self):
        with pytest.raises(EmptyResponseError, match="without valid token usage"):
            _invoke([_chunk({"content": "ACTION1"}, finish="stop")])

    @pytest.mark.parametrize("update", [
        {"stream": "true"}, {"n": 2},
        {"stream_options": {"include_usage": False}}, {"stream_options": None},
    ])
    def test_rejects_unsupported_settings_before_io(self, update):
        with pytest.raises(ValueError):
            OpenAIChatCompletionsAdapter(None).invoke(ModelRequest(
                messages=[], request_config={"stream": True, **update},
            ))

    def test_sdk_provider_error_is_not_a_partial_success(self):
        with pytest.raises(EmptyResponseError, match="stream failed"):
            _invoke([_chunk({"content": "ACTION1"}), {"error": {"message": "private error"}}])

    def test_transport_failure_preserves_usage_and_closes_stream(self):
        from openai.types.chat import ChatCompletionChunk

        from benchmarking.chat_streaming import consume_chat_stream

        class Stream:
            closed = False

            def __iter__(self):
                yield ChatCompletionChunk.model_validate(_chunk(usage=USAGE))
                raise httpx.ReadError("private transport error")

            def close(self):
                self.closed = True

        stream = Stream()
        with pytest.raises(EmptyResponseError) as exc:
            consume_chat_stream(stream)
        assert stream.closed
        assert exc.value.usage.total_tokens == 30
        assert "private transport error" not in str(exc.value)
