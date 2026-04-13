from types import SimpleNamespace

import pytest

from benchmarking.exceptions import EmptyResponseError
from benchmarking.runtime_adapters import (
    OpenAIChatCompletionsAdapter,
    OpenAIResponsesAdapter,
    build_model_runtime_adapter,
)
from benchmarking.runtime_models import Message, ModelRequest


class _FakeChatCompletions:
    def __init__(self, response: object) -> None:
        self._response = response
        self.calls: list[dict] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self._response


class _FakeResponses:
    def __init__(self, response: object) -> None:
        self._response = response
        self.calls: list[dict] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self._response


class _FakeChatOpenAIClient:
    def __init__(self, response: object) -> None:
        self.chat = SimpleNamespace(completions=_FakeChatCompletions(response))


class _FakeResponsesOpenAIClient:
    def __init__(self, response: object) -> None:
        self.responses = _FakeResponses(response)


def _chat_response(
    *,
    content: str = "MOVE_LEFT",
    reasoning: str | None = "because",
    prompt_tokens: int = 11,
    completion_tokens: int = 7,
    total_tokens: int = 18,
    reasoning_tokens: int = 3,
    cached_tokens: int = 5,
    cache_write_tokens: int = 2,
    cost: float = 0.42,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    reasoning=reasoning,
                )
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            prompt_tokens_details=SimpleNamespace(
                cached_tokens=cached_tokens,
                cache_write_tokens=cache_write_tokens,
            ),
            completion_tokens_details=SimpleNamespace(
                reasoning_tokens=reasoning_tokens,
            ),
            model_extra={"cost": cost, "cost_details": {"provider_cost": cost}},
        ),
    )


def _responses_output(
    *,
    text: str = "MOVE_LEFT",
    reasoning_summary: str | None = "because",
    input_tokens: int = 11,
    output_tokens: int = 7,
    total_tokens: int = 18,
    reasoning_tokens: int = 3,
    cached_tokens: int = 5,
    cache_write_tokens: int = 2,
    cost: float = 0.42,
) -> SimpleNamespace:
    output = [
        SimpleNamespace(
            type="message",
            role="assistant",
            content=[SimpleNamespace(type="output_text", text=text)],
        )
    ]
    if reasoning_summary is not None:
        output.insert(
            0,
            SimpleNamespace(
                type="reasoning",
                summary=[SimpleNamespace(text=reasoning_summary)],
                content=[],
            ),
        )

    return SimpleNamespace(
        output=output,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            input_tokens_details=SimpleNamespace(
                cached_tokens=cached_tokens,
                cache_write_tokens=cache_write_tokens,
            ),
            output_tokens_details=SimpleNamespace(
                reasoning_tokens=reasoning_tokens,
            ),
            model_extra={"cost": cost, "cost_details": {"provider_cost": cost}},
        ),
    )


def _chat_request() -> ModelRequest:
    return ModelRequest(
        messages=[
            Message(role="system", content="You are playing a game."),
            Message(role="user", content="frame"),
        ],
        request_config={"model": "gpt-5.4", "max_completion_tokens": 128},
    )


def _responses_request() -> ModelRequest:
    return ModelRequest(
        messages=[
            Message(role="system", content="You are playing a game."),
            Message(role="user", content="frame 1"),
            Message(role="assistant", content="MOVE_LEFT"),
            Message(role="user", content="frame 2"),
        ],
        request_config={"model": "gpt-5.4", "max_output_tokens": 128},
    )


@pytest.mark.unit
class TestOpenAIChatCompletionsAdapter:
    def test_translates_request_messages_into_chat_messages(self):
        client = _FakeChatOpenAIClient(_chat_response())
        adapter = OpenAIChatCompletionsAdapter(client)

        adapter.invoke(_chat_request())

        assert client.chat.completions.calls == [
            {
                "messages": [
                    {"role": "system", "content": "You are playing a game."},
                    {"role": "user", "content": "frame"},
                ],
                "model": "gpt-5.4",
                "max_completion_tokens": 128,
            }
        ]

    def test_preserves_analysis_mode_prompt_and_replay_content_in_chat_messages(self):
        client = _FakeChatOpenAIClient(_chat_response())
        adapter = OpenAIChatCompletionsAdapter(client)
        replay_content = (
            "<reasoning_summary>\n"
            "inspect the top row\n"
            "</reasoning_summary>\n\n"
            "MOVE_LEFT"
        )

        adapter.invoke(
            ModelRequest(
                messages=[
                    Message(
                        role="system",
                        content=(
                            "Use <reasoning_summary> blocks as compact helper context."
                        ),
                    ),
                    Message(role="user", content="frame 1"),
                    Message(role="assistant", content=replay_content),
                    Message(role="user", content="frame 2"),
                ],
                request_config={"model": "gpt-5.4"},
            )
        )

        assert client.chat.completions.calls[0]["messages"] == [
            {
                "role": "system",
                "content": "Use <reasoning_summary> blocks as compact helper context.",
            },
            {"role": "user", "content": "frame 1"},
            {"role": "assistant", "content": replay_content},
            {"role": "user", "content": "frame 2"},
        ]

    def test_passes_request_config_kwargs_including_model(self):
        client = _FakeChatOpenAIClient(_chat_response())
        adapter = OpenAIChatCompletionsAdapter(client)

        adapter.invoke(
            ModelRequest(
                messages=[Message(role="user", content="frame")],
                request_config={
                    "model": "gpt-5.4",
                    "max_completion_tokens": 99,
                    "reasoning_effort": "high",
                },
            )
        )

        assert client.chat.completions.calls[0]["model"] == "gpt-5.4"
        assert client.chat.completions.calls[0]["max_completion_tokens"] == 99
        assert client.chat.completions.calls[0]["reasoning_effort"] == "high"

    def test_returns_normalized_assistant_text(self):
        adapter = OpenAIChatCompletionsAdapter(
            _FakeChatOpenAIClient(_chat_response(content="RESET"))
        )

        response = adapter.invoke(_chat_request())

        assert response.output_text == "RESET"

    def test_returns_normalized_reasoning_text(self):
        adapter = OpenAIChatCompletionsAdapter(
            _FakeChatOpenAIClient(_chat_response(reasoning="inspect the top row"))
        )

        response = adapter.invoke(_chat_request())

        assert response.reasoning_text == "inspect the top row"

    def test_normalizes_usage(self):
        adapter = OpenAIChatCompletionsAdapter(
            _FakeChatOpenAIClient(
                _chat_response(
                    prompt_tokens=123,
                    completion_tokens=45,
                    total_tokens=168,
                    reasoning_tokens=11,
                    cached_tokens=7,
                    cache_write_tokens=3,
                    cost=0.75,
                )
            )
        )

        response = adapter.invoke(_chat_request())

        assert response.usage.input_tokens == 123
        assert response.usage.output_tokens == 45
        assert response.usage.total_tokens == 168
        assert response.usage.reasoning_tokens == 11
        assert response.usage.cached_tokens == 7
        assert response.usage.cache_write_tokens == 3
        assert response.usage.cost == 0.75
        assert response.usage.cost_details == {"provider_cost": 0.75}

    def test_empty_choices_response_raises_empty_response_error(self):
        adapter = OpenAIChatCompletionsAdapter(
            _FakeChatOpenAIClient(SimpleNamespace(choices=[]))
        )

        with pytest.raises(EmptyResponseError) as exc_info:
            adapter.invoke(_chat_request())

        assert str(exc_info.value) == "API returned 200 with empty choices."
        assert exc_info.value.response is not None


@pytest.mark.unit
class TestOpenAIResponsesAdapter:
    def test_translates_system_user_assistant_conversation(self):
        client = _FakeResponsesOpenAIClient(_responses_output())
        adapter = OpenAIResponsesAdapter(client)

        adapter.invoke(_responses_request())

        assert client.responses.calls == [
            {
                "model": "gpt-5.4",
                "max_output_tokens": 128,
                "instructions": "You are playing a game.",
                "input": [
                    {"role": "user", "content": "frame 1"},
                    {"role": "assistant", "content": "MOVE_LEFT"},
                    {"role": "user", "content": "frame 2"},
                ],
            }
        ]

    def test_preserves_analysis_mode_prompt_and_replay_content_in_responses_input(self):
        client = _FakeResponsesOpenAIClient(_responses_output())
        adapter = OpenAIResponsesAdapter(client)
        replay_content = (
            "<reasoning_summary>\n"
            "inspect the top row\n"
            "</reasoning_summary>\n\n"
            "MOVE_LEFT"
        )

        adapter.invoke(
            ModelRequest(
                messages=[
                    Message(
                        role="system",
                        content=(
                            "Use <reasoning_summary> blocks as compact helper context."
                        ),
                    ),
                    Message(role="user", content="frame 1"),
                    Message(role="assistant", content=replay_content),
                    Message(role="user", content="frame 2"),
                ],
                request_config={"model": "gpt-5.4"},
            )
        )

        assert client.responses.calls[0]["instructions"] == (
            "Use <reasoning_summary> blocks as compact helper context."
        )
        assert client.responses.calls[0]["input"] == [
            {"role": "user", "content": "frame 1"},
            {"role": "assistant", "content": replay_content},
            {"role": "user", "content": "frame 2"},
        ]

    def test_passes_request_config_kwargs_including_model(self):
        client = _FakeResponsesOpenAIClient(_responses_output())
        adapter = OpenAIResponsesAdapter(client)

        adapter.invoke(
            ModelRequest(
                messages=[Message(role="user", content="frame")],
                request_config={
                    "model": "gpt-5.4",
                    "max_output_tokens": 99,
                    "store": False,
                },
            )
        )

        assert client.responses.calls[0]["model"] == "gpt-5.4"
        assert client.responses.calls[0]["max_output_tokens"] == 99
        assert client.responses.calls[0]["store"] is False

    def test_passes_request_reasoning_config(self):
        client = _FakeResponsesOpenAIClient(_responses_output())
        adapter = OpenAIResponsesAdapter(client)

        adapter.invoke(
            ModelRequest(
                messages=[Message(role="user", content="frame")],
                request_config={
                    "model": "gpt-5.4",
                    "reasoning": {"effort": "high"},
                },
            )
        )

        assert client.responses.calls[0]["reasoning"] == {"effort": "high"}

    def test_extracts_assistant_text_from_responses_output(self):
        adapter = OpenAIResponsesAdapter(
            _FakeResponsesOpenAIClient(_responses_output(text="RESET"))
        )

        response = adapter.invoke(_responses_request())

        assert response.output_text == "RESET"

    def test_extracts_human_readable_reasoning_summary_if_present(self):
        adapter = OpenAIResponsesAdapter(
            _FakeResponsesOpenAIClient(
                _responses_output(reasoning_summary="inspect the top row")
            )
        )

        response = adapter.invoke(_responses_request())

        assert response.reasoning_text == "inspect the top row"

    def test_normalizes_usage_fields(self):
        adapter = OpenAIResponsesAdapter(
            _FakeResponsesOpenAIClient(
                _responses_output(
                    input_tokens=123,
                    output_tokens=45,
                    total_tokens=168,
                    reasoning_tokens=11,
                    cached_tokens=7,
                    cache_write_tokens=3,
                    cost=0.75,
                )
            )
        )

        response = adapter.invoke(_responses_request())

        assert response.usage.input_tokens == 123
        assert response.usage.output_tokens == 45
        assert response.usage.total_tokens == 168
        assert response.usage.reasoning_tokens == 11
        assert response.usage.cached_tokens == 7
        assert response.usage.cache_write_tokens == 3
        assert response.usage.cost == 0.75
        assert response.usage.cost_details == {"provider_cost": 0.75}

    def test_empty_or_malformed_output_raises_empty_response_error(self):
        adapter = OpenAIResponsesAdapter(
            _FakeResponsesOpenAIClient(SimpleNamespace(output=[]))
        )

        with pytest.raises(EmptyResponseError) as exc_info:
            adapter.invoke(_responses_request())

        assert str(exc_info.value) == "API returned 200 with empty output."
        assert exc_info.value.response is not None

    def test_maps_first_system_message_into_instructions(self):
        client = _FakeResponsesOpenAIClient(_responses_output())
        adapter = OpenAIResponsesAdapter(client)

        adapter.invoke(
            ModelRequest(
                messages=[
                    Message(role="system", content="System prompt"),
                    Message(role="user", content="frame"),
                ],
                request_config={"model": "gpt-5.4"},
            )
        )

        assert client.responses.calls[0]["instructions"] == "System prompt"

    def test_sends_remaining_turns_in_input(self):
        client = _FakeResponsesOpenAIClient(_responses_output())
        adapter = OpenAIResponsesAdapter(client)

        adapter.invoke(_responses_request())

        assert client.responses.calls[0]["input"] == [
            {"role": "user", "content": "frame 1"},
            {"role": "assistant", "content": "MOVE_LEFT"},
            {"role": "user", "content": "frame 2"},
        ]

    def test_omits_previous_response_id_and_conversation_for_manual_rolling(self):
        client = _FakeResponsesOpenAIClient(_responses_output())
        adapter = OpenAIResponsesAdapter(client)

        adapter.invoke(
            ModelRequest(
                messages=[Message(role="user", content="frame")],
                request_config={
                    "model": "gpt-5.4",
                    "previous_response_id": "resp_123",
                    "conversation": "conv_123",
                },
            )
        )

        assert "previous_response_id" not in client.responses.calls[0]
        assert "conversation" not in client.responses.calls[0]


@pytest.mark.unit
class TestBuildModelRuntimeAdapter:
    def test_selects_chat_completions_adapter_from_runtime_tuple(self):
        adapter = build_model_runtime_adapter(
            client=_FakeChatOpenAIClient(_chat_response()),
            runtime_config={
                "sdk": "openai-python",
                "api": "chat_completions",
                "state": "manual_rolling",
            },
            config_id="chat-config",
        )

        assert isinstance(adapter, OpenAIChatCompletionsAdapter)

    def test_selects_responses_adapter_from_runtime_tuple(self):
        adapter = build_model_runtime_adapter(
            client=_FakeResponsesOpenAIClient(_responses_output()),
            runtime_config={
                "sdk": "openai-python",
                "api": "responses",
                "state": "manual_rolling",
            },
            config_id="responses-config",
        )

        assert isinstance(adapter, OpenAIResponsesAdapter)

    def test_unsupported_runtime_state_fails_clearly(self):
        with pytest.raises(
            ValueError,
            match=(
                "Model config 'chat-config' uses runtime.state='previous_response_id', "
                "but only 'manual_rolling' is supported in phase 3."
            ),
        ):
            build_model_runtime_adapter(
                client=_FakeChatOpenAIClient(_chat_response()),
                runtime_config={
                    "sdk": "openai-python",
                    "api": "chat_completions",
                    "state": "previous_response_id",
                },
                config_id="chat-config",
            )
