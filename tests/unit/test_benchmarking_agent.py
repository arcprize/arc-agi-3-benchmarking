from types import SimpleNamespace

from arcengine import FrameData, GameAction, GameState
import pytest

from benchmarking.agent import BenchmarkingAgent
from benchmarking.runtime_adapters import (
    OpenAIChatCompletionsAdapter,
    OpenAIResponsesAdapter,
)
from benchmarking.runtime_models import (
    Message,
    ModelRequest,
    ModelResponse,
    NormalizedUsage,
)


class _FakeAdapter:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = responses
        self.requests: list[ModelRequest] = []

    def invoke(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self._responses.pop(0)


class _FakeChatCompletions:
    def __init__(self, response: object) -> None:
        self._response = response

    def create(self, **_kwargs: object) -> object:
        return self._response


class _FakeResponses:
    def __init__(self, response: object) -> None:
        self._response = response

    def create(self, **_kwargs: object) -> object:
        return self._response


class _FakeChatClient:
    def __init__(self, response: object) -> None:
        self.chat = SimpleNamespace(completions=_FakeChatCompletions(response))


class _FakeResponsesClient:
    def __init__(self, response: object) -> None:
        self.responses = _FakeResponses(response)


def _agent_for_request_kwargs(request_kwargs: dict) -> BenchmarkingAgent:
    agent = BenchmarkingAgent.__new__(BenchmarkingAgent)
    agent.conversation = []
    agent._request_kwargs = request_kwargs
    agent.MAX_CONTEXT_LENGTH = 1_000
    agent.ESTIMATED_CHARS_PER_TOKEN = 1.0
    agent.MAX_RETRIES = 2
    agent.analysis_mode = False
    agent.token_counter = 0
    return agent


def _agent_for_choose_action(
    *,
    analysis_mode: bool,
    responses: list[ModelResponse],
) -> BenchmarkingAgent:
    agent = _agent_for_request_kwargs({"model": "gpt-5.4"})
    agent.analysis_mode = analysis_mode
    agent._adapter = _FakeAdapter(responses)
    agent.MODEL = "gpt-5.4"
    agent._pricing = {}
    agent.step_counter = 0
    agent._level_action_budgets = []
    agent._level_action_counter = 0
    agent._last_levels_completed = 0
    agent._level_just_advanced = False
    agent.action_counter = 0
    agent._previous_action = None
    agent.MAX_ANIMATION_FRAMES = 7
    agent._saved_steps = []
    agent._save_step = agent._saved_steps.append
    return agent


def _playable_frame() -> FrameData:
    return FrameData(
        frame=[[[0, 1], [1, 0]]],
        state=GameState.NOT_FINISHED,
        levels_completed=0,
        available_actions=[GameAction.ACTION1.value],
    )


def _terminal_frame(state: GameState) -> FrameData:
    return FrameData(
        frame=[[[3, 3], [3, 3]]],
        state=state,
        levels_completed=0,
        available_actions=[GameAction.ACTION1.value],
    )


def _chat_response(text: str = "RESET") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=text,
                    reasoning="restart",
                )
            )
        ],
        usage=SimpleNamespace(total_tokens=6),
    )


def _responses_response(text: str = "RESET") -> SimpleNamespace:
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="reasoning",
                summary=[SimpleNamespace(text="restart")],
                content=[],
            ),
            SimpleNamespace(
                type="message",
                role="assistant",
                content=[SimpleNamespace(type="output_text", text=text)],
            ),
        ],
        usage=SimpleNamespace(total_tokens=6),
    )


@pytest.mark.unit
class TestBenchmarkingAgentModelRequests:
    def test_build_system_prompt_uses_baseline_prompt_in_normal_mode(self):
        agent = _agent_for_request_kwargs({"model": "gpt-5.4"})

        system_prompt = agent._build_system_prompt()

        assert "You are playing a game. Your goal is to win." in system_prompt
        assert "<reasoning_summary>" not in system_prompt
        assert "compact helper context" not in system_prompt

    def test_build_system_prompt_uses_helper_prompt_in_analysis_mode(self):
        agent = _agent_for_request_kwargs({"model": "gpt-5.4"})
        agent.analysis_mode = True

        system_prompt = agent._build_system_prompt()

        assert "You are playing a game. Your goal is to win." in system_prompt
        assert "<reasoning_summary>" in system_prompt
        assert "compact helper context" in system_prompt

    def test_build_assistant_turn_content_returns_plain_output_in_normal_mode(self):
        agent = _agent_for_request_kwargs({"model": "gpt-5.4"})

        assert (
            agent._build_assistant_turn_content("RESET", "restart because blocked")
            == "RESET"
        )

    def test_build_assistant_turn_content_wraps_reasoning_in_analysis_mode(self):
        agent = _agent_for_request_kwargs({"model": "gpt-5.4"})
        agent.analysis_mode = True

        assert agent._build_assistant_turn_content(
            "RESET",
            "restart because blocked",
        ) == (
            "<reasoning_summary>\n"
            "restart because blocked\n"
            "</reasoning_summary>\n\n"
            "RESET\n"
        )

    @pytest.mark.parametrize("reasoning_text", [None, ""])
    def test_build_assistant_turn_content_returns_plain_output_without_reasoning(
        self,
        reasoning_text,
    ):
        agent = _agent_for_request_kwargs({"model": "gpt-5.4"})
        agent.analysis_mode = True

        assert agent._build_assistant_turn_content("RESET", reasoning_text) == "RESET"

    def test_builds_model_request_from_conversation_state(self):
        agent = _agent_for_request_kwargs(
            {"model": "gpt-5.4", "max_completion_tokens": 128}
        )
        agent.conversation = [
            {"role": "system", "content": "You are playing a game."},
            {"role": "user", "content": "frame 1"},
            {"role": "assistant", "content": "RESET"},
        ]

        model_request = agent._build_model_request()

        assert model_request == ModelRequest(
            messages=[
                Message(role="system", content="You are playing a game."),
                Message(role="user", content="frame 1"),
                Message(role="assistant", content="RESET"),
            ],
            request_config={"model": "gpt-5.4", "max_completion_tokens": 128},
        )

    def test_trimmed_history_is_reflected_in_outbound_request(self):
        agent = _agent_for_request_kwargs(
            {"model": "gpt-5.4", "max_completion_tokens": 128}
        )
        agent.MAX_CONTEXT_LENGTH = 40
        agent.conversation = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "old-user-turn"},
            {"role": "assistant", "content": "old-assistant-turn"},
            {"role": "user", "content": "new-user-turn"},
        ]

        agent._trim_to_fit_context()
        model_request = agent._build_model_request()

        assert [message.model_dump() for message in model_request.messages] == [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "new-user-turn"},
        ]

    def test_responses_request_kwargs_use_same_agent_owned_trimming_strategy(self):
        agent = _agent_for_request_kwargs(
            {
                "model": "gpt-5.4",
                "max_output_tokens": 128,
                "reasoning": {"effort": "high"},
            }
        )
        agent.MAX_CONTEXT_LENGTH = 40
        agent.conversation = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "old-user-turn"},
            {"role": "assistant", "content": "old-assistant-turn"},
            {"role": "user", "content": "new-user-turn"},
        ]

        agent._trim_to_fit_context()
        model_request = agent._build_model_request()

        assert model_request == ModelRequest(
            messages=[
                Message(role="system", content="system"),
                Message(role="user", content="new-user-turn"),
            ],
            request_config={
                "model": "gpt-5.4",
                "max_output_tokens": 128,
                "reasoning": {"effort": "high"},
            },
        )

    def test_build_model_request_preserves_replay_shaped_assistant_content(self):
        agent = _agent_for_request_kwargs({"model": "gpt-5.4"})
        agent.analysis_mode = True
        replay_content = (
            "<reasoning_summary>\n"
            "I compared the options and chose reset.\n"
            "</reasoning_summary>\n\n"
            "RESET"
        )
        agent.conversation = [
            {"role": "system", "content": "You are playing a game."},
            {"role": "user", "content": "frame 1"},
            {"role": "assistant", "content": replay_content},
        ]

        model_request = agent._build_model_request()

        assert [message.model_dump() for message in model_request.messages] == [
            {"role": "system", "content": "You are playing a game."},
            {"role": "user", "content": "frame 1"},
            {"role": "assistant", "content": replay_content},
        ]


@pytest.mark.unit
class TestBenchmarkingAgentRetries:
    def test_unparseable_assistant_response_retries_until_action_is_parsed(self):
        agent = _agent_for_request_kwargs(
            {"model": "gpt-5.4", "max_completion_tokens": 128}
        )
        agent.conversation = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "frame"},
        ]
        responses = [
            ModelResponse(
                output_text="not an action",
                usage=NormalizedUsage(total_tokens=4),
            ),
            ModelResponse(
                output_text="RESET",
                usage=NormalizedUsage(total_tokens=6),
            ),
        ]

        def fake_call_api(_request: ModelRequest) -> ModelResponse:
            return responses.pop(0)

        agent._call_api = fake_call_api

        model_response, action, retries, messages_sent = agent._request_with_retries(
            [GameAction.RESET]
        )

        assert model_response.output_text == "RESET"
        assert model_response.usage.total_tokens == 10
        assert action == GameAction.RESET
        assert retries == 1
        assert messages_sent == [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "frame"},
        ]
        assert agent.token_counter == 10

    def test_normalized_responses_output_parses_action_like_chat_output(self):
        agent = _agent_for_request_kwargs(
            {
                "model": "gpt-5.4",
                "max_output_tokens": 128,
                "reasoning": {"effort": "high"},
            }
        )
        agent.conversation = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "frame"},
        ]
        agent._adapter = _FakeAdapter(
            [
                ModelResponse(
                    output_text="thinking without a valid action",
                    reasoning_text="no action yet",
                    usage=NormalizedUsage(total_tokens=4),
                ),
                ModelResponse(
                    output_text="RESET",
                    reasoning_text="restart",
                    usage=NormalizedUsage(total_tokens=6),
                ),
            ]
        )

        model_response, action, retries, messages_sent = agent._request_with_retries(
            [GameAction.RESET]
        )

        assert model_response.output_text == "RESET"
        assert model_response.reasoning_text == "restart"
        assert model_response.usage.total_tokens == 10
        assert action == GameAction.RESET
        assert retries == 1
        assert messages_sent == [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "frame"},
        ]
        assert agent.token_counter == 10

    @pytest.mark.parametrize(
        ("adapter", "request_kwargs"),
        [
            (
                OpenAIChatCompletionsAdapter(_FakeChatClient(_chat_response())),
                {"model": "gpt-5.4"},
            ),
            (
                OpenAIResponsesAdapter(_FakeResponsesClient(_responses_response())),
                {"model": "gpt-5.4"},
            ),
        ],
    )
    def test_chat_and_responses_adapters_yield_same_parsed_action(
        self,
        adapter,
        request_kwargs,
    ):
        agent = _agent_for_request_kwargs(request_kwargs)
        agent.MAX_RETRIES = 0
        agent.conversation = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "frame"},
        ]
        agent._adapter = adapter

        model_response, action, retries, messages_sent = agent._request_with_retries(
            [GameAction.RESET]
        )

        assert model_response.output_text == "RESET"
        assert model_response.reasoning_text == "restart"
        assert action == GameAction.RESET
        assert retries == 0
        assert messages_sent == [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "frame"},
        ]


@pytest.mark.unit
class TestBenchmarkingAgentConversationStorage:
    def test_choose_action_appends_plain_assistant_output_in_normal_mode(self):
        agent = _agent_for_choose_action(
            analysis_mode=False,
            responses=[
                ModelResponse(
                    output_text="ACTION1",
                    reasoning_text="choose the first action",
                    usage=NormalizedUsage(total_tokens=9),
                )
            ],
        )

        action = agent.choose_action([], _playable_frame())

        assert action == GameAction.ACTION1
        assert agent.conversation[-1] == {
            "role": "assistant",
            "content": "ACTION1",
        }
        assert agent._saved_steps[0].messages_sent == [
            {"role": "system", "content": agent._build_system_prompt()},
            {"role": "user", "content": agent.build_frame_content(_playable_frame())},
        ]
        assert agent._saved_steps[0].assistant_response == "ACTION1"
        assert agent._saved_steps[0].reasoning == "choose the first action"

    def test_choose_action_appends_replay_xml_assistant_content_in_analysis_mode(self):
        agent = _agent_for_choose_action(
            analysis_mode=True,
            responses=[
                ModelResponse(
                    output_text="ACTION1",
                    reasoning_text="choose the first action",
                    usage=NormalizedUsage(total_tokens=9),
                )
            ],
        )

        action = agent.choose_action([], _playable_frame())

        assert action == GameAction.ACTION1
        assert agent.conversation[-1] == {
            "role": "assistant",
            "content": (
                "<reasoning_summary>\n"
                "choose the first action\n"
                "</reasoning_summary>\n\n"
                "ACTION1\n"
            ),
        }
        assert agent._saved_steps[0].messages_sent == [
            {"role": "system", "content": agent._build_system_prompt()},
            {"role": "user", "content": agent.build_frame_content(_playable_frame())},
        ]
        assert agent._saved_steps[0].assistant_response == "ACTION1"
        assert agent._saved_steps[0].reasoning == "choose the first action"

    def test_choose_action_appends_plain_output_without_reasoning_in_analysis_mode(self):
        agent = _agent_for_choose_action(
            analysis_mode=True,
            responses=[
                ModelResponse(
                    output_text="ACTION1",
                    reasoning_text=None,
                    usage=NormalizedUsage(total_tokens=9),
                )
            ],
        )

        action = agent.choose_action([], _playable_frame())

        assert action == GameAction.ACTION1
        assert agent.conversation[-1] == {
            "role": "assistant",
            "content": "ACTION1",
        }

    def test_choose_action_parses_current_raw_output_not_rendered_replay_content(self):
        agent = _agent_for_choose_action(
            analysis_mode=True,
            responses=[
                ModelResponse(
                    output_text="ACTION1",
                    reasoning_text="ACTION2 would fail here",
                    usage=NormalizedUsage(total_tokens=9),
                )
            ],
        )
        parsed_inputs = []

        def fake_parse_action(text, available_actions):
            parsed_inputs.append(text)
            return GameAction.ACTION1

        agent._parse_action = fake_parse_action

        action = agent.choose_action([], _playable_frame())

        assert action == GameAction.ACTION1
        assert parsed_inputs == ["ACTION1"]

    def test_choose_action_records_prior_replay_content_in_messages_sent_not_current_reply(
        self,
    ):
        agent = _agent_for_choose_action(
            analysis_mode=True,
            responses=[
                ModelResponse(
                    output_text="ACTION1",
                    reasoning_text="current turn reasoning",
                    usage=NormalizedUsage(total_tokens=9),
                )
            ],
        )
        prior_replay_content = (
            "<reasoning_summary>\n"
            "prior turn reasoning\n"
            "</reasoning_summary>\n\n"
            "ACTION1\n"
        )
        agent.conversation = [
            {"role": "system", "content": agent._build_system_prompt()},
            {"role": "user", "content": "prior frame"},
            {"role": "assistant", "content": prior_replay_content},
        ]

        action = agent.choose_action([], _playable_frame())

        assert action == GameAction.ACTION1
        assert agent._saved_steps[0].messages_sent == [
            {"role": "system", "content": agent._build_system_prompt()},
            {"role": "user", "content": "prior frame"},
            {"role": "assistant", "content": prior_replay_content},
            {"role": "user", "content": agent.build_frame_content(_playable_frame())},
        ]
        assert agent._saved_steps[0].assistant_response == "ACTION1"
        assert agent._saved_steps[0].reasoning == "current turn reasoning"
        assert agent._saved_steps[0].model_dump()["messages_sent"][2]["content"] == (
            prior_replay_content
        )

    def test_resolve_action_records_game_over_frame_before_forced_reset_without_model_call(
        self,
    ):
        agent = _agent_for_choose_action(
            analysis_mode=True,
            responses=[
                ModelResponse(
                    output_text="ACTION1",
                    reasoning_text="should not be used",
                    usage=NormalizedUsage(total_tokens=9),
                )
            ],
        )
        agent.action_counter = 1

        action = agent._resolve_action([], _terminal_frame(GameState.GAME_OVER))

        assert action == GameAction.RESET
        assert agent._adapter.requests == []
        assert agent._saved_steps[0].parsed_action == "RESET"
        assert agent._saved_steps[0].messages_sent == [
            {
                "role": "user",
                "content": agent.build_frame_content(
                    _terminal_frame(GameState.GAME_OVER)
                ),
            }
        ]
        assert agent.conversation == agent._saved_steps[0].messages_sent

    def test_resolve_action_for_not_played_forces_reset_without_appending_transcript_frame(
        self,
    ):
        agent = _agent_for_choose_action(
            analysis_mode=True,
            responses=[
                ModelResponse(
                    output_text="ACTION1",
                    reasoning_text="should not be used",
                    usage=NormalizedUsage(total_tokens=9),
                )
            ],
        )

        action = agent._resolve_action([], _terminal_frame(GameState.NOT_PLAYED))

        assert action == GameAction.RESET
        assert agent._adapter.requests == []
        assert agent.conversation == []
        assert agent._saved_steps[0].parsed_action == "RESET"
        assert agent._saved_steps[0].messages_sent == []


@pytest.mark.unit
class TestBenchmarkingAgentContextTrimming:
    def test_estimate_conversation_tokens_counts_plain_transcript_text(self):
        agent = _agent_for_request_kwargs({"model": "gpt-5.4"})
        agent.conversation = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "frame"},
            {"role": "assistant", "content": "ACTION1"},
        ]

        assert agent._estimate_conversation_tokens() == len("systemframeACTION1")

    def test_estimate_conversation_tokens_counts_analysis_replay_xml(self):
        agent = _agent_for_request_kwargs({"model": "gpt-5.4"})
        agent.analysis_mode = True
        replay_content = agent._build_assistant_turn_content(
            "ACTION1",
            "reason through the old turn",
        )
        agent.conversation = [
            {"role": "system", "content": agent._build_system_prompt()},
            {"role": "user", "content": "frame"},
            {"role": "assistant", "content": replay_content},
        ]

        assert agent._estimate_conversation_tokens() == sum(
            len(message["content"]) for message in agent.conversation
        )
        assert agent._estimate_conversation_tokens() > len(
            f"{agent._build_system_prompt()}frameACTION1"
        )

    def test_trim_to_fit_context_removes_oldest_replay_turn_and_preserves_current_user(self):
        agent = _agent_for_request_kwargs({"model": "gpt-5.4"})
        agent.analysis_mode = True
        old_replay_content = agent._build_assistant_turn_content(
            "ACTION1",
            "reasoning that makes the old assistant turn much longer",
        )
        agent.conversation = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "old-user-turn"},
            {"role": "assistant", "content": old_replay_content},
            {"role": "user", "content": "current-user-turn"},
        ]
        agent.MAX_CONTEXT_LENGTH = len("systemcurrent-user-turn")

        agent._trim_to_fit_context()

        assert agent.conversation == [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "current-user-turn"},
        ]
        assert agent._estimate_conversation_tokens() <= agent.MAX_CONTEXT_LENGTH

    def test_trim_oldest_turn_removes_replay_xml_with_its_assistant_message(self):
        agent = _agent_for_request_kwargs({"model": "gpt-5.4"})
        replay_content = (
            "<reasoning_summary>\n"
            "prior reasoning\n"
            "</reasoning_summary>\n\n"
            "ACTION1\n"
        )
        agent.conversation = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "old-user-turn"},
            {"role": "assistant", "content": replay_content},
            {"role": "user", "content": "current-user-turn"},
        ]

        assert agent._trim_oldest_turn() is True

        assert agent.conversation == [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "current-user-turn"},
        ]
        assert replay_content not in [
            message["content"] for message in agent.conversation
        ]


@pytest.mark.unit
class TestBenchmarkingAgentAnalysisReplayIntegration:
    def test_analysis_mode_replays_turn_one_reasoning_into_turn_two_messages_sent(self):
        agent = _agent_for_choose_action(
            analysis_mode=True,
            responses=[
                ModelResponse(
                    output_text="ACTION1",
                    reasoning_text="inspect the first frame",
                    usage=NormalizedUsage(total_tokens=9),
                ),
                ModelResponse(
                    output_text="ACTION1",
                    reasoning_text="continue the same plan",
                    usage=NormalizedUsage(total_tokens=11),
                ),
            ],
        )

        first_frame = _playable_frame()
        second_frame = _playable_frame()

        first_action = agent.choose_action([], first_frame)
        second_action = agent.choose_action([first_frame], second_frame)

        expected_replay_content = (
            "<reasoning_summary>\n"
            "inspect the first frame\n"
            "</reasoning_summary>\n\n"
            "ACTION1\n"
        )
        assert first_action == GameAction.ACTION1
        assert second_action == GameAction.ACTION1
        assert agent.conversation[2] == {
            "role": "assistant",
            "content": expected_replay_content,
        }
        assert [
            message.model_dump() for message in agent._adapter.requests[1].messages
        ] == [
            {"role": "system", "content": agent._build_system_prompt()},
            {"role": "user", "content": agent.build_frame_content(first_frame)},
            {"role": "assistant", "content": expected_replay_content},
            {"role": "user", "content": agent.build_frame_content(second_frame)},
        ]
        assert agent._saved_steps[1].messages_sent == [
            {"role": "system", "content": agent._build_system_prompt()},
            {"role": "user", "content": agent.build_frame_content(first_frame)},
            {"role": "assistant", "content": expected_replay_content},
            {"role": "user", "content": agent.build_frame_content(second_frame)},
        ]

    def test_normal_mode_sends_plain_turn_one_assistant_content_into_turn_two(self):
        agent = _agent_for_choose_action(
            analysis_mode=False,
            responses=[
                ModelResponse(
                    output_text="ACTION1",
                    reasoning_text="inspect the first frame",
                    usage=NormalizedUsage(total_tokens=9),
                ),
                ModelResponse(
                    output_text="ACTION1",
                    reasoning_text="continue the same plan",
                    usage=NormalizedUsage(total_tokens=11),
                ),
            ],
        )

        first_frame = _playable_frame()
        second_frame = _playable_frame()

        agent.choose_action([], first_frame)
        agent.choose_action([first_frame], second_frame)

        assert [
            message.model_dump() for message in agent._adapter.requests[1].messages
        ] == [
            {"role": "system", "content": agent._build_system_prompt()},
            {"role": "user", "content": agent.build_frame_content(first_frame)},
            {"role": "assistant", "content": "ACTION1"},
            {"role": "user", "content": agent.build_frame_content(second_frame)},
        ]
