import json
from datetime import datetime, timezone

import pytest

from benchmarking.agent import BenchmarkingAgent
from benchmarking.compaction import (
    SUMMARY_BRIDGE_TEMPLATE,
    SUMMARY_REQUEST_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    SummaryCompactionPolicy,
    SummaryCompactor,
    build_summary_prompt_record,
    request_config_with_output_limit,
)
from benchmarking.exceptions import ContextOverflowError
from benchmarking.google_runtime import GoogleContinuousConversationRuntimeAdapter
from benchmarking.recording import RunRecord
from benchmarking.runtime_models import Message, ModelResponse, NormalizedUsage
from benchmarking.runtime_registry import ADAPTER_DESCRIPTORS
from benchmarking.runtime_state import (
    AcceptedTurn,
    ModelTurnRequest,
    RuntimeState,
    sanitize_settings,
)


class _FakeModelAdapter:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def invoke(self, request):
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _summary_response(text, *, tokens=30):
    return ModelResponse(
        output_text=text,
        usage=NormalizedUsage(
            input_tokens=tokens - 10,
            output_tokens=10,
            total_tokens=tokens,
        ),
        raw_response={
            "steps": [
                {
                    "type": "model_output",
                    "content": [{"type": "text", "text": text}],
                }
            ]
        },
    )


def _accepted_response(text, signature):
    return ModelResponse(
        output_text=text,
        reasoning_text=f"reasoning for {text}",
        usage=NormalizedUsage(total_tokens=30),
        raw_response={
            "steps": [
                {
                    "type": "thought",
                    "signature": signature,
                    "summary": [
                        {"type": "text", "text": f"reasoning for {text}"}
                    ],
                },
                {
                    "type": "model_output",
                    "content": [{"type": "text", "text": text}],
                },
            ]
        },
    )


def _google_adapter(responses):
    low_level = _FakeModelAdapter(responses)
    adapter = GoogleContinuousConversationRuntimeAdapter(
        model_adapter=low_level,
        descriptor=ADAPTER_DESCRIPTORS["google.interactions.v1"],
    )
    return adapter, low_level


def _request_config():
    return {
        "model": "gemini-3.8-flash",
        "store": False,
        "generation_config": {
            "max_output_tokens": 65_536,
            "thinking_level": "low",
            "thinking_summaries": "auto",
        },
    }


def _policy(*, trigger_tokens=175_000):
    return SummaryCompactionPolicy(
        strategy="harness_summary",
        trigger_tokens=trigger_tokens,
    )


@pytest.mark.unit
class TestSummaryCompactor:
    def test_replaces_native_history_with_one_summary_bridge(self):
        adapter, low_level = _google_adapter([_summary_response("Continue here.")])
        state = RuntimeState(
            adapter_id="google.interactions.v1",
            strategy="continuous_conversation",
            payload={
                "steps": [
                    {"type": "user_input", "content": "old"},
                    {"type": "thought", "signature": "opaque"},
                    {"type": "model_output", "content": "result"},
                ]
            },
        )
        compactor = SummaryCompactor(
            SummaryCompactionPolicy(
                strategy="harness_summary",
                trigger_tokens=175_000,
                summary_max_output_tokens=8_192,
            )
        )

        result = compactor.compact(
            adapter=adapter,
            state=state,
            request_config=_request_config(),
            trigger_tokens=175_100,
            max_context_length=1_048_576,
            max_retries=2,
        )

        assert result.summary == "Continue here."
        assert result.prompt == build_summary_prompt_record(state)
        assert result.prompt["system_prompt"] == SUMMARY_SYSTEM_PROMPT
        assert result.prompt["user_prompt"] == SUMMARY_REQUEST_PROMPT
        assert result.prompt["context"] == {
            "steps": [
                {"type": "user_input", "content": "old"},
                {"type": "thought"},
                {"type": "model_output", "content": "result"},
            ]
        }
        assert result.history_items_to_compact == 3
        assert "opaque" not in result.state.model_dump_json()
        assert result.state.payload["steps"] == [
            {
                "type": "user_input",
                "content": [
                    {
                        "type": "text",
                        "text": SUMMARY_BRIDGE_TEMPLATE.format(
                            summary="Continue here."
                        ),
                    }
                ],
            }
        ]
        summary_request = low_level.requests[0]
        assert summary_request.messages[0].content == SUMMARY_SYSTEM_PROMPT
        assert summary_request.messages[1].content == SUMMARY_REQUEST_PROMPT
        assert (
            sanitize_settings(summary_request.native_input[:-1])
            == result.prompt["context"]["steps"]
        )
        assert summary_request.native_input[-1] == {
            "type": "user_input",
            "content": [{"type": "text", "text": SUMMARY_REQUEST_PROMPT}],
        }
        assert (
            summary_request.request_config["generation_config"][
                "max_output_tokens"
            ]
            == 8_192
        )

    def test_retries_blank_summary_without_advancing_state(self):
        adapter, low_level = _google_adapter(
            [_summary_response(" ", tokens=20), _summary_response("usable", tokens=30)]
        )
        state = adapter.initial_state()
        compactor = SummaryCompactor(_policy(trigger_tokens=100))

        result = compactor.compact(
            adapter=adapter,
            state=state,
            request_config=_request_config(),
            trigger_tokens=200,
            max_context_length=100_000,
            max_retries=1,
        )

        assert len(low_level.requests) == 2
        assert low_level.requests[0].native_input == low_level.requests[1].native_input
        assert result.attempts == 2
        assert result.usage.total_tokens == 50

    def test_supports_repeated_compaction_cycles(self):
        adapter, _ = _google_adapter(
            [_summary_response("first summary"), _summary_response("second summary")]
        )
        compactor = SummaryCompactor(_policy(trigger_tokens=100))

        first = compactor.compact(
            adapter=adapter,
            state=adapter.initial_state(),
            request_config=_request_config(),
            trigger_tokens=200,
            max_context_length=100_000,
            max_retries=1,
        )
        state_after_more_history = adapter.buffer_inputs(
            first.state,
            [Message(role="user", content="newer input")],
        )
        second = compactor.compact(
            adapter=adapter,
            state=state_after_more_history,
            request_config=_request_config(),
            trigger_tokens=220,
            max_context_length=100_000,
            max_retries=1,
        )

        assert second.history_items_to_compact == 2
        assert "second summary" in second.state.model_dump_json()
        assert "first summary" not in second.state.model_dump_json()

    def test_fails_closed_after_repeated_blank_summaries(self):
        adapter, _ = _google_adapter([_summary_response(""), _summary_response(" ")])
        compactor = SummaryCompactor(_policy(trigger_tokens=100))

        with pytest.raises(RuntimeError, match="failed to produce non-empty text"):
            compactor.compact(
                adapter=adapter,
                state=adapter.initial_state(),
                request_config=_request_config(),
                trigger_tokens=200,
                max_context_length=100_000,
                max_retries=1,
            )

    def test_overflow_preserves_exact_native_tail_after_summary(self):
        adapter, low_level = _google_adapter(
            [ContextOverflowError("context overflow"), _summary_response("prefix")]
        )
        state = RuntimeState(
            adapter_id="google.interactions.v1",
            strategy="continuous_conversation",
            payload={
                "steps": [
                    {"type": "user_input", "content": "first"},
                    {"type": "thought", "signature": "opaque-first"},
                    {"type": "model_output", "content": "ACTION1"},
                    {"type": "user_input", "content": "recent"},
                    {"type": "thought", "signature": "opaque-recent"},
                    {"type": "model_output", "content": "ACTION2"},
                ]
            },
            accepted_turns=[
                AcceptedTurn(
                    start_item=0,
                    end_item=3,
                    messages=[
                        Message(role="user", content="first"),
                        Message(role="assistant", content="ACTION1"),
                    ],
                    reasoning_summary="first reasoning",
                ),
                AcceptedTurn(
                    start_item=3,
                    end_item=6,
                    messages=[
                        Message(role="user", content="recent"),
                        Message(role="assistant", content="ACTION2"),
                    ],
                    reasoning_summary="recent reasoning",
                ),
            ],
        )
        original_state = state.model_copy(deep=True)
        compactor = SummaryCompactor(_policy(trigger_tokens=100))

        result = compactor.compact(
            adapter=adapter,
            state=state,
            request_config=_request_config(),
            trigger_tokens=200,
            max_context_length=100_000,
            max_retries=1,
        )

        assert len(low_level.requests) == 2
        assert len(low_level.requests[0].native_input) == 7
        assert len(low_level.requests[1].native_input) == 4
        assert result.attempts == 2
        assert result.overflow_recoveries == 1
        assert result.excluded_turns == 1
        assert result.excluded_history_items == 3
        assert "first" in json.dumps(result.prompt)
        assert "recent" not in json.dumps(result.prompt)
        assert "signature" not in json.dumps(result.prompt)
        assert "opaque" not in json.dumps(result.prompt)
        bridge = result.state.payload["steps"][0]["content"][0]["text"]
        assert "prefix" in bridge
        assert "recent" not in bridge
        assert result.state.payload["steps"][1:] == state.payload["steps"][3:]
        assert result.state.payload["steps"][2]["signature"] == "opaque-recent"
        assert [
            (turn.start_item, turn.end_item)
            for turn in result.state.accepted_turns
        ] == [(1, 4)]
        assert result.state.accepted_turns[0].messages == state.accepted_turns[1].messages
        assert (
            result.state.accepted_turns[0].reasoning_summary
            == "recent reasoning"
        )
        assert "opaque-first" not in result.state.model_dump_json()
        assert state == original_state

    def test_overflow_at_protected_boundary_fails_without_mutating_state(self):
        adapter, low_level = _google_adapter(
            [ContextOverflowError("context overflow")]
        )
        state = adapter.buffer_inputs(
            adapter.initial_state(),
            [Message(role="user", content="protected continuation")],
        )
        compactor = SummaryCompactor(_policy(trigger_tokens=100))

        with pytest.raises(ContextOverflowError, match="protected context boundary"):
            compactor.compact(
                adapter=adapter,
                state=state,
                request_config=_request_config(),
                trigger_tokens=200,
                max_context_length=100_000,
                max_retries=1,
            )

        assert len(low_level.requests) == 1
        assert "protected continuation" in state.model_dump_json()

    def test_repeated_overflow_preserves_excluded_turn_order(self):
        adapter, _ = _google_adapter(
            [
                _accepted_response("ACTION1", "opaque-first"),
                _accepted_response("ACTION2", "opaque-second"),
                ContextOverflowError("first overflow"),
                ContextOverflowError("second overflow"),
                _summary_response("empty-prefix summary"),
            ]
        )
        first = adapter.invoke_turn(
            ModelTurnRequest(
                system_prompt="system",
                new_messages=[Message(role="user", content="first input")],
                request_config=_request_config(),
                previous_state=adapter.initial_state(),
            )
        )
        second = adapter.invoke_turn(
            ModelTurnRequest(
                system_prompt="system",
                new_messages=[Message(role="user", content="second input")],
                request_config=_request_config(),
                previous_state=first.state,
            )
        )
        accepted_state = second.state.model_copy(deep=True)
        compactor = SummaryCompactor(_policy(trigger_tokens=100))

        result = compactor.compact(
            adapter=adapter,
            state=second.state,
            request_config=_request_config(),
            trigger_tokens=200,
            max_context_length=100_000,
            max_retries=1,
        )

        assert result.state.payload["steps"][1:] == accepted_state.payload["steps"]
        assert [
            step.get("signature")
            for step in result.state.payload["steps"]
            if step["type"] == "thought"
        ] == ["opaque-first", "opaque-second"]
        assert [
            (turn.start_item, turn.end_item)
            for turn in result.state.accepted_turns
        ] == [(1, 4), (4, 7)]
        assert result.overflow_recoveries == 2
        assert result.excluded_turns == 2
        assert result.attempts == 3
        assert second.state == accepted_state

    def test_non_overflow_error_does_not_remove_history(self):
        adapter, _ = _google_adapter([RuntimeError("service unavailable")])
        state = adapter.initial_state()
        compactor = SummaryCompactor(_policy(trigger_tokens=100))

        with pytest.raises(RuntimeError, match="service unavailable"):
            compactor.compact(
                adapter=adapter,
                state=state,
                request_config=_request_config(),
                trigger_tokens=200,
                max_context_length=100_000,
                max_retries=1,
            )

    def test_oversized_continuation_state_fails_closed(self):
        adapter, _ = _google_adapter([_summary_response("large summary")])
        state = adapter.initial_state()
        compactor = SummaryCompactor(_policy(trigger_tokens=10))

        with pytest.raises(ContextOverflowError, match="continuation state"):
            compactor.compact(
                adapter=adapter,
                state=state,
                request_config=_request_config(),
                trigger_tokens=20,
                max_context_length=10,
                max_retries=1,
            )

    def test_continuation_size_check_includes_retained_native_turns(self):
        adapter, _ = _google_adapter(
            [ContextOverflowError("context overflow"), _summary_response("small")]
        )
        state = RuntimeState(
            adapter_id="google.interactions.v1",
            strategy="continuous_conversation",
            payload={
                "steps": [
                    {"type": "user_input", "content": "recent"},
                    {
                        "type": "thought",
                        "signature": "opaque-" + ("x" * 1_000),
                    },
                    {"type": "model_output", "content": "result"},
                ]
            },
            accepted_turns=[
                AcceptedTurn(
                    start_item=0,
                    end_item=3,
                    messages=[
                        Message(role="user", content="recent"),
                        Message(role="assistant", content="result"),
                    ],
                )
            ],
        )
        compactor = SummaryCompactor(_policy(trigger_tokens=10))

        with pytest.raises(ContextOverflowError, match="continuation state"):
            compactor.compact(
                adapter=adapter,
                state=state,
                request_config=_request_config(),
                trigger_tokens=20,
                max_context_length=800,
                max_retries=1,
            )

    def test_trigger_uses_reported_total_tokens(self):
        compactor = SummaryCompactor(_policy())

        assert compactor.should_compact(NormalizedUsage(total_tokens=175_000))
        assert not compactor.should_compact(
            NormalizedUsage(total_tokens=174_999)
        )

    def test_output_limit_override_does_not_mutate_request(self):
        request = _request_config()

        updated = request_config_with_output_limit(request, 8_192)

        assert updated["generation_config"]["max_output_tokens"] == 8_192
        assert request["generation_config"]["max_output_tokens"] == 65_536

    def test_prompts_are_domain_neutral(self):
        prompt = (
            f"{SUMMARY_SYSTEM_PROMPT}\n{SUMMARY_REQUEST_PROMPT}\n"
            f"{SUMMARY_BRIDGE_TEMPLATE}"
        ).lower()

        for excluded in ("arc", "game", "level", "frame", "coordinate"):
            assert excluded not in prompt

        assert SUMMARY_REQUEST_PROMPT == (
            "Summarize the conversation so you can continue making progress on "
            "the task in a future context. The conversation history will be "
            "replaced by this summary and will no longer be available.\n"
            "Use your judgment about what matters for this task."
        )


@pytest.mark.unit
def test_agent_persists_compaction_usage_for_next_action_attribution(tmp_path):
    adapter, _ = _google_adapter([_summary_response("retained state", tokens=140)])
    agent = BenchmarkingAgent.__new__(BenchmarkingAgent)
    agent._stateful_adapter = adapter
    agent._runtime_state = RuntimeState(
        adapter_id="google.interactions.v1",
        strategy="continuous_conversation",
        payload={"steps": [{"type": "thought", "signature": "opaque"}]},
    )
    agent._summary_compactor = SummaryCompactor(
        _policy()
    )
    agent._pending_compaction_trigger_tokens = 175_100
    agent._request_kwargs = _request_config()
    agent.MAX_CONTEXT_LENGTH = 1_048_576
    agent.MAX_RETRIES = 1
    agent.MODEL = "gemini-3.8-flash"
    agent._pricing = {"input": 0.75, "output": 3.75}
    agent._compaction_counter = 0
    agent.step_counter = 2
    agent.token_counter = 0
    agent.conversation = []
    agent.run_dir = str(tmp_path)
    agent.run_record = RunRecord(
        run_id="run",
        game_id="id",
        agent_name="agent",
        model=agent.MODEL,
        started_at=datetime.now(timezone.utc),
        run_dir=str(tmp_path),
        runtime={"compaction_count": 0},
    )

    agent._run_pending_compaction()

    payload = json.loads((tmp_path / "compaction_001.json").read_text())
    run_payload = json.loads((tmp_path / "run_meta.json").read_text())
    assert payload["before_step"] == 3
    assert payload["mechanism"] == "harness_summary"
    assert payload["prompt"] == {
        "system_prompt": SUMMARY_SYSTEM_PROMPT,
        "context": {"steps": [{"type": "thought"}]},
        "user_prompt": SUMMARY_REQUEST_PROMPT,
    }
    assert payload["overflow_recoveries"] == 0
    assert payload["excluded_turns"] == 0
    assert payload["history_items_to_compact"] == 1
    assert "history_items_before" not in payload
    assert "history_items_after" not in payload
    assert payload["usage"]["total_tokens"] == 140
    assert payload["usage"]["cost"] == 0
    assert payload["usage"]["cost_details"] == {}
    assert "estimated_cost" not in payload
    assert run_payload["total_usage"]["total_tokens"] == 140
    assert run_payload["total_usage"]["cost"] == 0
    assert "estimated_cost" not in run_payload
    assert run_payload["runtime"]["compaction_count"] == 1
    assert agent._pending_compaction_trigger_tokens is None
    assert agent._pending_compaction_usage.total_tokens == 140
    assert agent._pending_compaction_continuation.model_dump() == {
        "compaction": 1,
        "summary": "retained state",
        "bridge": SUMMARY_BRIDGE_TEMPLATE.format(summary="retained state"),
    }
    assert "opaque" not in agent._runtime_state.model_dump_json()
    assert "opaque" not in agent._pending_compaction_continuation.model_dump_json()
