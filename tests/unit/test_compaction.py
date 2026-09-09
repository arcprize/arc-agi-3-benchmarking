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
    request_config_with_output_limit,
)
from benchmarking.google_runtime import GoogleContinuousConversationRuntimeAdapter
from benchmarking.recording import RunRecord
from benchmarking.runtime_models import Message, ModelResponse, NormalizedUsage
from benchmarking.runtime_registry import ADAPTER_DESCRIPTORS
from benchmarking.runtime_state import RuntimeState


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
                summary_max_output_tokens=8_192,
            )
        )

        result = compactor.compact(
            adapter=adapter,
            state=state,
            request_config=_request_config(),
            trigger_tokens=175_100,
            max_context_length=175_000,
            max_retries=2,
        )

        assert result.summary == "Continue here."
        assert result.history_items_before == 3
        assert result.history_items_after == 1
        assert result.opaque_continuity_preserved is False
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
        compactor = SummaryCompactor(
            SummaryCompactionPolicy(strategy="harness_summary")
        )

        result = compactor.compact(
            adapter=adapter,
            state=state,
            request_config=_request_config(),
            trigger_tokens=200,
            max_context_length=100,
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
        compactor = SummaryCompactor(
            SummaryCompactionPolicy(strategy="harness_summary")
        )

        first = compactor.compact(
            adapter=adapter,
            state=adapter.initial_state(),
            request_config=_request_config(),
            trigger_tokens=200,
            max_context_length=100,
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
            max_context_length=100,
            max_retries=1,
        )

        assert second.history_items_before == 2
        assert second.history_items_after == 1
        assert "second summary" in second.state.model_dump_json()
        assert "first summary" not in second.state.model_dump_json()

    def test_fails_closed_after_repeated_blank_summaries(self):
        adapter, _ = _google_adapter([_summary_response(""), _summary_response(" ")])
        compactor = SummaryCompactor(
            SummaryCompactionPolicy(strategy="harness_summary")
        )

        with pytest.raises(RuntimeError, match="failed to produce non-empty text"):
            compactor.compact(
                adapter=adapter,
                state=adapter.initial_state(),
                request_config=_request_config(),
                trigger_tokens=200,
                max_context_length=100,
                max_retries=1,
            )

    def test_overflow_fails_immediately_without_replacing_accepted_state(self):
        adapter, low_level = _google_adapter([RuntimeError("context overflow")])
        state = adapter.buffer_inputs(
            adapter.initial_state(),
            [Message(role="user", content="accepted history")],
        )
        compactor = SummaryCompactor(
            SummaryCompactionPolicy(strategy="harness_summary")
        )

        with pytest.raises(RuntimeError, match="context overflow"):
            compactor.compact(
                adapter=adapter,
                state=state,
                request_config=_request_config(),
                trigger_tokens=200,
                max_context_length=100,
                max_retries=3,
            )

        assert len(low_level.requests) == 1
        assert "accepted history" in state.model_dump_json()

    def test_trigger_uses_reported_total_tokens(self):
        assert SummaryCompactor.should_compact(
            NormalizedUsage(total_tokens=175_000), 175_000
        )
        assert not SummaryCompactor.should_compact(
            NormalizedUsage(total_tokens=174_999), 175_000
        )

    def test_output_limit_override_does_not_mutate_request(self):
        request = _request_config()

        updated = request_config_with_output_limit(request, 8_192)

        assert updated["generation_config"]["max_output_tokens"] == 8_192
        assert request["generation_config"]["max_output_tokens"] == 65_536

    def test_prompts_are_domain_neutral(self):
        prompt = f"{SUMMARY_SYSTEM_PROMPT}\n{SUMMARY_REQUEST_PROMPT}\n{SUMMARY_BRIDGE_TEMPLATE}".lower()

        for excluded in ("arc", "game", "level", "frame", "coordinate"):
            assert excluded not in prompt


@pytest.mark.unit
def test_agent_persists_compaction_usage_and_cost(tmp_path):
    adapter, _ = _google_adapter([_summary_response("retained state", tokens=140)])
    agent = BenchmarkingAgent.__new__(BenchmarkingAgent)
    agent._stateful_adapter = adapter
    agent._runtime_state = RuntimeState(
        adapter_id="google.interactions.v1",
        strategy="continuous_conversation",
        payload={"steps": [{"type": "thought", "signature": "opaque"}]},
    )
    agent._summary_compactor = SummaryCompactor(
        SummaryCompactionPolicy(strategy="harness_summary")
    )
    agent._pending_compaction_trigger_tokens = 175_100
    agent._request_kwargs = _request_config()
    agent.MAX_CONTEXT_LENGTH = 175_000
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
    assert payload["opaque_continuity_preserved"] is False
    assert payload["usage"]["total_tokens"] == 140
    assert payload["usage"]["cost"] > 0
    assert run_payload["total_usage"]["total_tokens"] == 140
    assert run_payload["runtime"]["compaction_count"] == 1
    assert agent._pending_compaction_trigger_tokens is None
    assert "opaque" not in agent._runtime_state.model_dump_json()
