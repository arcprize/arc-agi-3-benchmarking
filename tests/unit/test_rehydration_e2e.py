"""Golden equivalence tests for rehydration (REHYDRATION_PLAN.md §8).

For each adapter and scenario:
  A: an uninterrupted run of TOTAL steps.
  B: the same run stopped after RESUME_AT steps (a crash at a loop boundary).
  R: a fresh agent rehydrated from B's snapshot and recording through the full
     offline path (prepare_rehydration), continuing to TOTAL steps.
R must send the same model calls as A after the resume point, take the same
game actions with the same reasoning metadata, and write identical step files,
compaction files, usage totals, and final snapshot content.
"""

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest
from arcengine import GameState

from benchmarking.agent import BenchmarkingAgent
from benchmarking.base import ExitReason
from benchmarking.rehydration import RehydrationArgs, prepare_rehydration
from benchmarking.runtime_models import ModelRequest, ModelResponse, NormalizedUsage
from tests.unit.test_agent_snapshot import (
    _build_agent,
    _comparable,
    _latest_snapshot,
    _ScriptedEnv,
)

RESUME_AT = 3
TOTAL = 5
TRIGGER = 200_000  # above every config's 175k compaction trigger


def _usage(big: bool = False) -> NormalizedUsage:
    if big:
        return NormalizedUsage(
            input_tokens=TRIGGER - 5, output_tokens=5, total_tokens=TRIGGER
        )
    return NormalizedUsage(input_tokens=10, output_tokens=5, total_tokens=15)


class _ScriptedModel:
    """Model adapter fake that returns a fixed response sequence."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple] = []

    def invoke(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(("invoke", request.model_copy(deep=True)))
        return self.responses.pop(0)

    def compact(self, *, model: str, input_items: list[dict]) -> ModelResponse:
        self.calls.append(("compact", model, deepcopy(input_items)))
        return self.responses.pop(0)


# ── Per-adapter native response shapes ───────────────────────────────────


def _manual_action(label: int, big: bool = False) -> ModelResponse:
    return ModelResponse(
        output_text=f"Step {label}: ACTION1",
        reasoning_text=f"thinking {label}",
        usage=_usage(big),
    )


def _openai_action(label: int, big: bool = False) -> ModelResponse:
    output = [
        {
            "type": "reasoning",
            "id": f"rs_{label}",
            "summary": [{"type": "summary_text", "text": f"thought {label}"}],
            "encrypted_content": f"opaque-{label}",
        },
        {
            "type": "message",
            "id": f"msg_{label}",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "ACTION1"}],
        },
    ]
    if big:
        # Server-side compaction: the compaction item replaces prior history.
        output.insert(
            0,
            {"type": "compaction", "id": f"cmp_{label}", "encrypted_content": "cmp"},
        )
    return ModelResponse(
        output_text="ACTION1", usage=_usage(), raw_response={"output": output}
    )


def _anthropic_raw(blocks: list[dict], stop_reason: str, usage: dict) -> ModelResponse:
    # The Anthropic runtime re-normalizes usage from the raw payload.
    return ModelResponse(
        output_text="",
        usage=NormalizedUsage(),
        raw_response={
            "model": "claude-opus-5",
            "role": "assistant",
            "content": blocks,
            "stop_reason": stop_reason,
            "usage": usage,
        },
    )


def _anthropic_action(label: int, big: bool = False) -> ModelResponse:
    return _anthropic_raw(
        [
            {"type": "thinking", "thinking": f"thought {label}", "signature": f"sig-{label}"},
            {"type": "text", "text": "ACTION1"},
        ],
        "end_turn",
        {"input_tokens": TRIGGER - 5 if big else 10, "output_tokens": 5},
    )


def _anthropic_compaction() -> ModelResponse:
    return _anthropic_raw(
        [{"type": "compaction", "content": "native summary", "signature": "cmp-sig"}],
        "compaction",
        {"input_tokens": 100, "output_tokens": 20},
    )


def _google_steps(text: str, label: int) -> list[dict]:
    return [
        {
            "type": "thought",
            "summary": [{"type": "text", "text": f"thought {label}"}],
            "signature": f"sig-{label}",
        },
        {"type": "model_output", "content": [{"type": "text", "text": text}]},
    ]


def _google_action(label: int, big: bool = False) -> ModelResponse:
    return ModelResponse(
        output_text="ACTION1",
        reasoning_text=f"thought {label}",
        usage=_usage(big),
        raw_response={"steps": _google_steps("ACTION1", label)},
    )


def _google_summary() -> ModelResponse:
    return ModelResponse(
        output_text="harness summary",
        usage=_usage(),
        raw_response={"steps": _google_steps("harness summary", 0)},
    )


def _xai_action(label: int, big: bool = False) -> ModelResponse:
    return ModelResponse(
        output_text="ACTION1",
        reasoning_text=f"thought {label}",
        usage=_usage(big),
        raw_response={
            "output": [
                {
                    "type": "reasoning",
                    "id": f"rs_{label}",
                    "summary": [{"type": "summary_text", "text": f"thought {label}"}],
                    "encrypted_content": f"opaque-{label}",
                },
                {
                    "type": "message",
                    "id": f"msg_{label}",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "ACTION1"}],
                },
            ]
        },
    )


def _xai_compaction() -> ModelResponse:
    return ModelResponse(
        output_text="",
        usage=_usage(),
        raw_response={
            "output": [{"type": "compaction", "id": "cmp", "encrypted_content": "cmp"}]
        },
    )


def _deepseek_response(text: str, tool: str, label: int, big: bool) -> ModelResponse:
    return ModelResponse(
        output_text=text,
        response_status="completed",
        usage=_usage(big),
        raw_response={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": f"thinking {label}",
                        "tool_calls": [
                            {
                                "id": f"call_{label}",
                                "type": "function",
                                "function": {
                                    "name": tool,
                                    "arguments": json.dumps({"text": text}),
                                },
                            }
                        ],
                    }
                }
            ]
        },
    )


def _deepseek_action(label: int, big: bool = False) -> ModelResponse:
    return _deepseek_response("ACTION1", "submit_action", label, big)


def _deepseek_summary() -> ModelResponse:
    return _deepseek_response("harness summary", "return_summary", 0, False)


@dataclass(frozen=True)
class Adapter:
    config_id: str
    action: Callable[..., ModelResponse]
    # Response for the extra model call made when compaction runs at the
    # resume point; None when compaction happens inside the action response.
    compaction: Callable[[], ModelResponse] | None = None
    continuous: bool = True

    def install(self, agent: BenchmarkingAgent, model: _ScriptedModel) -> None:
        if self.continuous:
            agent._stateful_adapter._model_adapter = model
        else:
            agent._adapter = model


ADAPTERS = {
    "manual": Adapter("openai-gpt-5-4-2026-03-05", _manual_action, continuous=False),
    "openai": Adapter("openai-gpt-5-6-sol-max-provider-adapter", _openai_action),
    "anthropic": Adapter(
        "anthropic-opus-5-low-provider-adapter", _anthropic_action, _anthropic_compaction
    ),
    "google": Adapter(
        "google-gemini-3-8-flash-low-provider-adapter", _google_action, _google_summary
    ),
    "xai": Adapter("xai-grok-4-7-low-provider-adapter", _xai_action, _xai_compaction),
    "deepseek": Adapter(
        "deepseek-v4-1-flash-low-provider-adapter", _deepseek_action, _deepseek_summary
    ),
}


# ── Scenarios ────────────────────────────────────────────────────────────


def _script(adapter: Adapter, scenario: str) -> list[ModelResponse]:
    """Model responses for the full uninterrupted run, in call order."""
    if scenario in ("plain", "level_up_before_resume", "level_budget_ends_run"):
        return [adapter.action(step) for step in range(1, TOTAL + 1)]
    if scenario == "compactions_before_and_at_resume":
        # Steps 1 and RESUME_AT report usage above the trigger. The first
        # compaction completes before the boundary (so counters and compacted
        # state carry over); the second is pending across the boundary and
        # runs before step RESUME_AT + 1.
        responses = []
        for step in range(1, TOTAL + 1):
            big = step in (1, RESUME_AT)
            responses.append(adapter.action(step, big=big))
            if big and adapter.compaction is not None:
                responses.append(adapter.compaction())
        return responses
    if scenario == "buffered_reset_at_resume":
        # GAME_OVER at step RESUME_AT - 1 forces a RESET at step RESUME_AT whose
        # observation is buffered, unsent, across the boundary.
        steps = [s for s in range(1, TOTAL + 1) if s != RESUME_AT]
        return [adapter.action(step) for step in steps]
    raise ValueError(scenario)


def _env(scenario: str, guid: str) -> _ScriptedEnv:
    return _ScriptedEnv(
        guid=guid,
        initial_state=GameState.NOT_FINISHED,
        game_over_at=RESUME_AT - 1 if scenario == "buffered_reset_at_resume" else None,
        # Level 1 completes before the boundary, so the resumed run must know
        # it already announced the new level and reset the level counter.
        level_up_at=2 if scenario == "level_up_before_resume" else None,
        # Every config multiplies baselines by 5, so a baseline of 1 gives a
        # per-level budget of exactly TOTAL actions.
        baseline_actions=[1] if scenario == "level_budget_ends_run" else None,
    )


def _run(
    monkeypatch,
    tmp_path,
    adapter: Adapter,
    scenario: str,
    *,
    steps: int,
    guid: str,
    responses: list[ModelResponse],
    rehydration=None,
) -> tuple[BenchmarkingAgent, _ScriptedModel]:
    agent = _build_agent(
        monkeypatch,
        tmp_path,
        adapter.config_id,
        env=_env(scenario, guid),
        rehydration=rehydration,
    )
    model = _ScriptedModel(responses)
    adapter.install(agent, model)
    if scenario == "level_budget_ends_run" and steps == TOTAL:
        # Only the per-level budget may stop the run, so a resumed run with a
        # wrong level counter would keep playing (and run out of responses).
        agent.MAX_ACTIONS = 100
    else:
        agent.MAX_ACTIONS = steps - 1  # main() runs while action_counter <= MAX_ACTIONS
    agent.main()
    assert agent.step_counter == steps
    if scenario == "level_budget_ends_run" and steps == TOTAL:
        assert agent.exit_reason == ExitReason.ACTION_BUDGET
    return agent, model


def _prepare_from_files(tmp_path, source: BenchmarkingAgent, step: int, config_id: str):
    recording = source.arc_env.write_recording(tmp_path / f"{source.guid}.jsonl")
    state = Path(source.run_dir) / "state" / f"state_step_{step:04d}.json"
    return prepare_rehydration(
        RehydrationArgs(recording=recording, state=state),
        config_id=config_id,
        game_ids=["game-id"],
    )


def _records(agent: BenchmarkingAgent, pattern: str) -> dict[str, dict]:
    records = {}
    for path in sorted(Path(agent.run_dir).glob(pattern)):
        data = json.loads(path.read_text())
        data.pop("timestamp")
        data.pop("duration_seconds")
        records[path.name] = data
    return records


def _assert_equivalent(
    uninterrupted: BenchmarkingAgent,
    uninterrupted_model: _ScriptedModel,
    resumed: BenchmarkingAgent,
    resumed_model: _ScriptedModel,
    prefix_calls: int,
) -> None:
    assert resumed_model.calls == uninterrupted_model.calls[prefix_calls:]
    assert resumed_model.responses == [] and uninterrupted_model.responses == []
    assert resumed.arc_env.calls == uninterrupted.arc_env.calls

    resumed_steps = _records(resumed, "step_*.json")
    expected_steps = {
        name: record
        for name, record in _records(uninterrupted, "step_*.json").items()
        if int(name[5:8]) > RESUME_AT
    }
    assert list(resumed_steps) == [
        f"step_{step:03d}.json" for step in range(RESUME_AT + 1, TOTAL + 1)
    ]
    assert resumed_steps == expected_steps
    # Compactions before the boundary belong to the source run's files.
    assert _records(resumed, "compaction_*.json") == {
        name: record
        for name, record in _records(uninterrupted, "compaction_*.json").items()
        if record["before_step"] > RESUME_AT
    }

    assert resumed.run_record.total_usage == uninterrupted.run_record.total_usage
    assert resumed.run_record.total_steps == uninterrupted.run_record.total_steps
    assert resumed.run_record.runtime == uninterrupted.run_record.runtime
    assert _comparable(_latest_snapshot(resumed)) == _comparable(
        _latest_snapshot(uninterrupted)
    )


SCENARIOS = [
    (name, scenario)
    for name in ADAPTERS
    for scenario in (
        "plain",
        "compactions_before_and_at_resume",
        "buffered_reset_at_resume",
        "level_up_before_resume",
        "level_budget_ends_run",
    )
    if not (scenario == "compactions_before_and_at_resume" and name == "manual")
]


@pytest.mark.unit
@pytest.mark.parametrize(("adapter_name", "scenario"), SCENARIOS)
def test_rehydrated_run_matches_uninterrupted_run(
    monkeypatch, tmp_path, adapter_name, scenario
):
    adapter = ADAPTERS[adapter_name]
    uninterrupted, uninterrupted_model = _run(
        monkeypatch, tmp_path, adapter, scenario,
        steps=TOTAL, guid="guid-a", responses=_script(adapter, scenario),
    )
    crashed, crashed_model = _run(
        monkeypatch, tmp_path, adapter, scenario,
        steps=RESUME_AT, guid="guid-b", responses=_script(adapter, scenario),
    )
    prefix_calls = len(crashed_model.calls)
    prepared = _prepare_from_files(tmp_path, crashed, RESUME_AT, adapter.config_id)

    resumed, resumed_model = _run(
        monkeypatch, tmp_path, adapter, scenario,
        steps=TOTAL, guid="guid-r",
        responses=_script(adapter, scenario)[prefix_calls:],
        rehydration=prepared,
    )

    _assert_equivalent(
        uninterrupted, uninterrupted_model, resumed, resumed_model, prefix_calls
    )


@pytest.mark.unit
def test_scenarios_exercise_their_boundary_conditions(monkeypatch, tmp_path):
    """Guard against scenarios silently degenerating into the plain case."""
    google = ADAPTERS["google"]
    crashed, _ = _run(
        monkeypatch, tmp_path, google, "compactions_before_and_at_resume",
        steps=RESUME_AT, guid="guid-b",
        responses=_script(google, "compactions_before_and_at_resume"),
    )
    snapshot = _latest_snapshot(crashed)
    assert snapshot.agent.pending_compaction_trigger_tokens == TRIGGER
    assert snapshot.agent.compaction_counter == 1  # one compaction already done

    anthropic = ADAPTERS["anthropic"]
    crashed, _ = _run(
        monkeypatch, tmp_path, anthropic, "compactions_before_and_at_resume",
        steps=RESUME_AT, guid="guid-b",
        responses=_script(anthropic, "compactions_before_and_at_resume"),
    )
    payload = _latest_snapshot(crashed).runtime_state.payload
    assert payload["context_tokens"] == TRIGGER
    assert payload["messages"][0]["content"][0]["type"] == "compaction"

    deepseek = ADAPTERS["deepseek"]
    crashed, _ = _run(
        monkeypatch, tmp_path, deepseek, "buffered_reset_at_resume",
        steps=RESUME_AT, guid="guid-b",
        responses=_script(deepseek, "buffered_reset_at_resume"),
    )
    snapshot = _latest_snapshot(crashed)
    assert crashed._previous_action.name == "RESET"
    assert len(snapshot.runtime_state.payload["pending_messages"]) == 1


@pytest.mark.unit
@pytest.mark.parametrize("adapter_name", ["manual", "openai"])
def test_steps_recorded_after_last_snapshot_are_truncated(
    monkeypatch, tmp_path, adapter_name
):
    """Crash after action N+1 was submitted but before snapshot N+1 (plan F6)."""
    adapter = ADAPTERS[adapter_name]
    uninterrupted, uninterrupted_model = _run(
        monkeypatch, tmp_path, adapter, "plain",
        steps=TOTAL, guid="guid-a", responses=_script(adapter, "plain"),
    )
    crashed, _ = _run(
        monkeypatch, tmp_path, adapter, "plain",
        steps=RESUME_AT + 1, guid="guid-b", responses=_script(adapter, "plain"),
    )
    (Path(crashed.run_dir) / "state" / f"state_step_{RESUME_AT + 1:04d}.json").unlink()
    prepared = _prepare_from_files(tmp_path, crashed, RESUME_AT, adapter.config_id)
    assert len(crashed.arc_env.events) == RESUME_AT + 1
    assert len(prepared.steps) == RESUME_AT

    resumed, resumed_model = _run(
        monkeypatch, tmp_path, adapter, "plain",
        steps=TOTAL, guid="guid-r",
        responses=_script(adapter, "plain")[RESUME_AT:],
        rehydration=prepared,
    )

    _assert_equivalent(
        uninterrupted, uninterrupted_model, resumed, resumed_model, RESUME_AT
    )
