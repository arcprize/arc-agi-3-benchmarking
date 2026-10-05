import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from arcengine import ActionInput, FrameData, FrameDataRaw, GameAction, GameState

from benchmarking.agent import BenchmarkingAgent
from benchmarking.base import ExitReason
from benchmarking.model_config import get_model_config, list_model_config_ids
from benchmarking.rehydration import (
    AgentSnapshot,
    PreparedRehydration,
    RehydrationError,
    config_sha256,
    frame_fingerprint,
    load_snapshot,
    parse_toolkit_recording,
)
from benchmarking.runtime_models import (
    Message,
    ModelRequest,
    ModelResponse,
    NormalizedUsage,
)

MANUAL_CONFIG = "openai-gpt-5-4-2026-03-05"
OPENAI_CONTINUOUS_CONFIG = "openai-gpt-5-6-sol-max-provider-adapter"
CONTINUOUS_CONFIGS = [
    config_id
    for config_id in list_model_config_ids()
    if get_model_config(config_id)["runtime"].get("state") == "continuous_conversation"
]

# Every BenchmarkingAgent instance attribute must be classified here (plan F2).
# A new attribute that affects future prompts or budgets must be added to the
# snapshot (rehydration.AgentFields) and to SNAPSHOTTED; otherwise rehydrated
# runs silently drift from uninterrupted ones.
SNAPSHOTTED = {
    "conversation",
    "token_counter",
    "step_counter",
    "action_counter",
    "_level_action_counter",
    "_last_levels_completed",
    "_level_just_advanced",
    "_compaction_counter",
    "_pending_compaction_trigger_tokens",
    "_previous_action",  # rebuilt by replay
    "_runtime_state",
    "timer",  # as elapsed_seconds
    "run_record",  # total_usage; identity fields are per-session
    "frames",  # rebuilt by replay; last frame is fingerprinted
    "guid",  # source guid; the new session gets its own
    "_lineage",
    "_elapsed_offset_seconds",  # from elapsed_seconds
}
# Must be empty at a loop boundary; _snapshot() refuses otherwise.
IN_FLIGHT = {
    "_pending_turn_messages",
    "_pending_action_reasoning",
    "_pending_compaction_usage",
    "_pending_compaction_continuation",
}
# Rebuilt from the model config, which rehydration validates by hash.
FROM_CONFIG = {
    "MAX_ACTIONS",  # derived from baseline_actions x multiplier when available
    "MAX_ACTIONS_BASELINE_MULTIPLIER",
    "MAX_ANIMATION_FRAMES",
    "MAX_CONTEXT_LENGTH",
    "MAX_RETRIES",
    "MAX_RUNTIME_SECONDS",
    "MODEL",
    "MODEL_CONFIG_ID",
    "_adapter",
    "_client",
    "_stateful_adapter",
    "_summary_compactor",
    "_continuous_conversation",
    "_server_state",
    "_request_kwargs",
    "_previous_response_id",  # server_state only, which rehydration refuses
    "_pending_user_messages",  # server_state only, which rehydration refuses
    "_pricing",
    "_model_config_sha256",
    "_level_action_budgets",  # recomputed on resume, so budgets may change
    "analysis_mode",
}
# Per-session identity or per-turn scratch that does not carry across steps.
SESSION = {
    "ROOT_URL",
    "_cleanup",
    "agent_name",
    "arc_env",
    "card_id",
    "config",
    "exit_reason",
    "game_id",
    "headers",
    "recorder",
    "run_dir",
    "_last_turn_result",
    "_timed_out",
    "_rehydration",  # this session's inputs
}
CLASSIFIED = SNAPSHOTTED | IN_FLIGHT | FROM_CONFIG | SESSION


class _FakeModelAdapter:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = responses
        self.requests: list[ModelRequest] = []

    def invoke(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self._responses.pop(0)


def _raw_frame(
    state: GameState,
    value: int,
    action: GameAction,
    guid: str = "guid-1",
    levels_completed: int = 0,
) -> FrameDataRaw:
    raw = FrameDataRaw()
    raw.game_id = "game-id"
    raw.frame = [np.array([[value]], dtype=np.int8)]
    raw.state = state
    raw.levels_completed = levels_completed
    raw.win_levels = 2
    raw.action_input = ActionInput(id=action, data={}, reasoning=None)
    raw.guid = guid
    raw.full_reset = False
    raw.available_actions = [GameAction.ACTION1.value, GameAction.ACTION6.value]
    return raw


def _recording_event(
    raw: FrameDataRaw,
    action: GameAction,
    data: dict,
    reasoning: dict,
    full_reset: bool = False,
) -> dict:
    return {
        "timestamp": "2026-10-01T00:00:00+00:00",
        "data": {
            "game_id": raw.game_id,
            "state": raw.state.name,
            "levels_completed": raw.levels_completed,
            "win_levels": raw.win_levels,
            "action_input": {
                "id": action.name,
                "data": dict(data),
                "reasoning": json.dumps(reasoning) if reasoning else None,
            },
            "guid": raw.guid,
            "full_reset": full_reset,
            "available_actions": raw.available_actions,
            "frame": [layer.tolist() for layer in raw.frame],
        },
    }


class _ScriptedEnv:
    """Deterministic fake game session.

    Starts NOT_PLAYED by default (forcing RESET). Each frame depends on the
    full action history, so a fresh session replaying the same actions sees the
    same frames and a different action diverges. ``game_over_at`` makes that
    step's frame GAME_OVER; ``level_up_at`` completes level 1 from that step on.
    Steps are also kept as toolkit-format recording events, with reasoning
    stored as the JSON string the remote client sends.
    """

    def __init__(
        self,
        guid: str = "guid-1",
        diverge_at: int | None = None,
        fail_at: int | None = None,
        initial_state: GameState = GameState.NOT_PLAYED,
        game_over_at: int | None = None,
        level_up_at: int | None = None,
        baseline_actions: list[int] | None = None,
    ) -> None:
        self.info = SimpleNamespace(baseline_actions=baseline_actions or [])
        self.guid = guid
        self.diverge_at = diverge_at
        self.fail_at = fail_at
        self.game_over_at = game_over_at
        self.level_up_at = level_up_at
        self.observation_space = _raw_frame(initial_state, 0, GameAction.RESET, guid)
        self._make_event = _recording_event(
            self.observation_space, GameAction.RESET, {}, {}, full_reset=True
        )
        self.history: list[int] = []
        self.calls: list[tuple[str, dict, dict]] = []
        self.events: list[dict] = []

    def step(
        self, action: GameAction, *, data: dict, reasoning: dict
    ) -> FrameDataRaw | None:
        self.calls.append((action.name, dict(data), reasoning))
        if len(self.calls) == self.fail_at:
            return None
        self.history.append(action.value + data.get("x", 0))
        value = len(self.history) * 10 + sum(self.history)
        if len(self.history) == self.diverge_at:
            value += 1
        state = (
            GameState.GAME_OVER
            if len(self.history) == self.game_over_at
            else GameState.NOT_FINISHED
        )
        levels = int(
            self.level_up_at is not None and len(self.history) >= self.level_up_at
        )
        raw = _raw_frame(state, value, action, self.guid, levels)
        self.events.append(_recording_event(raw, action, data, reasoning))
        self.observation_space = raw
        return raw

    def write_recording(self, path: Path) -> Path:
        """Write the toolkit recording: the implicit make() RESET, then each step."""
        events = [self._make_event, *self.events]
        path.write_text("".join(json.dumps(event) + "\n" for event in events))
        return path


def _usage() -> NormalizedUsage:
    return NormalizedUsage(input_tokens=10, output_tokens=5, total_tokens=15)


def _build_agent(
    monkeypatch, tmp_path, config_id: str, env=None, rehydration=None
) -> BenchmarkingAgent:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ARC_HARNESS_COMMIT_SHA", "commit-abc")
    monkeypatch.setattr(
        "benchmarking.agent.build_model_runtime_client", lambda **_kwargs: object()
    )
    return BenchmarkingAgent(
        card_id="card-1",
        game_id="game-id",
        agent_name="agent-name",
        ROOT_URL="https://arcprize.org",
        record=False,
        arc_env=env or SimpleNamespace(info=SimpleNamespace(baseline_actions=[])),
        config=config_id,
        rehydration=rehydration,
    )


def _manual_texts(actions: int) -> list[str]:
    # Step 1 is a forced RESET; the model chooses the rest, ending on ACTION6.
    return ["ACTION1"] * (actions - 2) + ["ACTION6 3 4"]


def _manual_responses(texts: list[str]) -> list[ModelResponse]:
    return [ModelResponse(output_text=text, usage=_usage()) for text in texts]


def _run_manual(monkeypatch, tmp_path, actions: int = 4) -> BenchmarkingAgent:
    agent = _build_agent(monkeypatch, tmp_path, MANUAL_CONFIG, env=_ScriptedEnv())
    agent._adapter = _FakeModelAdapter(_manual_responses(_manual_texts(actions)))
    agent.MAX_ACTIONS = actions - 1  # main() runs while action_counter <= MAX_ACTIONS
    agent.main()
    return agent


def _openai_response(step: int) -> ModelResponse:
    return ModelResponse(
        output_text="ACTION1",
        usage=_usage(),
        raw_response={
            "output": [
                {
                    "type": "reasoning",
                    "id": f"rs_{step}",
                    "summary": [{"type": "summary_text", "text": f"thought {step}"}],
                    "encrypted_content": f"opaque-{step}",
                },
                {
                    "type": "message",
                    "id": f"msg_{step}",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "ACTION1"}],
                },
            ]
        },
    )


def _run_openai_continuous(monkeypatch, tmp_path, actions: int = 3) -> BenchmarkingAgent:
    agent = _build_agent(
        monkeypatch, tmp_path, OPENAI_CONTINUOUS_CONFIG, env=_ScriptedEnv()
    )
    agent._stateful_adapter._model_adapter = _FakeModelAdapter(
        [_openai_response(step) for step in range(2, actions + 1)]
    )
    agent.MAX_ACTIONS = actions - 1
    agent.main()
    return agent


def _snapshot_names(agent: BenchmarkingAgent) -> list[str]:
    return sorted(p.name for p in (Path(agent.run_dir) / "state").iterdir())


def _latest_snapshot(agent: BenchmarkingAgent) -> AgentSnapshot:
    return load_snapshot(Path(agent.run_dir) / "state" / _snapshot_names(agent)[-1])


@pytest.mark.unit
class TestSnapshotsDuringRun:
    def test_writes_rolling_snapshots_every_step(self, monkeypatch, tmp_path):
        agent = _run_manual(monkeypatch, tmp_path, actions=5)
        assert agent.step_counter == 5
        assert _snapshot_names(agent) == [
            "state_step_0003.json",
            "state_step_0004.json",
            "state_step_0005.json",
        ]
        steps = [
            load_snapshot(Path(agent.run_dir) / "state" / name).step
            for name in _snapshot_names(agent)
        ]
        assert steps == [3, 4, 5]

    def test_latest_snapshot_matches_agent_state(self, monkeypatch, tmp_path):
        agent = _run_manual(monkeypatch, tmp_path)
        snapshot = _latest_snapshot(agent)
        latest_frame = agent.frames[-1]

        assert snapshot.step == agent.step_counter == agent.action_counter == 4
        assert snapshot.harness_commit_sha == "commit-abc"
        assert snapshot.source.model_dump() == {
            "run_id": agent.run_record.run_id,
            "guid": "guid-1",
            "game_id": "game-id",
            "card_id": "card-1",
        }
        assert snapshot.lineage == []
        assert snapshot.model_config_id == MANUAL_CONFIG
        assert snapshot.model_config_sha256 == config_sha256(
            get_model_config(MANUAL_CONFIG)
        )
        assert snapshot.pricing == agent._pricing
        assert snapshot.last_frame == frame_fingerprint(latest_frame)
        assert snapshot.runtime_state is None

        fields = snapshot.agent
        assert fields.conversation == agent.conversation
        assert len(fields.conversation) == 7  # system + 3 user/assistant pairs
        assert fields.token_counter == agent.token_counter == 45
        assert fields.level_action_counter == agent._level_action_counter == 4
        assert fields.last_levels_completed == 0
        assert fields.level_just_advanced is False
        assert fields.compaction_counter == 0
        assert fields.pending_compaction_trigger_tokens is None
        assert 0 <= fields.elapsed_seconds < 60
        assert fields.total_usage == agent.run_record.total_usage
        assert fields.total_usage.total_tokens == 45

    def test_continuous_snapshot_keeps_opaque_runtime_state(
        self, monkeypatch, tmp_path
    ):
        agent = _run_openai_continuous(monkeypatch, tmp_path)
        snapshot = _latest_snapshot(agent)

        assert snapshot.step == 3
        assert snapshot.runtime_state == agent._runtime_state
        items = snapshot.runtime_state.payload["input_items"]
        assert [item.get("encrypted_content") for item in items if item.get("type") == "reasoning"] == [
            "opaque-2",
            "opaque-3",
        ]
        assert len(snapshot.runtime_state.accepted_turns) == 2

    def test_snapshot_failure_does_not_stop_the_run(self, monkeypatch, tmp_path, caplog):
        def fail(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr("benchmarking.agent.write_snapshot_atomic", fail)
        with caplog.at_level(logging.ERROR):
            agent = _run_manual(monkeypatch, tmp_path)

        assert agent.step_counter == 4
        assert not (Path(agent.run_dir) / "state").exists()
        assert "Failed to write state snapshot after step 1" in caplog.text


@pytest.mark.unit
class TestSnapshotRules:
    def _boundary_agent(self, monkeypatch, tmp_path, config_id: str) -> BenchmarkingAgent:
        agent = _build_agent(monkeypatch, tmp_path, config_id)
        agent.frames.append(
            FrameData(
                frame=[[[1]]],
                state=GameState.NOT_FINISHED,
                levels_completed=0,
                available_actions=[1],
                guid="guid-1",
            )
        )
        agent.guid = "guid-1"
        agent.timer = 0.0
        agent.step_counter = agent.action_counter = 1
        return agent

    @pytest.mark.parametrize(
        ("attribute", "value"),
        [
            ("_pending_turn_messages", [Message(role="user", content="frame")]),
            ("_pending_action_reasoning", {"output": "x"}),
            ("_pending_compaction_usage", NormalizedUsage()),
            ("_pending_compaction_continuation", object()),
        ],
    )
    def test_refuses_in_flight_state(self, monkeypatch, tmp_path, attribute, value):
        agent = self._boundary_agent(monkeypatch, tmp_path, OPENAI_CONTINUOUS_CONFIG)
        setattr(agent, attribute, value)
        with pytest.raises(RuntimeError, match=attribute):
            agent._snapshot()

    def test_refuses_counter_mismatch(self, monkeypatch, tmp_path):
        agent = self._boundary_agent(monkeypatch, tmp_path, MANUAL_CONFIG)
        agent.action_counter = 2
        with pytest.raises(RuntimeError, match="step_counter=1 != action_counter=2"):
            agent._snapshot()

    def test_after_action_skips_when_no_frame(self, monkeypatch, tmp_path, caplog):
        agent = self._boundary_agent(monkeypatch, tmp_path, MANUAL_CONFIG)
        with caplog.at_level(logging.WARNING):
            agent._after_action(None)
        assert not (Path(agent.run_dir) / "state").exists()
        assert "action produced no frame" in caplog.text

    def test_after_action_logs_instead_of_raising(self, monkeypatch, tmp_path, caplog):
        agent = self._boundary_agent(monkeypatch, tmp_path, MANUAL_CONFIG)
        agent._pending_action_reasoning = {"output": "x"}
        with caplog.at_level(logging.ERROR):
            agent._after_action(agent.frames[-1])
        assert not (Path(agent.run_dir) / "state").exists()
        assert "Failed to write state snapshot" in caplog.text

    def test_missing_commit_sha_is_recorded_as_none(self, monkeypatch, tmp_path):
        agent = self._boundary_agent(monkeypatch, tmp_path, MANUAL_CONFIG)
        monkeypatch.delenv("ARC_HARNESS_COMMIT_SHA")
        assert agent._snapshot().harness_commit_sha is None

    @pytest.mark.parametrize("config_id", CONTINUOUS_CONFIGS)
    def test_runtime_state_round_trips_for_every_adapter(
        self, monkeypatch, tmp_path, config_id
    ):
        agent = self._boundary_agent(monkeypatch, tmp_path, config_id)
        # A buffered observation (e.g. after a forced RESET on GAME_OVER) is
        # legitimate carried state at a boundary.
        agent._runtime_state = agent._stateful_adapter.buffer_inputs(
            agent._runtime_state, [Message(role="user", content="frame text")]
        )
        snapshot = agent._snapshot()

        loaded = AgentSnapshot.model_validate_json(snapshot.model_dump_json())

        assert loaded == snapshot
        assert loaded.runtime_state == agent._runtime_state
        loaded.runtime_state.validate_for(
            adapter_id=agent._stateful_adapter.descriptor.adapter_id,
            strategy=agent._stateful_adapter.strategy,
        )


def _assert_classified(agent: BenchmarkingAgent) -> None:
    unclassified = set(vars(agent)) - CLASSIFIED
    assert not unclassified, (
        f"Unclassified BenchmarkingAgent attributes {sorted(unclassified)}: "
        "add them to the rehydration snapshot (SNAPSHOTTED) or to the "
        "IN_FLIGHT, FROM_CONFIG, or SESSION sets in this test."
    )


@pytest.mark.unit
class TestAttributeGuard:
    """Fails when a new agent attribute is not classified (plan F2)."""

    def test_classification_sets_are_disjoint(self):
        sets = [SNAPSHOTTED, IN_FLIGHT, FROM_CONFIG, SESSION]
        assert sum(len(s) for s in sets) == len(CLASSIFIED)

    @pytest.mark.parametrize("config_id", list_model_config_ids())
    def test_constructed_agent_attributes_are_classified(
        self, monkeypatch, tmp_path, config_id
    ):
        _assert_classified(_build_agent(monkeypatch, tmp_path, config_id))

    def test_manual_run_attributes_are_classified(self, monkeypatch, tmp_path):
        _assert_classified(_run_manual(monkeypatch, tmp_path))

    def test_continuous_run_attributes_are_classified(self, monkeypatch, tmp_path):
        _assert_classified(_run_openai_continuous(monkeypatch, tmp_path))


def _prepare(
    source: BenchmarkingAgent, tmp_path: Path, step: int, **snapshot_updates
) -> PreparedRehydration:
    recording = source.arc_env.write_recording(tmp_path / f"{source.guid}.jsonl")
    snapshot = load_snapshot(
        Path(source.run_dir) / "state" / f"state_step_{step:04d}.json"
    )
    if snapshot_updates:
        snapshot = snapshot.model_copy(update=snapshot_updates)
    return PreparedRehydration(
        snapshot=snapshot, steps=parse_toolkit_recording(recording)[1 : step + 1]
    )


def _rehydrated_manual(
    monkeypatch, tmp_path, prepared: PreparedRehydration, total_actions: int, **env_kwargs
) -> BenchmarkingAgent:
    agent = _build_agent(
        monkeypatch,
        tmp_path,
        MANUAL_CONFIG,
        env=_ScriptedEnv(guid="guid-2", **env_kwargs),
        rehydration=prepared,
    )
    remaining = _manual_texts(total_actions)[prepared.snapshot.step - 1 :]
    agent._adapter = _FakeModelAdapter(_manual_responses(remaining))
    agent.MAX_ACTIONS = total_actions - 1
    return agent


def _comparable(snapshot: AgentSnapshot) -> dict:
    """Snapshot content that must match an uninterrupted run."""
    data = snapshot.model_dump(
        mode="json", exclude={"created_at", "source", "lineage"}
    )
    data["agent"].pop("elapsed_seconds")
    return data


@pytest.mark.unit
class TestRehydrationReplay:
    def test_resumed_manual_run_matches_uninterrupted_run(self, monkeypatch, tmp_path):
        original = _run_manual(monkeypatch, tmp_path, actions=5)
        prepared = _prepare(original, tmp_path, step=3)
        resumed = _rehydrated_manual(monkeypatch, tmp_path, prepared, total_actions=5)

        resumed.main()

        # Same actions and reasoning metadata reach the game, replayed or not.
        assert resumed.arc_env.calls == original.arc_env.calls
        assert [e["data"]["action_input"] for e in resumed.arc_env.events] == [
            e["data"]["action_input"] for e in original.arc_env.events
        ]
        # The model sees exactly what the uninterrupted run sent at steps 4-5.
        assert resumed._adapter.requests == original._adapter.requests[2:]
        assert resumed.step_counter == resumed.action_counter == 5
        assert resumed.run_record.total_usage == original.run_record.total_usage
        assert resumed.run_record.total_steps == 5
        # Replayed steps write no step files.
        assert sorted(p.name for p in Path(resumed.run_dir).glob("step_*.json")) == [
            "step_004.json",
            "step_005.json",
        ]
        assert _comparable(_latest_snapshot(resumed)) == _comparable(
            _latest_snapshot(original)
        )

    def test_snapshots_carry_lineage_and_new_session_identity(
        self, monkeypatch, tmp_path
    ):
        original = _run_manual(monkeypatch, tmp_path, actions=5)
        prepared = _prepare(original, tmp_path, step=3)
        resumed = _rehydrated_manual(monkeypatch, tmp_path, prepared, total_actions=5)

        resumed.main()

        # Step 3 is snapshotted at the resume point, before any new step.
        assert _snapshot_names(resumed) == [
            "state_step_0003.json",
            "state_step_0004.json",
            "state_step_0005.json",
        ]
        latest = _latest_snapshot(resumed)
        assert latest.source.run_id == resumed.run_record.run_id
        assert latest.source.guid == "guid-2"
        assert [entry.model_dump() for entry in latest.lineage] == [
            {
                "run_id": original.run_record.run_id,
                "guid": "guid-1",
                "rehydrated_at_step": 3,
            }
        ]

    def test_elapsed_time_carries_forward(self, monkeypatch, tmp_path):
        original = _run_manual(monkeypatch, tmp_path, actions=5)
        prepared = _prepare(original, tmp_path, step=3)
        prepared.snapshot.agent.elapsed_seconds = 1_000.0
        resumed = _rehydrated_manual(monkeypatch, tmp_path, prepared, total_actions=5)

        resumed.main()

        assert resumed._elapsed_offset_seconds == 1_000.0
        resume_point = load_snapshot(
            Path(resumed.run_dir) / "state" / "state_step_0003.json"
        )
        assert 1_000.0 <= resume_point.agent.elapsed_seconds < 1_060.0
        assert 1_000.0 <= _latest_snapshot(resumed).agent.elapsed_seconds < 1_060.0

    def test_chained_rehydration(self, monkeypatch, tmp_path):
        original = _run_manual(monkeypatch, tmp_path, actions=5)
        first = _rehydrated_manual(
            monkeypatch, tmp_path, _prepare(original, tmp_path, step=3), total_actions=5
        )
        first.main()
        second = _build_agent(
            monkeypatch,
            tmp_path,
            MANUAL_CONFIG,
            env=_ScriptedEnv(guid="guid-3"),
            rehydration=_prepare(first, tmp_path, step=4),
        )
        second._adapter = _FakeModelAdapter(_manual_responses(_manual_texts(5)[3:]))
        second.MAX_ACTIONS = 4

        second.main()

        assert second.arc_env.calls == original.arc_env.calls
        assert _comparable(_latest_snapshot(second)) == _comparable(
            _latest_snapshot(original)
        )
        assert [
            (entry.guid, entry.rehydrated_at_step)
            for entry in _latest_snapshot(second).lineage
        ] == [("guid-1", 3), ("guid-2", 4)]

    def test_resumed_continuous_run_sends_identical_native_request(
        self, monkeypatch, tmp_path
    ):
        original = _run_openai_continuous(monkeypatch, tmp_path, actions=4)
        resumed = _build_agent(
            monkeypatch,
            tmp_path,
            OPENAI_CONTINUOUS_CONFIG,
            env=_ScriptedEnv(guid="guid-2"),
            rehydration=_prepare(original, tmp_path, step=3),
        )
        resumed._stateful_adapter._model_adapter = _FakeModelAdapter(
            [_openai_response(4)]
        )
        resumed.MAX_ACTIONS = 3

        resumed.main()

        original_requests = original._stateful_adapter._model_adapter.requests
        resumed_requests = resumed._stateful_adapter._model_adapter.requests
        assert resumed_requests == original_requests[-1:]
        assert "opaque-3" in json.dumps(resumed_requests[0].native_input)
        assert resumed._runtime_state == original._runtime_state
        assert resumed.arc_env.calls == original.arc_env.calls


@pytest.mark.unit
class TestRehydrationFailures:
    def _assert_failed_before_model(self, agent: BenchmarkingAgent) -> None:
        assert agent.exit_reason == ExitReason.REHYDRATION_ERROR
        assert agent._adapter.requests == []
        assert not (Path(agent.run_dir) / "state").exists()
        assert list(Path(agent.run_dir).glob("step_*.json")) == []

    def test_divergent_frame_aborts_replay(self, monkeypatch, tmp_path):
        original = _run_manual(monkeypatch, tmp_path, actions=5)
        resumed = _rehydrated_manual(
            monkeypatch,
            tmp_path,
            _prepare(original, tmp_path, step=3),
            total_actions=5,
            diverge_at=2,
        )
        with pytest.raises(RehydrationError, match="diverged at step 2"):
            resumed.main()
        assert len(resumed.arc_env.calls) == 2
        self._assert_failed_before_model(resumed)

    def test_failed_step_aborts_without_retry(self, monkeypatch, tmp_path):
        original = _run_manual(monkeypatch, tmp_path, actions=5)
        resumed = _rehydrated_manual(
            monkeypatch,
            tmp_path,
            _prepare(original, tmp_path, step=3),
            total_actions=5,
            fail_at=2,
        )
        with pytest.raises(ValueError, match="None frame data"):
            resumed.main()
        assert len(resumed.arc_env.calls) == 2
        self._assert_failed_before_model(resumed)

    def test_changed_level_budgets_use_current_budgets(self, monkeypatch, tmp_path):
        original = _run_manual(monkeypatch, tmp_path, actions=5)
        prepared = _prepare(original, tmp_path, step=3)
        resumed = _rehydrated_manual(
            monkeypatch, tmp_path, prepared, total_actions=5, baseline_actions=[2]
        )
        resumed.main()
        assert original._level_action_budgets == []
        assert resumed._level_action_budgets == [10]
        assert resumed.step_counter == 5



@pytest.mark.unit
class TestRehydratedAttributeGuard:
    def test_rehydrated_run_attributes_are_classified(self, monkeypatch, tmp_path):
        original = _run_manual(monkeypatch, tmp_path, actions=5)
        resumed = _rehydrated_manual(
            monkeypatch, tmp_path, _prepare(original, tmp_path, step=3), total_actions=5
        )
        resumed.main()
        _assert_classified(resumed)


def _run_meta(agent: BenchmarkingAgent) -> dict:
    return json.loads((Path(agent.run_dir) / "run_meta.json").read_text())


@pytest.mark.unit
class TestRehydrationRunMeta:
    def test_normal_run_meta_has_no_rehydration_key(self, monkeypatch, tmp_path):
        agent = _run_manual(monkeypatch, tmp_path)
        assert "rehydration" not in _run_meta(agent)

    def test_rehydrated_run_meta_records_provenance(self, monkeypatch, tmp_path):
        original = _run_manual(monkeypatch, tmp_path, actions=5)
        prepared = _prepare(original, tmp_path, step=3)
        resumed = _rehydrated_manual(monkeypatch, tmp_path, prepared, total_actions=5)

        resumed.main()

        meta = _run_meta(resumed)
        assert meta["rehydration"] == {
            "source_run_id": original.run_record.run_id,
            "source_guid": "guid-1",
            "source_card_id": "card-1",
            "replayed_steps": 3,
            "prior_elapsed_seconds": prepared.snapshot.agent.elapsed_seconds,
            "lineage": [
                {
                    "run_id": original.run_record.run_id,
                    "guid": "guid-1",
                    "rehydrated_at_step": 3,
                }
            ],
        }
        assert meta["total_steps"] == 5

    def test_pricing_change_is_recorded_not_blocking(
        self, monkeypatch, tmp_path, caplog
    ):
        original = _run_manual(monkeypatch, tmp_path, actions=5)
        prepared = _prepare(original, tmp_path, step=3, pricing={"input": 0.1})
        resumed = _rehydrated_manual(monkeypatch, tmp_path, prepared, total_actions=5)

        with caplog.at_level(logging.WARNING):
            resumed.main()

        assert resumed.step_counter == 5
        assert _run_meta(resumed)["rehydration"]["pricing_changed"] == {
            "previous": {"input": 0.1},
            "current": resumed._pricing,
        }
        assert "Pricing changed since the snapshot" in caplog.text


@pytest.mark.unit
class TestRunIdentity:
    def test_run_is_named_after_session_guid(self, monkeypatch, tmp_path):
        agent = _build_agent(
            monkeypatch, tmp_path, MANUAL_CONFIG, env=_ScriptedEnv(guid="session-1")
        )
        assert agent.run_record.run_id == "session-1"
        assert agent.run_dir == f"recordings/{agent.name}.session-1"
        meta = _run_meta(agent)
        assert (meta["run_id"], meta["guid"], meta["card_id"]) == (
            "session-1",
            "session-1",
            "card-1",
        )

    def test_falls_back_to_random_id_without_session_guid(self, monkeypatch, tmp_path):
        agent = _build_agent(monkeypatch, tmp_path, MANUAL_CONFIG)
        assert len(agent.run_record.run_id) == 36  # uuid4
        assert _run_meta(agent)["guid"] is None

    def test_reused_session_guid_fails_instead_of_merging_runs(
        self, monkeypatch, tmp_path
    ):
        _build_agent(monkeypatch, tmp_path, MANUAL_CONFIG, env=_ScriptedEnv(guid="dup"))
        with pytest.raises(FileExistsError):
            _build_agent(
                monkeypatch, tmp_path, MANUAL_CONFIG, env=_ScriptedEnv(guid="dup")
            )

    def test_rehydrated_run_uses_new_session_guid(self, monkeypatch, tmp_path):
        original = _run_manual(monkeypatch, tmp_path, actions=5)
        resumed = _rehydrated_manual(
            monkeypatch, tmp_path, _prepare(original, tmp_path, step=3), total_actions=5
        )
        resumed.main()
        assert original.run_record.run_id == "guid-1"
        assert resumed.run_record.run_id == "guid-2"
        assert _run_meta(resumed)["rehydration"]["source_run_id"] == "guid-1"
