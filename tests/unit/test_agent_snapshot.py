import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from arcengine import ActionInput, FrameData, FrameDataRaw, GameAction, GameState

from benchmarking.agent import BenchmarkingAgent
from benchmarking.model_config import get_model_config, list_model_config_ids
from benchmarking.rehydration import (
    AgentSnapshot,
    config_sha256,
    frame_fingerprint,
    load_snapshot,
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
    "_previous_action",
    "_previous_response_id",
    "_pending_user_messages",
    "_runtime_state",
    "timer",  # as elapsed_seconds
    "run_record",  # total_usage; identity fields are per-session
    "frames",  # rebuilt by replay; last frame is fingerprinted
    "guid",  # source guid; the new session gets its own
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
    "_pricing",
    "_model_config_sha256",
    "_level_action_budgets",  # also stored in the snapshot and validated
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
}
CLASSIFIED = SNAPSHOTTED | IN_FLIGHT | FROM_CONFIG | SESSION


class _FakeModelAdapter:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = responses
        self.requests: list[ModelRequest] = []

    def invoke(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self._responses.pop(0)


def _raw_frame(state: GameState, value: int, action: GameAction) -> FrameDataRaw:
    raw = FrameDataRaw()
    raw.game_id = "game-id"
    raw.frame = [np.array([[value]], dtype=np.int8)]
    raw.state = state
    raw.levels_completed = 0
    raw.win_levels = 1
    raw.action_input = ActionInput(id=action, data={}, reasoning=None)
    raw.guid = "guid-1"
    raw.full_reset = False
    raw.available_actions = [GameAction.ACTION1.value, GameAction.ACTION6.value]
    return raw


class _ScriptedEnv:
    """Starts NOT_PLAYED (forcing RESET), then returns a distinct frame per step."""

    def __init__(self) -> None:
        self.info = SimpleNamespace(baseline_actions=[])
        self.observation_space = _raw_frame(GameState.NOT_PLAYED, 0, GameAction.RESET)
        self.steps = 0

    def step(self, action: GameAction, *, data: dict, reasoning: dict) -> FrameDataRaw:
        self.steps += 1
        self.observation_space = _raw_frame(GameState.NOT_FINISHED, self.steps, action)
        return self.observation_space


def _usage() -> NormalizedUsage:
    return NormalizedUsage(input_tokens=10, output_tokens=5, total_tokens=15)


def _build_agent(monkeypatch, tmp_path, config_id: str, env=None) -> BenchmarkingAgent:
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
    )


def _run_manual(monkeypatch, tmp_path, actions: int = 4) -> BenchmarkingAgent:
    agent = _build_agent(monkeypatch, tmp_path, MANUAL_CONFIG, env=_ScriptedEnv())
    # Step 1 is a forced RESET; the model chooses the rest, ending on ACTION6.
    texts = ["ACTION1"] * (actions - 2) + ["ACTION6 3 4"]
    agent._adapter = _FakeModelAdapter(
        [ModelResponse(output_text=text, usage=_usage()) for text in texts]
    )
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
        assert snapshot.level_action_budgets == agent._level_action_budgets
        assert snapshot.last_frame == frame_fingerprint(
            state=latest_frame.state,
            levels_completed=latest_frame.levels_completed,
            available_actions=latest_frame.available_actions,
            frame=latest_frame.frame,
        )
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
        assert fields.previous_action.model_dump() == {
            "name": "ACTION6",
            "data": {"x": 3, "y": 4},
        }
        assert fields.previous_response_id is None
        assert fields.pending_user_messages == []
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
        agent.action_counter = 1
        with pytest.raises(RuntimeError, match="step_counter=0 != action_counter=1"):
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


@pytest.mark.unit
class TestAttributeGuard:
    """Fails when a new agent attribute is not classified (plan F2)."""

    def _assert_classified(self, agent: BenchmarkingAgent) -> None:
        unclassified = set(vars(agent)) - CLASSIFIED
        assert not unclassified, (
            f"Unclassified BenchmarkingAgent attributes {sorted(unclassified)}: "
            "add them to the rehydration snapshot (SNAPSHOTTED) or to the "
            "IN_FLIGHT, FROM_CONFIG, or SESSION sets in this test."
        )

    def test_classification_sets_are_disjoint(self):
        sets = [SNAPSHOTTED, IN_FLIGHT, FROM_CONFIG, SESSION]
        assert sum(len(s) for s in sets) == len(CLASSIFIED)

    @pytest.mark.parametrize("config_id", list_model_config_ids())
    def test_constructed_agent_attributes_are_classified(
        self, monkeypatch, tmp_path, config_id
    ):
        self._assert_classified(_build_agent(monkeypatch, tmp_path, config_id))

    def test_manual_run_attributes_are_classified(self, monkeypatch, tmp_path):
        self._assert_classified(_run_manual(monkeypatch, tmp_path))

    def test_continuous_run_attributes_are_classified(self, monkeypatch, tmp_path):
        self._assert_classified(_run_openai_continuous(monkeypatch, tmp_path))
