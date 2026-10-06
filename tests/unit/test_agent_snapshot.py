import json
import logging
from pathlib import Path

import pytest
from arcengine import FrameData, GameState

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
from tests.unit.rehydration_fakes import (
    ScriptedEnv,
    build_agent,
    comparable_snapshot,
    latest_snapshot,
    snapshot_names,
)

MANUAL_CONFIG = "openai-gpt-5-4-2026-03-05"
OPENAI_CONTINUOUS_CONFIG = "openai-gpt-5-6-sol-max-provider-adapter"
CONTINUOUS_CONFIGS = [
    config_id
    for config_id in list_model_config_ids()
    if get_model_config(config_id)["runtime"].get("state") == "continuous_conversation"
]

# Every BenchmarkingAgent instance attribute must be classified here.
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
    "_max_actions_hard_cap",  # from MAX_ACTIONS_HARD_CAP env var, re-read on resume
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


def _usage() -> NormalizedUsage:
    return NormalizedUsage(input_tokens=10, output_tokens=5, total_tokens=15)


def _manual_texts(actions: int) -> list[str]:
    # Step 1 is a forced RESET; the model chooses the rest, ending on ACTION6.
    return ["ACTION1"] * (actions - 2) + ["ACTION6 3 4"]


def _manual_responses(texts: list[str]) -> list[ModelResponse]:
    return [ModelResponse(output_text=text, usage=_usage()) for text in texts]


def _run_manual(monkeypatch, tmp_path, actions: int = 4) -> BenchmarkingAgent:
    agent = build_agent(monkeypatch, tmp_path, MANUAL_CONFIG, env=ScriptedEnv())
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


def _run_openai_continuous(
    monkeypatch, tmp_path, actions: int = 3, env=None
) -> BenchmarkingAgent:
    agent = build_agent(
        monkeypatch, tmp_path, OPENAI_CONTINUOUS_CONFIG, env=env or ScriptedEnv()
    )
    agent._stateful_adapter._model_adapter = _FakeModelAdapter(
        [_openai_response(step) for step in range(2, actions + 1)]
    )
    agent.MAX_ACTIONS = actions - 1
    agent.main()
    return agent


def _run_meta(agent: BenchmarkingAgent) -> dict:
    return json.loads((Path(agent.run_dir) / "run_meta.json").read_text())


@pytest.mark.unit
class TestSnapshotsDuringRun:
    def test_writes_rolling_snapshots_every_step(self, monkeypatch, tmp_path):
        agent = _run_manual(monkeypatch, tmp_path, actions=5)
        assert agent.step_counter == 5
        assert snapshot_names(agent) == [
            "state_step_0003.json",
            "state_step_0004.json",
            "state_step_0005.json",
        ]

    def test_latest_snapshot_matches_agent_state(self, monkeypatch, tmp_path):
        agent = _run_manual(monkeypatch, tmp_path)
        snapshot = latest_snapshot(agent)
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

    def test_snapshot_failure_does_not_stop_the_run(self, monkeypatch, tmp_path, caplog):
        def fail(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr("benchmarking.agent.write_snapshot_atomic", fail)
        with caplog.at_level(logging.ERROR):
            agent = _run_manual(monkeypatch, tmp_path)

        assert agent.step_counter == 4
        assert not (Path(agent.run_dir) / "state").exists()
        assert "Failed to write state snapshot after step 1" in caplog.text


def _messages_sent(agent: BenchmarkingAgent, step: int) -> list[dict]:
    path = Path(agent.run_dir) / f"step_{step:03d}.json"
    return json.loads(path.read_text())["messages_sent"]


@pytest.mark.unit
class TestContinuousTranscript:
    """Continuous configs keep no conversation mirror, so logs and snapshots
    stay bounded by provider state, which compaction keeps small."""

    def test_keeps_no_mirror_and_logs_only_new_input(
        self, monkeypatch, tmp_path, caplog
    ):
        with caplog.at_level(logging.INFO):
            agent = _run_openai_continuous(monkeypatch, tmp_path, actions=4)

        assert agent.step_counter == 4
        assert agent.conversation == []
        assert latest_snapshot(agent).agent.conversation == []
        assert "messages:" not in caplog.text
        # OpenAI returns no readable projection, so each step logs the system
        # prompt and that turn's new frame instead of a growing history.
        for step in (2, 3, 4):
            assert [m["role"] for m in _messages_sent(agent, step)] == [
                "system",
                "user",
            ]

    def test_forced_reset_logs_only_the_buffered_observation(
        self, monkeypatch, tmp_path
    ):
        agent = _run_openai_continuous(
            monkeypatch, tmp_path, actions=4, env=ScriptedEnv(game_over_at=2)
        )

        assert agent.step_counter == 4
        assert agent.conversation == []
        # Step 1 resets a NOT_PLAYED game, so there is nothing to observe.
        assert _messages_sent(agent, 1) == []
        # Step 3 resets after GAME_OVER: only that frame is logged.
        [observation] = _messages_sent(agent, 3)
        assert observation["role"] == "user"
        assert observation["content"].startswith("State: GAME_OVER")

    def test_manual_runs_keep_their_transcript(self, monkeypatch, tmp_path, caplog):
        with caplog.at_level(logging.INFO):
            agent = _run_manual(monkeypatch, tmp_path)

        assert len(agent.conversation) == 7
        assert "messages: 6" in caplog.text


@pytest.mark.unit
class TestSnapshotRules:
    def _boundary_agent(self, monkeypatch, tmp_path, config_id: str) -> BenchmarkingAgent:
        agent = build_agent(monkeypatch, tmp_path, config_id)
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
        assert snapshot.runtime_state == agent._runtime_state
        assert AgentSnapshot.model_validate_json(snapshot.model_dump_json()) == snapshot


def _assert_classified(agent: BenchmarkingAgent) -> None:
    unclassified = set(vars(agent)) - CLASSIFIED
    assert not unclassified, (
        f"Unclassified BenchmarkingAgent attributes {sorted(unclassified)}: "
        "add them to the rehydration snapshot (SNAPSHOTTED) or to the "
        "IN_FLIGHT, FROM_CONFIG, or SESSION sets in this test."
    )


@pytest.mark.unit
class TestAttributeGuard:
    """Fails when a new agent attribute is not classified."""

    def test_classification_sets_are_disjoint(self):
        sets = [SNAPSHOTTED, IN_FLIGHT, FROM_CONFIG, SESSION]
        assert sum(len(s) for s in sets) == len(CLASSIFIED)

    @pytest.mark.parametrize("config_id", list_model_config_ids())
    def test_constructed_agent_attributes_are_classified(
        self, monkeypatch, tmp_path, config_id
    ):
        _assert_classified(build_agent(monkeypatch, tmp_path, config_id))

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
    agent = build_agent(
        monkeypatch,
        tmp_path,
        MANUAL_CONFIG,
        env=ScriptedEnv(guid="guid-2", **env_kwargs),
        rehydration=prepared,
    )
    remaining = _manual_texts(total_actions)[prepared.snapshot.step - 1 :]
    agent._adapter = _FakeModelAdapter(_manual_responses(remaining))
    agent.MAX_ACTIONS = total_actions - 1
    return agent


@pytest.mark.unit
class TestRehydrationReplay:
    def test_resumed_run_has_new_identity_lineage_and_provenance(
        self, monkeypatch, tmp_path
    ):
        original = _run_manual(monkeypatch, tmp_path, actions=5)
        prepared = _prepare(original, tmp_path, step=3)
        resumed = _rehydrated_manual(monkeypatch, tmp_path, prepared, total_actions=5)

        resumed.main()

        assert resumed.run_record.run_id == "guid-2"
        # Step 3 is snapshotted at the resume point, before any new step.
        assert snapshot_names(resumed) == [
            "state_step_0003.json",
            "state_step_0004.json",
            "state_step_0005.json",
        ]
        latest = latest_snapshot(resumed)
        assert (latest.source.run_id, latest.source.guid) == ("guid-2", "guid-2")
        lineage = [{"run_id": "guid-1", "guid": "guid-1", "rehydrated_at_step": 3}]
        assert [entry.model_dump() for entry in latest.lineage] == lineage
        meta = _run_meta(resumed)
        assert meta["rehydration"] == {
            "source_run_id": "guid-1",
            "source_guid": "guid-1",
            "source_card_id": "card-1",
            "replayed_steps": 3,
            "prior_elapsed_seconds": prepared.snapshot.agent.elapsed_seconds,
            "lineage": lineage,
        }
        assert meta["total_steps"] == 5
        _assert_classified(resumed)

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
        assert 1_000.0 <= latest_snapshot(resumed).agent.elapsed_seconds < 1_060.0

    def test_chained_rehydration(self, monkeypatch, tmp_path):
        original = _run_manual(monkeypatch, tmp_path, actions=5)
        first = _rehydrated_manual(
            monkeypatch, tmp_path, _prepare(original, tmp_path, step=3), total_actions=5
        )
        first.main()
        second = build_agent(
            monkeypatch,
            tmp_path,
            MANUAL_CONFIG,
            env=ScriptedEnv(guid="guid-3"),
            rehydration=_prepare(first, tmp_path, step=4),
        )
        second._adapter = _FakeModelAdapter(_manual_responses(_manual_texts(5)[3:]))
        second.MAX_ACTIONS = 4

        second.main()

        assert second.arc_env.calls == original.arc_env.calls
        assert comparable_snapshot(latest_snapshot(second)) == comparable_snapshot(
            latest_snapshot(original)
        )
        assert [
            (entry.guid, entry.rehydrated_at_step)
            for entry in latest_snapshot(second).lineage
        ] == [("guid-1", 3), ("guid-2", 4)]

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


@pytest.mark.unit
class TestRehydrationRunMeta:
    def test_normal_run_meta_has_no_rehydration_key(self, monkeypatch, tmp_path):
        agent = _run_manual(monkeypatch, tmp_path)
        assert "rehydration" not in _run_meta(agent)

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
        agent = build_agent(
            monkeypatch, tmp_path, MANUAL_CONFIG, env=ScriptedEnv(guid="session-1")
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
        agent = build_agent(monkeypatch, tmp_path, MANUAL_CONFIG)
        assert len(agent.run_record.run_id) == 36  # uuid4
        assert _run_meta(agent)["guid"] is None

    def test_reused_session_guid_fails_instead_of_merging_runs(
        self, monkeypatch, tmp_path
    ):
        build_agent(monkeypatch, tmp_path, MANUAL_CONFIG, env=ScriptedEnv(guid="dup"))
        with pytest.raises(FileExistsError):
            build_agent(
                monkeypatch, tmp_path, MANUAL_CONFIG, env=ScriptedEnv(guid="dup")
            )

