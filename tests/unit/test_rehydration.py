import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from arcengine import GameAction, GameState
from pydantic import ValidationError

from benchmarking import rehydration
from benchmarking.model_config import get_model_config
from benchmarking.rehydration import (
    AgentFields,
    AgentSnapshot,
    FrameFingerprint,
    RehydrationArgs,
    RehydrationError,
    SnapshotSource,
    config_sha256,
    frame_fingerprint,
    load_snapshot,
    parse_rehydrate_args,
    parse_toolkit_recording,
    prepare_rehydration,
    write_snapshot_atomic,
)
from benchmarking.runtime_state import RuntimeState

FRAME = [[[0, 1], [2, 3]]]


def _event(
    action_id: str = "ACTION1",
    *,
    data: dict | None = None,
    reasoning: object = None,
    state: str = "NOT_FINISHED",
    frame: list | None = FRAME,
    guid: str = "guid-1",
    game_id: str = "ls20-abc123",
) -> dict:
    event_data = {
        "game_id": game_id,
        "state": state,
        "levels_completed": 0,
        "win_levels": 3,
        "action_input": {"id": action_id, "data": data or {}, "reasoning": reasoning},
        "guid": guid,
        "full_reset": False,
        "available_actions": [1, 2, 6],
    }
    if frame is not None:
        event_data["frame"] = frame
    return {"timestamp": "2026-10-01T00:00:00+00:00", "data": event_data}


def _fingerprint(**fields) -> FrameFingerprint:
    return frame_fingerprint(SimpleNamespace(**fields))


def _write_jsonl(path: Path, lines: list) -> Path:
    path.write_text(
        "".join(
            (line if isinstance(line, str) else json.dumps(line)) + "\n"
            for line in lines
        ),
        encoding="utf-8",
    )
    return path


def _snapshot(step: int = 3, runtime_state: RuntimeState | None = None) -> AgentSnapshot:
    return AgentSnapshot(
        harness_commit_sha="abc123",
        created_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
        source=SnapshotSource(
            run_id="run-1", guid="guid-1", game_id="ls20-abc123", card_id="card-1"
        ),
        model_config_id="cfg",
        model_config_sha256=config_sha256({"request": {"model": "m"}}),
        pricing={"input": 1.0, "output": 2.0},
        step=step,
        last_frame=_fingerprint(
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            available_actions=[1, 2],
            frame=FRAME,
        ),
        agent=AgentFields(
            conversation=[{"role": "system", "content": "play"}],
            token_counter=100,
            level_action_counter=3,
            last_levels_completed=0,
            level_just_advanced=False,
            compaction_counter=0,
            pending_compaction_trigger_tokens=175_000,
            elapsed_seconds=12.5,
            total_usage={"prompt_tokens": 60, "completion_tokens": 40, "total_tokens": 100},
        ),
        runtime_state=runtime_state,
    )


@pytest.mark.unit
class TestConfigHash:
    BASE = {
        "agent": {"MAX_CONTEXT_LENGTH": 1000},
        "runtime": {"sdk": "openai-python", "state": "manual_rolling"},
        "client": {"api_key_env": "OPENAI_API_KEY"},
        "request": {"model": "gpt", "max_output_tokens": 10},
        "pricing": {"input": 1.0, "output": 2.0},
    }

    def test_pricing_only_change_keeps_hash(self):
        changed = {**self.BASE, "pricing": {"input": 9.0, "output": 9.0}}
        assert config_sha256(changed) == config_sha256(self.BASE)

    def test_budget_multiplier_change_keeps_hash(self):
        agent = {**self.BASE["agent"], "MAX_ACTIONS_BASELINE_MULTIPLIER": 9.0}
        assert config_sha256({**self.BASE, "agent": agent}) == config_sha256(self.BASE)

    @pytest.mark.parametrize("section", ["agent", "runtime", "client", "request"])
    def test_any_other_section_change_changes_hash(self, section):
        changed = {**self.BASE, section: {**self.BASE[section], "new_key": 1}}
        assert config_sha256(changed) != config_sha256(self.BASE)

    def test_new_top_level_key_changes_hash(self):
        assert config_sha256({**self.BASE, "future": {}}) != config_sha256(self.BASE)

    def test_key_order_does_not_matter(self):
        reordered = dict(reversed(list(self.BASE.items())))
        assert config_sha256(reordered) == config_sha256(self.BASE)


@pytest.mark.unit
class TestFrameFingerprint:
    def test_enum_and_string_state_match(self):
        kwargs = {"levels_completed": 1, "available_actions": [1], "frame": FRAME}
        assert _fingerprint(state=GameState.WIN, **kwargs) == _fingerprint(
            state="WIN", **kwargs
        )

    def test_frame_change_changes_hash(self):
        kwargs = {"state": "NOT_FINISHED", "levels_completed": 0, "available_actions": [1]}
        assert _fingerprint(frame=FRAME, **kwargs) != _fingerprint(
            frame=[[[0, 1], [2, 4]]], **kwargs
        )

    def test_recorded_step_matches_live_frame(self, tmp_path):
        path = _write_jsonl(tmp_path / "r.jsonl", [_event()])
        recorded = frame_fingerprint(parse_toolkit_recording(path)[0])
        live = _fingerprint(
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            available_actions=[1, 2, 6],
            frame=FRAME,
        )
        assert recorded == live


@pytest.mark.unit
class TestParseToolkitRecording:
    def test_parses_action_events_in_order(self, tmp_path):
        path = _write_jsonl(
            tmp_path / "r.jsonl",
            [
                _event("RESET"),
                _event("ACTION6", data={"x": 3, "y": 4, "game_id": "ignored"}),
                _event("ACTION1", reasoning={"output": "go"}),
            ],
        )
        steps = parse_toolkit_recording(path)

        assert [(s.index, s.action) for s in steps] == [
            (0, "RESET"),
            (1, "ACTION6"),
            (2, "ACTION1"),
        ]
        assert steps[1].data == {"x": 3, "y": 4}
        assert steps[0].data == {}
        assert steps[2].reasoning == {"output": "go"}
        assert steps[0].guid == "guid-1"

    def test_game_action_carries_coordinates(self, tmp_path):
        path = _write_jsonl(tmp_path / "r.jsonl", [_event("ACTION6", data={"x": 3, "y": 4})])
        action = parse_toolkit_recording(path)[0].game_action()
        assert action == GameAction.ACTION6
        assert (action.action_data.x, action.action_data.y) == (3, 4)

    def test_skips_non_action_lines_and_blank_lines(self, tmp_path):
        path = _write_jsonl(
            tmp_path / "r.jsonl",
            [
                {"timestamp": "t", "data": {"scorecard": {}}},
                {"timestamp": "t", "data": {**_event()["data"], "action_input": None}},
                "",
                _event("ACTION2"),
            ],
        )
        steps = parse_toolkit_recording(path)
        assert [(s.index, s.action) for s in steps] == [(0, "ACTION2")]

    @pytest.mark.parametrize("reasoning", [None, "", {}])
    def test_empty_reasoning_normalizes_to_empty_dict(self, tmp_path, reasoning):
        path = _write_jsonl(tmp_path / "r.jsonl", [_event(reasoning=reasoning)])
        assert parse_toolkit_recording(path)[0].reasoning == {}

    def test_string_reasoning_round_trips_byte_identically(self, tmp_path):
        original = {
            "output": "Move to the café ✓",
            "reasoning": "line1\nline2 \"quoted\"",
            "usage": {"input_tokens": 12, "total_tokens": 20},
            "cost": {"input_cost": 0.000123, "total_cost": 1e-07},
            "state": {"native_compaction": {"usage": {"total_tokens": 5}}},
        }
        # The remote client sends exactly this string to the server.
        stored = json.dumps(original)
        path = _write_jsonl(tmp_path / "r.jsonl", [_event(reasoning=stored)])

        reasoning = parse_toolkit_recording(path)[0].reasoning

        assert reasoning == original
        assert json.dumps(reasoning) == stored

    def test_string_reasoning_that_does_not_round_trip_raises(self, tmp_path):
        stored = json.dumps({"output": "go"}, separators=(",", ":"))
        path = _write_jsonl(tmp_path / "r.jsonl", [_event(reasoning=stored)])
        with pytest.raises(RehydrationError, match="round-trip"):
            parse_toolkit_recording(path)

    @pytest.mark.parametrize(
        ("line", "message"),
        [
            ("not json", "invalid JSON"),
            ({"timestamp": "t"}, "missing 'data'"),
            (_event("ACTION99"), "unknown action id"),
            (_event("ACTION6", data={"x": 1}), "integer x and y"),
            (_event("ACTION6", data={"x": "1", "y": 2}), "integer x and y"),
            (_event(reasoning="not json"), "not valid JSON"),
            (_event(reasoning='["list"]'), "decode to an object"),
            (_event(reasoning=42), "unsupported reasoning type"),
            (
                {"data": {k: v for k, v in _event()["data"].items() if k != "state"}},
                "malformed action event",
            ),
            (_event(frame=None), "malformed action event"),
        ],
    )
    def test_malformed_lines_raise_with_location(self, tmp_path, line, message):
        path = _write_jsonl(tmp_path / "r.jsonl", [_event(), line])
        with pytest.raises(RehydrationError, match=message) as excinfo:
            parse_toolkit_recording(path)
        assert f"{path}:2" in str(excinfo.value)


@pytest.mark.unit
class TestAgentSnapshot:
    def test_round_trips_through_json(self, tmp_path):
        runtime_state = RuntimeState(
            adapter_id="openai.responses.v1",
            strategy="continuous_conversation",
            payload={"input_items": [{"type": "reasoning", "encrypted_content": "opaque"}]},
        )
        snapshot = _snapshot(runtime_state=runtime_state)
        path = write_snapshot_atomic(tmp_path, snapshot)

        loaded = load_snapshot(path)

        assert loaded == snapshot

    def test_rejects_step_zero(self):
        with pytest.raises(ValidationError):
            _snapshot(step=0)

    def test_rejects_unsupported_schema_version(self):
        data = _snapshot().model_dump(mode="json")
        data["snapshot_schema_version"] = 999
        with pytest.raises(ValidationError, match="snapshot_schema_version"):
            AgentSnapshot.model_validate(data)

    def test_rejects_unknown_fields(self):
        data = _snapshot().model_dump(mode="json")
        data["agent"]["unexpected"] = 1
        with pytest.raises(ValidationError):
            AgentSnapshot.model_validate(data)

    def test_rejects_invalid_runtime_state(self):
        data = _snapshot().model_dump(mode="json")
        data["runtime_state"] = {
            "schema_version": 999,
            "adapter_id": "x",
            "strategy": "continuous_conversation",
        }
        with pytest.raises(ValidationError):
            AgentSnapshot.model_validate(data)


@pytest.mark.unit
class TestWriteSnapshotAtomic:
    def _names(self, tmp_path: Path) -> list[str]:
        return sorted(p.name for p in (tmp_path / "state").iterdir())

    def test_writes_padded_filename_in_state_dir(self, tmp_path):
        path = write_snapshot_atomic(tmp_path, _snapshot(step=7))
        assert path == tmp_path / "state" / "state_step_0007.json"
        assert self._names(tmp_path) == ["state_step_0007.json"]

    def test_keeps_only_latest_three_and_ignores_other_files(self, tmp_path):
        (tmp_path / "state").mkdir()
        (tmp_path / "state" / "notes.txt").write_text("keep me")
        for step in range(1, 6):
            write_snapshot_atomic(tmp_path, _snapshot(step=step))
        assert self._names(tmp_path) == [
            "notes.txt",
            "state_step_0003.json",
            "state_step_0004.json",
            "state_step_0005.json",
        ]

    def test_prunes_numerically_past_padding_width(self, tmp_path):
        for step in (9998, 9999, 10000, 10001):
            write_snapshot_atomic(tmp_path, _snapshot(step=step))
        assert self._names(tmp_path) == [
            "state_step_10000.json",
            "state_step_10001.json",
            "state_step_9999.json",
        ]

    def test_failed_write_leaves_previous_snapshots_intact(self, tmp_path, monkeypatch):
        write_snapshot_atomic(tmp_path, _snapshot(step=1))
        write_snapshot_atomic(tmp_path, _snapshot(step=2))
        before = {
            p.name: p.read_bytes() for p in (tmp_path / "state").iterdir()
        }

        def crash(*_args):
            raise OSError("disk full")

        monkeypatch.setattr(rehydration.os, "replace", crash)
        with pytest.raises(OSError, match="disk full"):
            write_snapshot_atomic(tmp_path, _snapshot(step=3))

        after = {p.name: p.read_bytes() for p in (tmp_path / "state").iterdir()}
        assert after == before  # no partial file, no temp file, nothing pruned



@pytest.mark.unit
class TestParseRehydrateArgs:
    def test_builds_args_from_pairs(self, tmp_path):
        recording = _write_jsonl(tmp_path / "r.jsonl", [])
        state = _write_jsonl(tmp_path / "s.json", [])
        args = parse_rehydrate_args([f"recording={recording}", f"state={state}"])
        assert (args.recording, args.state) == (recording, state)

    def test_path_may_contain_equals(self, tmp_path):
        recording = _write_jsonl(tmp_path / "a=b.jsonl", [])
        args = parse_rehydrate_args([f"recording={recording}", f"state={recording}"])
        assert args.recording == recording

    @pytest.mark.parametrize("pair", ["recording", "=path", "recording="])
    def test_rejects_malformed_pairs(self, pair):
        with pytest.raises(RehydrationError, match="expected KEY=PATH"):
            parse_rehydrate_args([pair])

    def test_rejects_duplicate_keys(self, tmp_path):
        recording = _write_jsonl(tmp_path / "r.jsonl", [])
        with pytest.raises(RehydrationError, match="Duplicate"):
            parse_rehydrate_args([f"state={recording}", f"state={recording}"])

    @pytest.mark.parametrize(
        "extra", [[], ["bogus=x"], ["state=does-not-exist.json"]]
    )
    def test_rejects_missing_unknown_or_nonexistent_inputs(self, tmp_path, extra):
        recording = _write_jsonl(tmp_path / "r.jsonl", [])
        with pytest.raises(ValidationError):
            parse_rehydrate_args([f"recording={recording}", *extra])


CONFIG_ID = "openai-gpt-5-4-2026-03-05"
GAME_ID = "ls20-abc123"


def _step_event(value: int, action_id: str = "ACTION1", **kwargs) -> dict:
    return _event(action_id, frame=[[[value]]], **kwargs)


def _recording(*values: int) -> list[dict]:
    """The implicit make() RESET followed by one agent step per frame value."""
    return [_step_event(0, "RESET")] + [_step_event(value) for value in values]


def _valid_snapshot(step: int = 3, **updates) -> AgentSnapshot:
    snapshot = _snapshot(step=step)
    return snapshot.model_copy(
        update={
            "model_config_id": CONFIG_ID,
            "model_config_sha256": config_sha256(get_model_config(CONFIG_ID)),
            "last_frame": _fingerprint(
                state="NOT_FINISHED",
                levels_completed=0,
                available_actions=[1, 2, 6],
                frame=[[[step]]],
            ),
            **updates,
        }
    )


@pytest.mark.unit
class TestPrepareRehydration:
    @pytest.fixture(autouse=True)
    def _commit_sha(self, monkeypatch):
        monkeypatch.setenv("ARC_HARNESS_COMMIT_SHA", "abc123")

    def _prepare(
        self,
        tmp_path,
        snapshot: AgentSnapshot,
        events: list,
        game_ids: list[str] | None = None,
        config_id: str = CONFIG_ID,
    ):
        state = tmp_path / "state.json"
        state.write_text(snapshot.model_dump_json())
        recording = _write_jsonl(tmp_path / "r.jsonl", events)
        return prepare_rehydration(
            RehydrationArgs(recording=recording, state=state),
            config_id=config_id,
            game_ids=[GAME_ID] if game_ids is None else game_ids,
        )

    def test_skips_implicit_reset_and_truncates_extra_steps(self, tmp_path):
        prepared = self._prepare(tmp_path, _valid_snapshot(step=3), _recording(1, 2, 3, 4, 5))
        assert [step.index for step in prepared.steps] == [1, 2, 3]
        assert prepared.steps[-1].frame == [[[3]]]

    def test_no_op_step_aligns_after_implicit_reset(self, tmp_path):
        # Step 2 leaves the frame unchanged, so events 1 and 2 have the same frame.
        events = _recording(2) + [_step_event(2, "ACTION2")]
        prepared = self._prepare(tmp_path, _valid_snapshot(step=2), events)
        assert [step.action for step in prepared.steps] == ["ACTION1", "ACTION2"]

    def test_rejects_recording_without_leading_reset(self, tmp_path):
        events = [_step_event(value) for value in (1, 2, 3)]
        with pytest.raises(RehydrationError, match="not the implicit RESET"):
            self._prepare(tmp_path, _valid_snapshot(step=3), events)

    def test_rejects_recording_without_matching_step(self, tmp_path):
        with pytest.raises(RehydrationError, match="no step 3 matching"):
            self._prepare(tmp_path, _valid_snapshot(step=3), _recording(1, 2, 9))

    def test_rejects_recording_shorter_than_snapshot(self, tmp_path):
        with pytest.raises(RehydrationError, match="no step 3 matching"):
            self._prepare(tmp_path, _valid_snapshot(step=3), _recording(1, 2))

    def test_rejects_empty_recording(self, tmp_path):
        with pytest.raises(RehydrationError, match="no action events"):
            self._prepare(tmp_path, _valid_snapshot(), [])

    def test_rejects_events_from_another_session(self, tmp_path):
        events = _recording(1) + [_step_event(2, guid="other"), _step_event(3)]
        with pytest.raises(RehydrationError, match="guid 'other'"):
            self._prepare(tmp_path, _valid_snapshot(step=3), events)

    @pytest.mark.parametrize(
        ("runtime_sha", "snapshot_sha", "message"),
        [
            (None, "abc123", "commit SHA must be known"),
            ("abc123", None, "commit SHA must be known"),
            ("abc123", "other", "commit mismatch"),
        ],
    )
    def test_requires_matching_commit_sha(
        self, tmp_path, monkeypatch, runtime_sha, snapshot_sha, message
    ):
        if runtime_sha is None:
            monkeypatch.delenv("ARC_HARNESS_COMMIT_SHA")
        snapshot = _valid_snapshot(harness_commit_sha=snapshot_sha)
        with pytest.raises(RehydrationError, match=message):
            self._prepare(tmp_path, snapshot, [_step_event(3)])

    def test_rejects_different_config_id(self, tmp_path):
        with pytest.raises(RehydrationError, match="Model config mismatch"):
            self._prepare(
                tmp_path,
                _valid_snapshot(),
                [_step_event(3)],
                config_id="openai-gpt-5.4-openrouter",
            )

    @pytest.mark.parametrize(
        "change",
        [
            lambda entry: {**entry, "pricing": {"input": 99.0, "output": 99.0}},
            lambda entry: {
                **entry,
                "agent": {**entry["agent"], "MAX_ACTIONS_BASELINE_MULTIPLIER": 9.0},
            },
        ],
        ids=["pricing", "budget_multiplier"],
    )
    def test_allows_unhashed_config_changes(self, tmp_path, monkeypatch, change):
        changed = change(get_model_config(CONFIG_ID))
        monkeypatch.setattr(
            "benchmarking.rehydration.get_model_config", lambda _id: changed
        )
        events = _recording(1, 2, 3)
        assert self._prepare(tmp_path, _valid_snapshot(), events).snapshot.step == 3

    def test_rejects_behavioral_config_change(self, tmp_path, monkeypatch):
        entry = get_model_config(CONFIG_ID)
        changed = {**entry, "request": {**entry["request"], "max_completion_tokens": 1}}
        monkeypatch.setattr(
            "benchmarking.rehydration.get_model_config", lambda _id: changed
        )
        with pytest.raises(RehydrationError, match="changed since the snapshot"):
            self._prepare(tmp_path, _valid_snapshot(), [_step_event(3)])

    def test_rejects_server_state_configs(self, tmp_path, monkeypatch):
        entry = get_model_config(CONFIG_ID)
        server = {**entry, "runtime": {**entry["runtime"], "state": "previous_response_id"}}
        monkeypatch.setattr(
            "benchmarking.rehydration.get_model_config", lambda _id: server
        )
        snapshot = _valid_snapshot(model_config_sha256=config_sha256(server))
        with pytest.raises(RehydrationError, match="previous_response_id"):
            self._prepare(tmp_path, snapshot, [_step_event(3)])

    @pytest.mark.parametrize(
        "game_ids", [[], [GAME_ID, "ft09-xyz"], ["ls20-other"]]
    )
    def test_requires_exactly_the_snapshot_game(self, tmp_path, game_ids):
        with pytest.raises(RehydrationError, match=f"Pass -g {GAME_ID}"):
            self._prepare(
                tmp_path, _valid_snapshot(), [_step_event(3)], game_ids=game_ids
            )

    def test_rejects_win_snapshot(self, tmp_path):
        snapshot = _valid_snapshot()
        snapshot = snapshot.model_copy(
            update={"last_frame": snapshot.last_frame.model_copy(update={"state": "WIN"})}
        )
        with pytest.raises(RehydrationError, match="WIN"):
            self._prepare(tmp_path, snapshot, [_step_event(3)])

    def test_rejects_snapshot_without_time_budget(self, tmp_path, monkeypatch):
        entry = get_model_config(CONFIG_ID)
        limited = {**entry, "agent": {**entry["agent"], "MAX_RUNTIME_SECONDS": 10}}
        monkeypatch.setattr(
            "benchmarking.rehydration.get_model_config", lambda _id: limited
        )
        snapshot = _valid_snapshot(model_config_sha256=config_sha256(limited))
        assert snapshot.agent.elapsed_seconds >= 10
        with pytest.raises(RehydrationError, match="no time budget left"):
            self._prepare(tmp_path, snapshot, [_step_event(3)])

