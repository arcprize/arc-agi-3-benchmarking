import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
from arcengine import GameAction, GameState
from pydantic import ValidationError

from benchmarking import rehydration
from benchmarking.rehydration import (
    AgentFields,
    AgentSnapshot,
    FrameFingerprint,
    PreparedRehydration,
    RehydrationArgs,
    RehydrationError,
    SnapshotSource,
    config_sha256,
    fingerprints_match,
    frame_fingerprint,
    load_snapshot,
    parse_toolkit_recording,
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
) -> dict:
    event_data = {
        "game_id": "ls20-abc123",
        "state": state,
        "levels_completed": 0,
        "win_levels": 3,
        "action_input": {"id": action_id, "data": data or {}, "reasoning": reasoning},
        "guid": "guid-1",
        "full_reset": False,
        "available_actions": [1, 2, 6],
    }
    if frame is not None:
        event_data["frame"] = frame
    return {"timestamp": "2026-10-01T00:00:00+00:00", "data": event_data}


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
        level_action_budgets=[10, 20],
        step=step,
        last_frame=frame_fingerprint(
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
            previous_action={"name": "ACTION6", "data": {"x": 1, "y": 2}},
            previous_response_id=None,
            pending_user_messages=[],
            elapsed_seconds=12.5,
            total_usage={"prompt_tokens": 60, "completion_tokens": 40, "total_tokens": 100},
        ),
        runtime_state=runtime_state,
    )


@pytest.mark.unit
class TestRehydrationArgs:
    def test_requires_existing_files(self, tmp_path):
        recording = _write_jsonl(tmp_path / "r.jsonl", [])
        state = tmp_path / "s.json"
        state.write_text("{}")
        args = RehydrationArgs(recording=recording, state=state)
        assert args.recording == recording

        with pytest.raises(ValidationError):
            RehydrationArgs(recording=recording, state=tmp_path / "missing.json")

    def test_rejects_unknown_keys(self, tmp_path):
        recording = _write_jsonl(tmp_path / "r.jsonl", [])
        with pytest.raises(ValidationError):
            RehydrationArgs(recording=recording, state=recording, extra=recording)


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
        assert config_sha256({k: v for k, v in self.BASE.items() if k != "pricing"}) == (
            config_sha256(self.BASE)
        )

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
        assert frame_fingerprint(state=GameState.WIN, **kwargs) == frame_fingerprint(
            state="WIN", **kwargs
        )

    def test_frame_change_changes_hash(self):
        kwargs = {"state": "NOT_FINISHED", "levels_completed": 0, "available_actions": [1]}
        assert (
            frame_fingerprint(frame=FRAME, **kwargs).frame_sha256
            != frame_fingerprint(frame=[[[0, 1], [2, 4]]], **kwargs).frame_sha256
        )

    def test_missing_frame_has_no_hash(self):
        fingerprint = frame_fingerprint(
            state="NOT_FINISHED", levels_completed=0, available_actions=[], frame=None
        )
        assert fingerprint.frame_sha256 is None

    def test_match_compares_grid_when_recorded(self):
        kwargs = {"state": "NOT_FINISHED", "levels_completed": 0, "available_actions": [1]}
        live = frame_fingerprint(frame=FRAME, **kwargs)
        assert fingerprints_match(live, frame_fingerprint(frame=FRAME, **kwargs))
        assert not fingerprints_match(
            live, frame_fingerprint(frame=[[[9]]], **kwargs)
        )

    def test_match_falls_back_without_recorded_grid(self):
        kwargs = {"levels_completed": 0, "available_actions": [1]}
        live = frame_fingerprint(state="NOT_FINISHED", frame=FRAME, **kwargs)
        assert fingerprints_match(
            live, frame_fingerprint(state="NOT_FINISHED", frame=None, **kwargs)
        )
        assert not fingerprints_match(
            live, frame_fingerprint(state="GAME_OVER", frame=None, **kwargs)
        )


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
        assert steps[0].game_id == "ls20-abc123"

    def test_game_action_carries_coordinates(self, tmp_path):
        path = _write_jsonl(tmp_path / "r.jsonl", [_event("ACTION6", data={"x": 3, "y": 4})])
        action = parse_toolkit_recording(path)[0].game_action()
        assert action == GameAction.ACTION6
        assert (action.action_data.x, action.action_data.y) == (3, 4)

    def test_fingerprint_matches_live_frame_fingerprint(self, tmp_path):
        path = _write_jsonl(tmp_path / "r.jsonl", [_event()])
        recorded = parse_toolkit_recording(path)[0].fingerprint()
        live = frame_fingerprint(
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            available_actions=[1, 2, 6],
            frame=FRAME,
        )
        assert recorded == live

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

    def test_missing_frame_is_allowed(self, tmp_path):
        path = _write_jsonl(tmp_path / "r.jsonl", [_event(frame=None)])
        assert parse_toolkit_recording(path)[0].frame is None

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
        # Opaque provider state is intentionally preserved in snapshots.
        assert loaded.runtime_state.payload["input_items"][0]["encrypted_content"] == "opaque"

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

    def test_last_frame_is_a_fingerprint(self):
        assert isinstance(_snapshot().last_frame, FrameFingerprint)


@pytest.mark.unit
class TestPreparedRehydration:
    def _steps(self, tmp_path: Path, count: int):
        path = _write_jsonl(tmp_path / "r.jsonl", [_event() for _ in range(count)])
        return parse_toolkit_recording(path)

    def test_accepts_one_recorded_step_per_snapshot_step(self, tmp_path):
        prepared = PreparedRehydration(
            snapshot=_snapshot(step=3), steps=self._steps(tmp_path, 3)
        )
        assert len(prepared.steps) == 3

    @pytest.mark.parametrize("count", [2, 4])
    def test_rejects_misaligned_step_count(self, tmp_path, count):
        with pytest.raises(ValidationError, match="Expected 3 recorded steps"):
            PreparedRehydration(
                snapshot=_snapshot(step=3), steps=self._steps(tmp_path, count)
            )

    def test_rejects_step_zero(self):
        with pytest.raises(ValidationError, match="before step 1"):
            PreparedRehydration(snapshot=_snapshot(step=0), steps=[])


@pytest.mark.unit
class TestWriteSnapshotAtomic:
    def _names(self, tmp_path: Path) -> list[str]:
        return sorted(p.name for p in (tmp_path / "state").iterdir())

    def test_writes_padded_filename_in_state_dir(self, tmp_path):
        path = write_snapshot_atomic(tmp_path, _snapshot(step=7))
        assert path == tmp_path / "state" / "state_step_0007.json"
        assert self._names(tmp_path) == ["state_step_0007.json"]

    def test_keeps_only_latest_three(self, tmp_path):
        for step in range(1, 6):
            write_snapshot_atomic(tmp_path, _snapshot(step=step))
        assert self._names(tmp_path) == [
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

    def test_ignores_unrelated_files_when_pruning(self, tmp_path):
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        (state_dir / "notes.txt").write_text("keep me")
        for step in range(1, 5):
            write_snapshot_atomic(tmp_path, _snapshot(step=step))
        assert "notes.txt" in self._names(tmp_path)

    def test_rejects_keep_below_one(self, tmp_path):
        with pytest.raises(ValueError):
            write_snapshot_atomic(tmp_path, _snapshot(), keep=0)

    def test_snapshot_file_is_fsynced_before_rename(self, tmp_path, monkeypatch):
        calls: list[str] = []
        real_fsync, real_replace = os.fsync, os.replace
        monkeypatch.setattr(
            rehydration.os, "fsync", lambda fd: (calls.append("fsync"), real_fsync(fd))
        )
        monkeypatch.setattr(
            rehydration.os,
            "replace",
            lambda src, dst: (calls.append("replace"), real_replace(src, dst)),
        )
        write_snapshot_atomic(tmp_path, _snapshot())
        assert calls == ["fsync", "replace"]
