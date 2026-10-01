"""Agent state snapshots and toolkit recording parsing for rehydration.

See REHYDRATION_PLAN.md. This module is self-contained: it defines the
snapshot schema, parses toolkit ``.jsonl`` recordings, fingerprints frames, and
writes rolling snapshots atomically. Wiring into the agent happens elsewhere.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from arcengine import GameAction, GameState
from pydantic import BaseModel, ConfigDict, Field, FilePath, model_validator

from .base import Agent
from .model_config import get_model_config
from .recording import StepUsage
from .runtime_state import SERVER_RUNTIME_STATE, RuntimeState, harness_commit_sha

SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_DIR = "state"
SNAPSHOT_KEEP = 3
_SNAPSHOT_FILENAME = re.compile(r"^state_step_(\d+)\.json$")


class RehydrationError(ValueError):
    """Raised when rehydration inputs are malformed or incompatible."""


# ── Inputs ──────────────────────────────────────────────────────────────


class RehydrationArgs(BaseModel):
    """Local files needed to rehydrate. Add new inputs as optional fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    recording: FilePath
    state: FilePath


# ── Snapshot schema ─────────────────────────────────────────────────────


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FrameFingerprint(_Strict):
    """Session-independent identity of one frame (no guid or timestamps)."""

    state: str
    levels_completed: int
    available_actions: list[int]
    frame_sha256: str | None = None


class SnapshotSource(_Strict):
    run_id: str
    guid: str
    game_id: str
    card_id: str


class LineageEntry(_Strict):
    run_id: str
    guid: str
    rehydrated_at_step: int


class PreviousAction(_Strict):
    name: str
    data: dict[str, Any] = Field(default_factory=dict)


class AgentFields(_Strict):
    """Agent attributes outside RuntimeState that affect future behavior."""

    conversation: list[dict[str, Any]]
    token_counter: int
    level_action_counter: int
    last_levels_completed: int
    level_just_advanced: bool
    compaction_counter: int
    pending_compaction_trigger_tokens: int | None
    previous_action: PreviousAction | None
    previous_response_id: str | None
    pending_user_messages: list[dict[str, Any]]
    elapsed_seconds: float
    total_usage: StepUsage


class AgentSnapshot(_Strict):
    """Serialized agent state at a clean loop boundary after ``step`` actions."""

    snapshot_schema_version: int = SNAPSHOT_SCHEMA_VERSION
    harness_commit_sha: str | None
    created_at: datetime
    source: SnapshotSource
    lineage: list[LineageEntry] = Field(default_factory=list)
    model_config_id: str
    model_config_sha256: str
    pricing: dict[str, float] = Field(default_factory=dict)
    level_action_budgets: list[int]
    step: int = Field(ge=0)
    last_frame: FrameFingerprint
    agent: AgentFields
    runtime_state: RuntimeState | None = None

    @model_validator(mode="after")
    def validate_schema_version(self) -> AgentSnapshot:
        if self.snapshot_schema_version != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported snapshot_schema_version={self.snapshot_schema_version}; "
                f"expected {SNAPSHOT_SCHEMA_VERSION}."
            )
        return self


def fingerprints_match(live: FrameFingerprint, recorded: FrameFingerprint) -> bool:
    """Compare frames; ignore the grid hash when the recording has no frame data."""
    if recorded.frame_sha256 is None:
        live = live.model_copy(update={"frame_sha256": None})
    return live == recorded


def config_sha256(entry: dict[str, Any]) -> str:
    """Hash a raw model config entry. Pricing is excluded by design."""
    hashed = {key: value for key, value in entry.items() if key != "pricing"}
    canonical = json.dumps(hashed, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def frame_fingerprint(
    *,
    state: GameState | str,
    levels_completed: int,
    available_actions: list[int],
    frame: list[Any] | None,
) -> FrameFingerprint:
    frame_sha256 = None
    if frame is not None:
        canonical = json.dumps(frame, separators=(",", ":"))
        frame_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return FrameFingerprint(
        state=state.name if isinstance(state, GameState) else state,
        levels_completed=levels_completed,
        available_actions=list(available_actions),
        frame_sha256=frame_sha256,
    )


# ── Snapshot persistence ────────────────────────────────────────────────


def load_snapshot(path: str | os.PathLike[str]) -> AgentSnapshot:
    return AgentSnapshot.model_validate_json(Path(path).read_text(encoding="utf-8"))


def write_snapshot_atomic(
    run_dir: str | os.PathLike[str],
    snapshot: AgentSnapshot,
    keep: int = SNAPSHOT_KEEP,
) -> Path:
    """Write ``state/state_step_NNNN.json`` atomically, then prune old snapshots.

    The file is written to a temp file in the same directory, fsynced, and
    renamed into place, so a crash never leaves a partial snapshot under the
    final name. Older snapshots are pruned only after the new one is in place.
    """
    if keep < 1:
        raise ValueError("keep must be at least 1")
    state_dir = Path(run_dir) / SNAPSHOT_DIR
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / f"state_step_{snapshot.step:04d}.json"

    fd, tmp_name = tempfile.mkstemp(dir=state_dir, prefix=".state_step_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(snapshot.model_dump_json())
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise

    snapshots = sorted(
        (int(match.group(1)), entry)
        for entry in state_dir.iterdir()
        if (match := _SNAPSHOT_FILENAME.match(entry.name))
    )
    for _, stale in snapshots[:-keep]:
        stale.unlink(missing_ok=True)
    return path


# ── Toolkit recording parsing ───────────────────────────────────────────


class RecordedStep(BaseModel):
    """One action event from a toolkit recording, ready to replay.

    ``index`` is the 0-based position among action events in the file, not the
    agent step number; ``_align_recording`` maps events to agent steps.
    """

    index: int
    action: str
    data: dict[str, int] = Field(default_factory=dict)
    reasoning: dict[str, Any] = Field(default_factory=dict)
    guid: str | None
    game_id: str
    state: str
    levels_completed: int
    available_actions: list[int]
    frame: list[Any] | None = None

    def game_action(self) -> GameAction:
        action = GameAction.from_name(self.action)
        if action.is_complex():
            action.set_data(dict(self.data))
        return action

    def fingerprint(self) -> FrameFingerprint:
        return frame_fingerprint(
            state=self.state,
            levels_completed=self.levels_completed,
            available_actions=self.available_actions,
            frame=self.frame,
        )


class PreparedRehydration(BaseModel):
    """Inputs ready for replay: ``steps[i]`` is agent step ``i + 1``."""

    snapshot: AgentSnapshot
    steps: list[RecordedStep]

    @model_validator(mode="after")
    def validate_alignment(self) -> PreparedRehydration:
        if self.snapshot.step < 1:
            raise ValueError("Cannot rehydrate from a snapshot before step 1.")
        if len(self.steps) != self.snapshot.step:
            raise ValueError(
                f"Expected {self.snapshot.step} recorded steps, got {len(self.steps)}."
            )
        return self


def _normalize_reasoning(value: Any, where: str) -> dict[str, Any]:
    """Return reasoning as the dict originally passed to ``arc_env.step``.

    The remote client sends ``json.dumps(reasoning)``, so the server may store a
    string. Re-sending must reproduce that string exactly (plan F27).
    """
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise RehydrationError(f"{where}: reasoning is not valid JSON.") from exc
        if not isinstance(parsed, dict):
            raise RehydrationError(f"{where}: reasoning must decode to an object.")
        if json.dumps(parsed) != value:
            raise RehydrationError(
                f"{where}: reasoning does not round-trip byte-identically."
            )
        return parsed
    raise RehydrationError(
        f"{where}: unsupported reasoning type {type(value).__name__}."
    )


def _parse_action_event(data: dict[str, Any], index: int, where: str) -> RecordedStep:
    action_input = data["action_input"]
    name = action_input.get("id")
    try:
        action = GameAction.from_name(name)
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        raise RehydrationError(f"{where}: unknown action id {name!r}.") from exc

    action_data: dict[str, int] = {}
    if action.is_complex():
        raw_data = action_input.get("data") or {}
        x, y = raw_data.get("x"), raw_data.get("y")
        if type(x) is not int or type(y) is not int:
            raise RehydrationError(f"{where}: {name} requires integer x and y.")
        action_data = {"x": x, "y": y}

    reasoning = _normalize_reasoning(action_input.get("reasoning"), where)
    try:
        return RecordedStep(
            index=index,
            action=action.name,
            data=action_data,
            reasoning=reasoning,
            guid=data.get("guid"),
            game_id=data["game_id"],
            state=data["state"],
            levels_completed=data["levels_completed"],
            available_actions=data["available_actions"],
            frame=data.get("frame"),
        )
    except (KeyError, ValueError) as exc:
        raise RehydrationError(f"{where}: malformed action event ({exc}).") from exc


def parse_toolkit_recording(path: str | os.PathLike[str]) -> list[RecordedStep]:
    """Parse every action event from a toolkit ``.jsonl`` recording, in order.

    Lines whose ``data`` has no ``action_input`` object are skipped as
    non-action events. Anything else malformed raises ``RehydrationError``.
    """
    steps: list[RecordedStep] = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            where = f"{path}:{line_number}"
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RehydrationError(f"{where}: invalid JSON.") from exc
            data = event.get("data") if isinstance(event, dict) else None
            if not isinstance(data, dict):
                raise RehydrationError(f"{where}: missing 'data' object.")
            if not isinstance(data.get("action_input"), dict):
                continue
            steps.append(_parse_action_event(data, len(steps), where))
    return steps


# ── Offline validation (before any scorecard or network call) ───────────


def parse_rehydrate_args(pairs: list[str]) -> RehydrationArgs:
    """Build RehydrationArgs from repeated ``--rehydrate KEY=PATH`` values."""
    values: dict[str, str] = {}
    for pair in pairs:
        key, separator, value = pair.partition("=")
        if not separator or not key or not value:
            raise RehydrationError(
                f"Invalid --rehydrate value {pair!r}; expected KEY=PATH."
            )
        if key in values:
            raise RehydrationError(f"Duplicate --rehydrate key {key!r}.")
        values[key] = value
    return RehydrationArgs.model_validate(values)


def _validate_snapshot(
    snapshot: AgentSnapshot, *, config_id: str, game_ids: list[str]
) -> None:
    current_sha = harness_commit_sha()
    if not current_sha or not snapshot.harness_commit_sha:
        raise RehydrationError(
            "Harness commit SHA must be known for both the snapshot and this "
            "runtime (set ARC_HARNESS_COMMIT_SHA)."
        )
    if snapshot.harness_commit_sha != current_sha:
        raise RehydrationError(
            f"Harness commit mismatch: snapshot={snapshot.harness_commit_sha}, "
            f"runtime={current_sha}."
        )
    if snapshot.model_config_id != config_id:
        raise RehydrationError(
            f"Model config mismatch: snapshot={snapshot.model_config_id!r}, "
            f"selected={config_id!r}."
        )
    entry = get_model_config(config_id)
    if config_sha256(entry) != snapshot.model_config_sha256:
        raise RehydrationError(
            f"Model config {config_id!r} changed since the snapshot was taken "
            "(only pricing may differ)."
        )
    if entry.get("runtime", {}).get("state") == SERVER_RUNTIME_STATE:
        raise RehydrationError(
            f"Rehydration does not support runtime.state={SERVER_RUNTIME_STATE!r}."
        )
    if game_ids != [snapshot.source.game_id]:
        raise RehydrationError(
            f"Rehydration requires exactly game {snapshot.source.game_id!r}; "
            f"resolved {game_ids}. Pass -g {snapshot.source.game_id}."
        )
    if snapshot.last_frame.state == GameState.WIN.name:
        raise RehydrationError("Snapshot is at a WIN frame; nothing to resume.")
    max_runtime = entry.get("agent", {}).get(
        "MAX_RUNTIME_SECONDS", Agent.MAX_RUNTIME_SECONDS
    )
    if snapshot.agent.elapsed_seconds >= max_runtime:
        raise RehydrationError(
            f"Snapshot has no time budget left: elapsed="
            f"{snapshot.agent.elapsed_seconds}s, limit={max_runtime}s."
        )


def _replay_key(steps: list[RecordedStep]) -> list[tuple[Any, ...]]:
    return [
        (step.action, step.data, step.reasoning, step.fingerprint()) for step in steps
    ]


def _align_recording(
    events: list[RecordedStep], snapshot: AgentSnapshot
) -> list[RecordedStep]:
    """Select the recorded events for agent steps ``1..snapshot.step``.

    The toolkit server may or may not record the implicit reset from
    ``Arcade.make()`` before the agent's first action (plan F3), so both
    alignments are tried. The step-N frame must match the snapshot. If both
    match, they must replay identically.
    """
    candidates: list[list[RecordedStep]] = []
    for offset in (0, 1):
        if offset and events[0].action != GameAction.RESET.name:
            continue
        steps = events[offset : offset + snapshot.step]
        if len(steps) < snapshot.step:
            continue
        if fingerprints_match(snapshot.last_frame, steps[-1].fingerprint()):
            candidates.append(steps)
    if not candidates:
        raise RehydrationError(
            f"Recording ({len(events)} action events) has no step "
            f"{snapshot.step} matching the snapshot's last frame."
        )
    if len(candidates) == 2 and _replay_key(candidates[0]) != _replay_key(
        candidates[1]
    ):
        raise RehydrationError(
            "Recording alignment is ambiguous: steps match the snapshot with and "
            "without a leading implicit reset."
        )
    return candidates[0]


def prepare_rehydration(
    args: RehydrationArgs, *, config_id: str, game_ids: list[str]
) -> PreparedRehydration:
    """Load and cross-check rehydration inputs without touching the network."""
    snapshot = load_snapshot(args.state)
    _validate_snapshot(snapshot, config_id=config_id, game_ids=game_ids)

    events = parse_toolkit_recording(args.recording)
    if not events:
        raise RehydrationError("Recording contains no action events.")
    for event in events:
        if event.game_id != snapshot.source.game_id:
            raise RehydrationError(
                f"Recording event {event.index} is for game {event.game_id!r}, "
                f"not {snapshot.source.game_id!r}."
            )
        if event.guid != snapshot.source.guid:
            raise RehydrationError(
                f"Recording event {event.index} has guid {event.guid!r}; the "
                f"snapshot came from session {snapshot.source.guid!r}."
            )
    return PreparedRehydration(snapshot=snapshot, steps=_align_recording(events, snapshot))
