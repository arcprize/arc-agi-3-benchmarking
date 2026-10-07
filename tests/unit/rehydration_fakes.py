"""Fakes and helpers shared by the rehydration and snapshot tests."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from arcengine import ActionInput, FrameDataRaw, GameAction, GameState

from benchmarking.agent import BenchmarkingAgent
from benchmarking.rehydration import AgentSnapshot, load_snapshot


def raw_frame(
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


def recording_event(
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


class ScriptedEnv:
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
        self.observation_space = raw_frame(initial_state, 0, GameAction.RESET, guid)
        self._make_event = recording_event(
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
        raw = raw_frame(state, value, action, self.guid, levels)
        self.events.append(recording_event(raw, action, data, reasoning))
        self.observation_space = raw
        return raw

    def write_recording(self, path: Path) -> Path:
        """Write the toolkit recording: the implicit make() RESET, then each step."""
        events = [self._make_event, *self.events]
        path.write_text("".join(json.dumps(event) + "\n" for event in events))
        return path


def build_agent(
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


def snapshot_names(agent: BenchmarkingAgent) -> list[str]:
    return sorted(p.name for p in (Path(agent.run_dir) / "state").iterdir())


def latest_snapshot(agent: BenchmarkingAgent) -> AgentSnapshot:
    return load_snapshot(Path(agent.run_dir) / "state" / snapshot_names(agent)[-1])


def comparable_snapshot(snapshot: AgentSnapshot) -> dict:
    """Snapshot content that must match an uninterrupted run."""
    data = snapshot.model_dump(
        mode="json", exclude={"created_at", "source", "lineage"}
    )
    data["agent"].pop("elapsed_seconds")
    return data
