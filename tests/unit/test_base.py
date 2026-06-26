import numpy as np
import pytest
from arcengine import ActionInput, FrameData, FrameDataRaw, GameAction, GameState

from benchmarking.base import Agent, ExitReason


class _FakeEnv:
    def __init__(self, observation_space: FrameData, step_frame: FrameData) -> None:
        self.observation_space = observation_space
        self.step_frame = step_frame
        self.actions: list[GameAction] = []
        self.reasonings: list[dict] = []

    def step(
        self,
        action: GameAction,
        data: dict,
        reasoning: dict,
    ) -> FrameData:
        self.actions.append(action)
        self.reasonings.append(reasoning)
        return self.step_frame


class _FailingStepEnv(_FakeEnv):
    """Env whose step() (the ARC API boundary) raises, as on an API error."""

    def step(self, action: GameAction, data: dict, reasoning: dict) -> FrameData:
        raise RuntimeError("api issue")


class _TestAgent(Agent):
    MAX_ACTIONS = 0

    def __init__(self, arc_env: _FakeEnv) -> None:
        super().__init__(
            card_id="card-1",
            game_id="game-1",
            agent_name="test-agent",
            ROOT_URL="https://example.com",
            record=False,
            arc_env=arc_env,
        )
        self.choose_action_calls: list[FrameData] = []
        self.forced_observations: list[tuple[GameState, GameAction]] = []

    def _convert_raw_frame_data(self, raw):  # noqa: ANN001
        return raw

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return False

    def choose_action(
        self,
        frames: list[FrameData],
        latest_frame: FrameData,
    ) -> GameAction:
        self.choose_action_calls.append(latest_frame)
        return GameAction.ACTION1

    def _record_forced_action_observation(
        self,
        frames: list[FrameData],
        latest_frame: FrameData,
        forced_action: GameAction,
    ) -> None:
        self.forced_observations.append((latest_frame.state, forced_action))


def _frame(state: GameState) -> FrameData:
    return FrameData(
        frame=[[[0]]],
        state=state,
        levels_completed=0,
        available_actions=[GameAction.ACTION1.value],
    )


@pytest.mark.unit
class TestAgentForcedReset:
    @pytest.mark.parametrize(
        "state",
        [
            GameState.GAME_OVER,
            GameState.NOT_PLAYED,
        ],
    )
    def test_main_forces_reset_for_terminal_states_before_choose_action(self, state):
        arc_env = _FakeEnv(
            observation_space=_frame(state),
            step_frame=_frame(GameState.NOT_FINISHED),
        )
        agent = _TestAgent(arc_env)

        agent.main()

        assert agent.choose_action_calls == []
        assert agent.forced_observations == [(state, GameAction.RESET)]
        assert arc_env.actions == [GameAction.RESET]
        assert len(agent.frames) == 2
        assert agent.frames[-1].state == GameState.NOT_FINISHED

    def test_main_uses_choose_action_for_non_terminal_state(self):
        arc_env = _FakeEnv(
            observation_space=_frame(GameState.NOT_FINISHED),
            step_frame=_frame(GameState.NOT_FINISHED),
        )
        agent = _TestAgent(arc_env)

        agent.main()

        assert [frame.state for frame in agent.choose_action_calls] == [
            GameState.NOT_FINISHED,
        ]
        assert agent.forced_observations == []
        assert arc_env.actions == [GameAction.ACTION1]

    def test_main_stops_when_runtime_budget_exceeded(self, monkeypatch):
        arc_env = _FakeEnv(
            observation_space=_frame(GameState.NOT_FINISHED),
            step_frame=_frame(GameState.NOT_FINISHED),
        )
        agent = _TestAgent(arc_env)
        # Allow several actions so the runtime budget is the limiting factor.
        agent.MAX_ACTIONS = 100
        agent.MAX_RUNTIME_SECONDS = 5

        # First call sets the timer (start); the next call (first loop check)
        # already exceeds the budget, so the loop exits before any action.
        times = iter([1000.0, 2000.0])
        monkeypatch.setattr(
            "benchmarking.base.time.time",
            lambda: next(times, 2000.0),
        )

        agent.main()

        assert agent._timed_out is True
        assert agent.choose_action_calls == []
        assert arc_env.actions == []

    def test_main_does_not_time_out_within_budget(self, monkeypatch):
        arc_env = _FakeEnv(
            observation_space=_frame(GameState.NOT_FINISHED),
            step_frame=_frame(GameState.NOT_FINISHED),
        )
        agent = _TestAgent(arc_env)
        agent.MAX_RUNTIME_SECONDS = 10_000

        monkeypatch.setattr("benchmarking.base.time.time", lambda: 1000.0)

        agent.main()

        assert agent._timed_out is False
        assert arc_env.actions == [GameAction.ACTION1]

    def test_default_runtime_budget_is_twelve_hours(self):
        assert Agent.MAX_RUNTIME_SECONDS == 12 * 60 * 60

    def test_convert_raw_frame_data_preserves_action_input_reasoning(self):
        arc_env = _FakeEnv(
            observation_space=_frame(GameState.NOT_FINISHED),
            step_frame=_frame(GameState.NOT_FINISHED),
        )
        agent = _TestAgent(arc_env)
        raw = FrameDataRaw()
        raw.game_id = "game-1"
        raw.frame = [np.array([[0, 1]], dtype=np.int8)]
        raw.state = GameState.NOT_FINISHED
        raw.levels_completed = 0
        raw.win_levels = 1
        raw.action_input = ActionInput(
            id=GameAction.ACTION1,
            data={"x": 1},
            reasoning={"usage": {"total_tokens": 5}},
        )
        raw.guid = "guid-1"
        raw.full_reset = False
        raw.available_actions = [GameAction.ACTION1.value]

        frame = agent._convert_raw_frame_data(raw)

        assert frame.action_input.id is GameAction.ACTION1
        assert frame.action_input.data == {"x": 1}
        assert frame.action_input.reasoning == {"usage": {"total_tokens": 5}}


class _BudgetAgent(_TestAgent):
    """Never reports done, so main() runs until MAX_ACTIONS is exhausted."""

    MAX_ACTIONS = 2


class _ResolveFailsAgent(_TestAgent):
    """Agent whose action selection (not the API call) raises."""

    MAX_ACTIONS = 5

    def choose_action(
        self,
        frames: list[FrameData],
        latest_frame: FrameData,
    ) -> GameAction:
        raise RuntimeError("model issue")


@pytest.mark.unit
class TestAgentExitReason:
    def test_main_sets_action_budget_when_max_actions_reached(self):
        arc_env = _FakeEnv(
            observation_space=_frame(GameState.NOT_FINISHED),
            step_frame=_frame(GameState.NOT_FINISHED),
        )
        agent = _BudgetAgent(arc_env)

        agent.main()

        assert agent.action_counter >= agent.MAX_ACTIONS
        assert agent.exit_reason is ExitReason.ACTION_BUDGET

    def test_main_sets_agent_error_when_resolve_action_raises(self):
        arc_env = _FakeEnv(
            observation_space=_frame(GameState.NOT_FINISHED),
            step_frame=_frame(GameState.NOT_FINISHED),
        )
        agent = _ResolveFailsAgent(arc_env)

        with pytest.raises(RuntimeError, match="model issue"):
            agent.main()

        # Failure happened before the API call, so it is the agent's fault.
        assert agent.exit_reason is ExitReason.AGENT_ERROR
        assert arc_env.actions == []

    def test_main_sets_api_error_when_take_action_raises(self):
        arc_env = _FailingStepEnv(
            observation_space=_frame(GameState.NOT_FINISHED),
            step_frame=_frame(GameState.NOT_FINISHED),
        )
        agent = _TestAgent(arc_env)

        with pytest.raises(RuntimeError, match="api issue"):
            agent.main()

        # Failure happened while submitting the action to the ARC API.
        assert agent.exit_reason is ExitReason.API_ERROR
