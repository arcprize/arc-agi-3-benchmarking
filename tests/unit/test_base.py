import numpy as np
import pytest
from arcengine import ActionInput, FrameData, FrameDataRaw, GameAction, GameState

from benchmarking.base import Agent


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
