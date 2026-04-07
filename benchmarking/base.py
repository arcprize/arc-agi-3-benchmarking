import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Optional

from arc_agi import EnvironmentWrapper
from arc_agi.scorecard import EnvironmentScorecard
from arcengine import FrameData, FrameDataRaw, GameAction, GameState
from pydantic import ValidationError

from .recorder import Recorder

logger = logging.getLogger()


class Agent(ABC):
    """Interface for an agent that plays one ARC-AGI-3 game."""

    MAX_ACTIONS: int = 80  # to avoid looping forever if agent doesnt exit
    ROOT_URL: str

    action_counter: int = 0

    timer: float = 0
    agent_name: str
    card_id: str
    game_id: str
    guid: str
    frames: list[FrameData]

    recorder: Recorder
    headers: dict[str, str]
    arc_env: EnvironmentWrapper
    _previous_action: Optional[GameAction]

    def __init__(
        self,
        card_id: str,
        game_id: str,
        agent_name: str,
        ROOT_URL: str,
        record: bool,
        arc_env: EnvironmentWrapper,
        config: Optional[str] = None,
    ) -> None:
        self.ROOT_URL = ROOT_URL
        self.card_id = card_id
        self.game_id = game_id
        self.guid = ""
        self.agent_name = agent_name
        self.config = config
        self.frames = [FrameData(levels_completed=0)]
        self._cleanup = True
        if record:
            self.start_recording()
        self.headers = {
            "X-API-Key": os.getenv("ARC_API_KEY", ""),
            "Accept": "application/json",
        }
        self.arc_env = arc_env
        self._previous_action = None

    def main(self) -> None:
        """The main agent loop. Play the game_id until finished, then exits."""
        self.timer = time.time()
        while (
            not self.is_done(self.frames, self.frames[-1])
            and self.action_counter <= self.MAX_ACTIONS
        ):
            action = self.choose_action(
                self.frames,
                self._convert_raw_frame_data(
                    self.arc_env.observation_space if self.arc_env else None
                ),
            )
            if frame := self.take_action(action):
                self.append_frame(frame)
                logger.info(
                    f"{self.game_id} - {action.name}: count {self.action_counter}, levels completed {frame.levels_completed}, avg fps {self.fps})"
                )
            self.action_counter += 1

        self.cleanup()

    @property
    def state(self) -> GameState:
        return self.frames[-1].state

    @property
    def levels_completed(self) -> int:
        return self.frames[-1].levels_completed  # type: ignore[no-any-return]

    @property
    def seconds(self) -> float:
        return (time.time() - self.timer) * 100 // 1 / 100

    @property
    def fps(self) -> float:
        if self.action_counter == 0:
            return 0.0
        elapsed_time = max(self.seconds, 0.1)
        return round(self.action_counter / elapsed_time, 2)

    @property
    def name(self) -> str:
        n = self.__class__.__name__.lower()
        return f"{self.game_id}.{n}"

    def start_recording(self) -> None:
        self.recorder = Recorder(prefix=self.name)
        logger.info(
            f"created new recording for {self.name} into {self.recorder.filename}"
        )

    def append_frame(self, frame: FrameData) -> None:
        self.frames.append(frame)
        if frame.guid:
            self.guid = frame.guid
        if hasattr(self, "recorder"):
            self.recorder.record(json.loads(frame.model_dump_json()))

    def is_reset_a_valid_action(self) -> bool:
        return self.action_counter > 0 and self._previous_action != GameAction.RESET

    def do_action_request(self, action: GameAction) -> FrameData:
        data = action.action_data.model_dump()

        if self.is_reset_a_valid_action():
            raw = self.arc_env.step(
                action,
                data=data,
                reasoning=data["reasoning"] if "reasoning" in data else {},
            )
        else:
            return self.frames[-1]

        self._previous_action = action

        return self._convert_raw_frame_data(raw)

    def _convert_raw_frame_data(self, raw: FrameDataRaw | None) -> FrameData:
        if raw is None:
            raise ValueError("Received None frame data from environment")
        out = FrameData(
            game_id=raw.game_id,
            frame=[arr.tolist() for arr in raw.frame],
            state=raw.state,
            levels_completed=raw.levels_completed,
            win_levels=raw.win_levels,
            guid=raw.guid,
            full_reset=raw.full_reset,
            available_actions=raw.available_actions,
        )
        return out

    def take_action(self, action: GameAction) -> Optional[FrameData]:
        """Submits the specific action and gets the next frame."""
        frame_data = self.do_action_request(action)
        try:
            frame = FrameData.model_validate(frame_data)
        except ValidationError as e:
            logger.warning(f"Incoming frame data did not validate: {e}")
            return None
        return frame

    def cleanup(self, scorecard: Optional[EnvironmentScorecard] = None) -> None:
        """Called after main loop is finished."""
        if self._cleanup:
            self._cleanup = False  # only cleanup once per agent
            if hasattr(self, "recorder"):
                if scorecard:
                    self.recorder.record(scorecard.get(self.game_id))
                logger.info(
                    f"recording for {self.name} is available in {self.recorder.filename}"
                )
            if self.action_counter >= self.MAX_ACTIONS:
                logger.info(
                    f"Exiting: agent reached MAX_ACTIONS of {self.MAX_ACTIONS}, took {self.seconds} seconds ({self.fps} average fps)"
                )
            else:
                logger.info(
                    f"Finishing: agent took {self.action_counter} actions, took {self.seconds} seconds ({self.fps} average fps)"
                )

    @abstractmethod
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""
        raise NotImplementedError

    @abstractmethod
    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Choose which action the Agent should take, fill in any arguments, and return it."""
        raise NotImplementedError
