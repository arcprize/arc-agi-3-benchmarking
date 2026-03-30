from __future__ import annotations

import copy
import logging
import time
import traceback
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from threadsafe_datastore import Datastore

from arcagi3.adapters import create_provider
from arcagi3.breakpoints.manager import BreakpointHook, BreakpointManager
from arcagi3.breakpoints.spec import BreakpointSpec, load_breakpoint_spec, merge_breakpoint_specs
from arcagi3.game_client import GameClient
from arcagi3.schemas import ActionData, Cost, GameActionRecord, GameResult, GameStep
from arcagi3.utils import errors
from arcagi3.utils.context import GameProgress, SessionContext

logger = logging.getLogger(__name__)


# Game action vocabulary
HUMAN_ACTIONS: Dict[str, str] = {
    "ACTION1": "Move Up",
    "ACTION2": "Move Down",
    "ACTION3": "Move Left",
    "ACTION4": "Move Right",
    "ACTION5": "Perform Action",
    "ACTION6": "Click object on screen (describe object and relative position)",
    "ACTION7": "Undo",
}


HUMAN_ACTIONS_LIST = list(HUMAN_ACTIONS.keys())


class MultimodalAgent(ABC):
    """
    Abstract orchestrator for ARC-AGI-3 games to build agents around.

    The goal of this class to to provide a simple harness to easily
    connect AI agents to the ARC-AGI-3 games. To do this, implementing
    classes of this define their own `step(context) -> GameStep`
    function. Retries, provider management, checkpointing, and game
    clients are managed for the developer within.
    """

    def __init__(
        self,
        config: str,
        game_client: GameClient,
        card_id: str,
        max_actions: int = 40,
        num_plays: int = 1,
        max_episode_actions: int = 0,
        checkpoint_frequency: int = 1,
        checkpoint_dir: Optional[str] = None,
        breakpoints_enabled: bool = False,
        breakpoint_ws_url: str = "ws://localhost:8765/ws",
        breakpoint_schema_path: Optional[str] = None,
    ):
        self.config = config
        self.game_client = game_client
        self.card_id = card_id
        self.max_actions = max_actions
        self.num_plays = num_plays
        self.max_episode_actions = max_episode_actions

        self.provider = create_provider(config)

        self.checkpoint_frequency = checkpoint_frequency
        self.checkpoint_dir = checkpoint_dir

        self._breakpoints_enabled = breakpoints_enabled
        self._breakpoint_ws_url = breakpoint_ws_url
        self._breakpoint_schema_path = breakpoint_schema_path
        self._breakpoint_config_spec = load_breakpoint_spec(breakpoint_schema_path)
        self.breakpoint_manager: Optional[BreakpointManager] = None

        super().__init__()
        if self._breakpoints_enabled and self._breakpoint_config_spec:
            self.breakpoint_manager = BreakpointManager(
                enabled=True,
                ws_url=self._breakpoint_ws_url,
                spec=self._breakpoint_config_spec,
                hooks={},
            )
            self.breakpoint_manager.update_identity(config=self.config, card_id=self.card_id)

    def get_breakpoint_spec(self) -> Optional[BreakpointSpec]:
        return None

    def get_breakpoint_hooks(self) -> Dict[str, BreakpointHook]:
        return {}

    def register_breakpoints(
        self,
        *,
        runtime_spec: Optional[BreakpointSpec] = None,
        hooks: Optional[Dict[str, BreakpointHook]] = None,
    ) -> None:
        if not self._breakpoints_enabled:
            return
        merged = merge_breakpoint_specs(self._breakpoint_config_spec, runtime_spec)
        if self.breakpoint_manager is None:
            self.breakpoint_manager = BreakpointManager(
                enabled=True,
                ws_url=self._breakpoint_ws_url,
                spec=merged,
                hooks=hooks or {},
            )
        else:
            self.breakpoint_manager.update_spec(merged)
            if hooks is not None:
                self.breakpoint_manager.update_hooks(hooks)
        self.breakpoint_manager.update_identity(config=self.config, card_id=self.card_id)

    def save_checkpoint(self, context: SessionContext) -> None:
        """
        Save current invocation context to a checkpoint within
        the set checkpoint directory.
        """
        state = self.get_state(context)
        context.save_checkpoint_state(state)

    @abstractmethod
    def step(self, context: SessionContext) -> GameStep:
        """
        Perform one cognitive step in the game.

        Base loop expectations:
        - Must return a `GameStep`.
        - `GameStep.action` must contain at least an `"action"` string.
        - Optional `"data"` dict inside `GameStep.action` is passed through to the ARC API as action payload.
        - `GameStep.reasoning` must be a dict; it is deep-copied and sent to the ARC API as the `reasoning` field.
        - For `"ACTION6"`, you may alternatively return `"x"`/`"y"` at pixel-ish scale (0..127). The base loop
          clamps and downscales to the API coordinate system.
        """
        raise NotImplementedError

    def _execute_game_action(
        self,
        action_name: str,
        action_data: Optional[Dict[str, Any]],
        game_id: str,
        guid: Optional[str],
        reasoning: Optional[Dict[str, Any]] = None,
        context: Optional["SessionContext"] = None,
    ) -> Dict[str, Any]:
        if self.breakpoint_manager:
            payload = {
                "action": action_name,
                "action_data": action_data or {},
                "game_id": game_id,
                "guid": guid,
                "reasoning": reasoning,
            }
            updated = self.breakpoint_manager.pause(
                "execute_action.pre",
                payload,
                context=context,
                step_name="execute_action.pre",
            )
            action_name = updated.get("action", action_name)
            action_data = updated.get("action_data", action_data)
            game_id = updated.get("game_id", game_id)
            guid = updated.get("guid", guid)
            reasoning = updated.get("reasoning", reasoning)
        data: Dict[str, Any] = {"game_id": game_id}
        if guid:
            data["guid"] = guid
        if action_data:
            data.update(action_data)
        # Allow sending empty dicts; use None to omit the field entirely.
        if reasoning is not None:
            data["reasoning"] = reasoning
        result = self.game_client.execute_action(action_name, data)
        if self.breakpoint_manager:
            post_payload = {"result": result}
            updated = self.breakpoint_manager.pause(
                "execute_action.post",
                post_payload,
                context=context,
                step_name="execute_action.post",
            )
            if isinstance(updated.get("result"), dict):
                result = updated["result"]
        return result

    def get_state(self, context: SessionContext) -> Dict[str, Any]:
        """Return serializable invocation state for checkpointing."""
        return context.get_state(
            extra_metadata={
                "config": self.config,
                "checkpoint_id": context.checkpoint_id,
                "max_actions": self.max_actions,
                "num_plays": self.num_plays,
                "max_episode_actions": self.max_episode_actions,
            }
        )

    def play_game(
        self,
        game_id: str,
        resume_from_checkpoint: bool = False,
        checkpoint_id: Optional[str] = None,
    ) -> GameResult:
        checkpoint_id = checkpoint_id or self.card_id
        if self.breakpoint_manager:
            self.breakpoint_manager.update_identity(game_id=game_id)

        if resume_from_checkpoint:
            try:
                restored_context = SessionContext.restore_from_checkpoint(
                    checkpoint_id=checkpoint_id,
                    checkpoint_dir=self.checkpoint_dir,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to resume from checkpoint '{checkpoint_id}' "
                    f"in '{self.checkpoint_dir or 'default'}': {e}"
                ) from e
            else:
                if restored_context.game.game_id:
                    game_id = restored_context.game.game_id
                logger.info(f"Resuming game {game_id} from checkpoint")

                # Enforce max_actions on checkpoint continuation
                if (
                    self.max_actions > 0
                    and restored_context.game.action_counter >= self.max_actions
                ):
                    logger.warning(
                        f"Cannot resume from checkpoint: action counter ({restored_context.game.action_counter}) "
                        f"already exceeds or equals max_actions ({self.max_actions}). "
                        f"Please increase --max-actions or start a new game."
                    )
                    raise RuntimeError(
                        f"Checkpoint action counter ({restored_context.game.action_counter}) exceeds max_actions ({self.max_actions})"
                    )

        # Create or reuse invocation context
        if resume_from_checkpoint and "restored_context" in locals():
            context = restored_context
        else:
            context = SessionContext(
                checkpoint_id=checkpoint_id,
                checkpoint_dir=self.checkpoint_dir,
                datastore=Datastore(),
                game=GameProgress(game_id=game_id, play_num=1),
            )

        # Ensure restored contexts know where to checkpoint back to
        context.checkpoint_id = checkpoint_id
        context.checkpoint_dir = self.checkpoint_dir

        logger.info(f"Starting game {game_id} with config {self.config}")
        overall_start_time = time.time()

        best_result: Optional[GameResult] = None
        guid: Optional[str] = context.game.guid if resume_from_checkpoint else None

        start_play = context.game.play_num if resume_from_checkpoint else 1
        play_num = start_play

        while True:
            if self.num_plays > 0 and play_num > self.num_plays:
                break

            if self.max_actions > 0 and context.game.action_counter >= self.max_actions:
                logger.info(f"Global max_actions ({self.max_actions}) reached. Stopping.")
                break

            context.set_play_num(play_num)
            play_start_time = time.time()

            if play_num > 1:
                if self.num_plays == 0:
                    logger.info(f"Starting play {play_num}")
                else:
                    logger.info(f"Starting play {play_num}/{self.num_plays}")

            session_restored = False
            state: Dict[str, Any] = {}

            # Skip reset if resuming from checkpoint in the middle of a play
            if (
                resume_from_checkpoint
                and play_num == start_play
                and context.game.play_action_counter > 0
            ):
                logger.info(
                    f"Resuming play {play_num} at action {context.game.play_action_counter}"
                )

                if context.game.guid:
                    guid = context.game.guid
                    current_score = context.game.current_score
                    current_state = context.game.current_state or "IN_PROGRESS"
                    session_restored = True
                    state = {
                        "guid": guid,
                        "score": current_score,
                        "state": current_state,
                        "frame": list(context.frames.frame_grids),
                        "available_actions": list(context.game.available_actions),
                    }
                    logger.info(f"Continuing session with guid: {guid}, score: {current_score}")

                if not session_restored:
                    logger.info("No GUID found, starting new game session with restored state...")
                    state = self.game_client.reset_game(self.card_id, game_id, guid=None)
                    guid = state.get("guid")
                    current_score = state.get("levels_completed", 0)
                    current_state = state.get("state", "IN_PROGRESS")
                    context.set_available_actions(
                        state.get("available_actions", context.game.available_actions)
                    )

                    new_action_counter = context.game.action_counter + 1
                    context.set_counters(action_counter=new_action_counter)
                    context.append_action_record(
                        GameActionRecord(
                            action_num=new_action_counter,
                            action="RESET",
                            action_data=None,
                            reasoning={"system": "reset_game (checkpoint recovery)"},
                            result_score=current_score,
                            result_state=current_state,
                            cost=Cost(
                                prompt_cost=0.0,
                                completion_cost=0.0,
                                reasoning_cost=0.0,
                                total_cost=0.0,
                            ),
                        )
                    )

                play_action_counter = context.game.play_action_counter if session_restored else 1
                resume_from_checkpoint = False
            else:
                state = self.game_client.reset_game(self.card_id, game_id, guid=guid)
                guid = state.get("guid")
                current_score = state.get("levels_completed", 0)
                current_state = state.get("state", "IN_PROGRESS")
                context.set_available_actions(
                    state.get(
                        "available_actions",
                        list(context.game.available_actions) or list(HUMAN_ACTIONS.keys()),
                    )
                )

                # First RESET of play 1 is free; later resets count
                count_reset = resume_from_checkpoint or play_num > 1
                if count_reset:
                    new_action_counter = context.game.action_counter + 1
                    context.set_counters(action_counter=new_action_counter)
                    context.append_action_record(
                        GameActionRecord(
                            action_num=new_action_counter,
                            action="RESET",
                            action_data=None,
                            reasoning={"system": f"reset_game (start play {play_num})"},
                            result_score=current_score,
                            result_state=current_state,
                            cost=Cost(
                                prompt_cost=0.0,
                                completion_cost=0.0,
                                reasoning_cost=0.0,
                                total_cost=0.0,
                            ),
                        )
                    )
                    play_action_counter = 1
                else:
                    play_action_counter = 0

            context.set_game_identity(guid=guid)
            context.set_counters(play_action_counter=play_action_counter)

            session_result = self._run_session_loop(
                game_id=game_id, initial_state=state, context=context
            )

            current_score = session_result["score"]
            current_state = session_result["state"]
            play_action_counter = session_result["actions_taken"]
            play_action_history = session_result["action_history"]

            play_duration = time.time() - play_start_time
            scorecard_url = f"{self.game_client.ROOT_URL}/scorecards/{self.card_id}"

            play_result = GameResult(
                game_id=game_id,
                config=self.config,
                final_score=current_score,
                final_state=current_state,
                actions_taken=play_action_counter,
                duration_seconds=play_duration,
                total_cost=context.metrics.total_cost,
                usage=context.metrics.total_usage,
                actions=play_action_history,
                final_memory=None,
                timestamp=datetime.now(timezone.utc),
                scorecard_url=scorecard_url,
                card_id=self.card_id,
            )

            if best_result is None:
                best_result = play_result
            elif current_state == "WIN" and best_result.final_state != "WIN":
                best_result = play_result
            elif current_state == "WIN" and best_result.final_state == "WIN":
                if current_score > best_result.final_score:
                    best_result = play_result
            elif current_score > best_result.final_score:
                best_result = play_result

            if self.checkpoint_frequency > 0:
                self.save_checkpoint(context)

            if current_state == "WIN":
                logger.info(f"Game won on play {play_num}! Stopping early.")
                break

            if self.max_actions > 0 and context.game.action_counter >= self.max_actions:
                logger.info(f"Global max_actions ({self.max_actions}) reached. Stopping.")
                break

            play_num += 1

        overall_duration = time.time() - overall_start_time

        # Update best result with overall stats
        assert best_result is not None
        best_result.actions_taken = context.game.action_counter
        best_result.duration_seconds = overall_duration

        logger.info(
            f"All plays completed. Best: {best_result.final_state}, "
            f"Score: {best_result.final_score}, Total Actions: {context.game.action_counter}, "
            f"Cost: ${context.metrics.total_cost.total_cost:.4f}"
        )

        return best_result

    def _run_session_loop(
        self, game_id: str, initial_state: Dict[str, Any], context: SessionContext
    ) -> Dict[str, Any]:
        state = initial_state
        guid = state.get("guid")
        current_score = state.get("levels_completed", 0)
        current_state = state.get("state", "IN_PROGRESS")
        play_action_counter = context.game.play_action_counter

        play_action_history: List[GameActionRecord] = []

        # Reconstruct per-play history if resuming
        if guid and play_action_counter > 0:
            start_action_num = context.game.action_counter - play_action_counter + 1
            end_action_num = context.game.action_counter
            play_action_history = [
                action
                for action in context.history.actions
                if start_action_num <= action.action_num <= end_action_num
            ]

        context.set_game_identity(game_id=game_id, guid=guid)

        while (
            current_state not in ["WIN", "GAME_OVER"]
            and (self.max_episode_actions == 0 or play_action_counter < self.max_episode_actions)
            and (self.max_actions == 0 or context.game.action_counter < self.max_actions)
        ):
            try:
                frames = state.get("frame", [])
                if not frames:
                    logger.warning("No frames in state, breaking")
                    break

                # Update context with current state before step
                context.update(
                    frame_grids=frames,
                    current_score=current_score,
                    current_state=current_state,
                    guid=guid,
                )

                cost_before = context.metrics_snapshot()

                step = self.step(context)

                game_action_dict = step.action or {}
                action_name = game_action_dict.get("action")
                if not action_name:
                    raise ValueError("No action name in response")

                action_data_dict: Dict[str, Any] = {}
                if isinstance(game_action_dict.get("data"), dict):
                    action_data_dict = dict(game_action_dict.get("data") or {})
                elif action_name == "ACTION6":
                    x = game_action_dict.get("x", 0)
                    y = game_action_dict.get("y", 0)
                    action_data_dict = {
                        "x": max(0, min(int(x), 127)) // 2,
                        "y": max(0, min(int(y), 127)) // 2,
                    }

                reasoning_for_api = copy.deepcopy(step.reasoning or {})

                state = self._execute_game_action(
                    action_name, action_data_dict, game_id, guid, reasoning_for_api, context=context
                )
                guid = state.get("guid", guid)
                new_score = state.get("levels_completed", current_score)
                current_state = state.get("state", "IN_PROGRESS")

                context.update(
                    frame_grids=state.get("frame", []),
                    current_score=new_score,
                    current_state=current_state,
                    guid=guid,
                )

                current_cost = context.metrics.total_cost
                action_cost = Cost(
                    prompt_cost=current_cost.prompt_cost - cost_before.prompt_cost,
                    completion_cost=current_cost.completion_cost - cost_before.completion_cost,
                    reasoning_cost=(current_cost.reasoning_cost or 0)
                    - (cost_before.reasoning_cost or 0),
                    total_cost=current_cost.total_cost - cost_before.total_cost,
                )

                new_action_counter = context.game.action_counter + 1
                context.set_counters(action_counter=new_action_counter)
                action_record = GameActionRecord(
                    action_num=new_action_counter,
                    action=action_name,
                    action_data=ActionData(**action_data_dict) if action_data_dict else None,
                    reasoning=reasoning_for_api or None,
                    result_score=new_score,
                    result_state=current_state,
                    cost=action_cost,
                )
                play_action_history.append(action_record)
                context.append_action_record(action_record)

                current_score = new_score
                play_action_counter += 1
                context.set_counters(play_action_counter=play_action_counter)
                context.set_game_identity(guid=guid)

                if self.max_actions > 0 and context.game.action_counter >= self.max_actions:
                    logger.info(
                        f"Global max_actions ({self.max_actions}) reached. Stopping session."
                    )
                    break
                if self.max_episode_actions > 0 and play_action_counter >= self.max_episode_actions:
                    logger.info(
                        f"Episode max_episode_actions ({self.max_episode_actions}) reached. Stopping session."
                    )
                    break

                if (
                    self.checkpoint_frequency > 0
                    and play_action_counter % self.checkpoint_frequency == 0
                ):
                    self.save_checkpoint(context)

            except Exception as e:
                trace = traceback.format_exc()
                payload = errors.build_error_payload(
                    e,
                    context={
                        "game_id": game_id,
                        "config": self.config,
                        "card_id": self.card_id,
                        "checkpoint_id": context.checkpoint_id,
                        "phase": "session_loop",
                    },
                    trace=trace,
                )
                setattr(e, "_friendly_logged", True)
                logger.error(errors.format_user_message(payload))
                logger.error("Traceback:\n%s", trace)
                raise

        return {
            "score": current_score,
            "state": current_state,
            "actions_taken": play_action_counter,
            "action_history": play_action_history,
        }
