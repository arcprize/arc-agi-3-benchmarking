from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Sequence, Tuple

from PIL import Image
from threadsafe_datastore import Datastore

from arcagi3.checkpoint import CheckpointManager
from arcagi3.schemas import CompletionTokensDetails, Cost, GameActionRecord, ModelCallRecord, Usage
from arcagi3.types import FrameGrid, FrameGridSequence, FrameImageSequence
from arcagi3.utils.image import grid_to_image

Size = Tuple[int, int]
Resize = int | Size | None
_UNSET = object()
_TERMINAL_GAME_STATES = ("WIN", "GAME_OVER")


@dataclass(frozen=True)
class CheckpointState:
    checkpoint_id: Optional[str]
    checkpoint_dir: str


@dataclass(frozen=True)
class FrameState:
    frame_grids: Tuple[FrameGrid, ...] = ()
    previous_grids: Tuple[FrameGrid, ...] = ()


@dataclass(frozen=True)
class GameProgress:
    game_id: str = ""
    guid: Optional[str] = None
    current_score: int = 0
    current_state: str = "IN_PROGRESS"
    previous_score: int = 0
    play_num: int = 1
    play_action_counter: int = 0
    action_counter: int = 0
    available_actions: Tuple[str, ...] = ()


@dataclass(frozen=True)
class MetricsState:
    """
    Metrics are treated as write-only via SessionContext methods.

    Underlying Cost/Usage are mutable pydantic models; do not mutate them directly.
    """

    total_cost: Cost = field(
        default_factory=lambda: Cost(prompt_cost=0.0, completion_cost=0.0, total_cost=0.0)
    )
    total_usage: Usage = field(
        default_factory=lambda: Usage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            completion_tokens_details=CompletionTokensDetails(),
        )
    )


@dataclass(frozen=True)
class HistoryState:
    actions: Tuple[GameActionRecord, ...] = ()
    model_calls: Tuple[ModelCallRecord, ...] = ()


class SessionContext:
    """
    Context object containing session state and datastore for agent steps.

    This object is passed to each step() call and provides access to:
    - The thread-safe datastore for storing arbitrary state
    - Current game state (frames, score, etc.)
    - Metadata about the current play and action
    - Previous state for comparison
    """

    def __init__(
        self,
        checkpoint_id: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        datastore: Optional[Datastore] = None,
        frames: Optional[FrameState] = None,
        game: Optional[GameProgress] = None,
        metrics: Optional[MetricsState] = None,
        history: Optional[HistoryState] = None,
    ):
        """
        Initialize the session context with grouped state objects.

        All parameters are optional with sensible defaults. State can be updated
        via properties and the update() method after creation.

        Args:
            checkpoint_id: Checkpoint identifier for saving/loading state
            checkpoint_dir: Directory for checkpoint storage
            datastore: Thread-safe datastore instance (creates new if None)
            frames: Frame state (current and previous grids)
            game: Game progress state (score, state, counters, etc.)
            metrics: Metrics state (costs and usage)
            history: Action history state
        """
        self._lock = threading.RLock()

        self._checkpoint = CheckpointState(
            checkpoint_id=checkpoint_id,
            checkpoint_dir=checkpoint_dir or CheckpointManager.DEFAULT_CHECKPOINT_DIR,
        )
        self._datastore = datastore or Datastore()

        frames = frames or FrameState()
        game = game or GameProgress()
        metrics = metrics or MetricsState()
        history = history or HistoryState()

        self._frames = replace(
            frames,
            frame_grids=tuple(frames.frame_grids),
            previous_grids=tuple(frames.previous_grids),
        )
        self._game = replace(
            game,
            available_actions=tuple(game.available_actions),
        )
        self._metrics = metrics
        self._history = replace(history, actions=tuple(history.actions))
        self._checkpoint_manager: Optional[CheckpointManager] = None

    @property
    def datastore(self) -> Datastore:
        """Thread-safe datastore instance for this invocation."""
        return self._datastore

    def _datastore_snapshot_locked(self) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {}
        for k, v in self._datastore.items():
            if not isinstance(k, str):
                raise TypeError(f"Datastore key must be str for checkpointing; got {type(k)}")
            snapshot[k] = v
        return snapshot

    def datastore_snapshot(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable snapshot of the datastore.

        Contract:
        - Keys must be strings.
        - Values must be JSON-serializable.

        NOTE: It is the implementing agent's responsibility to only store
        JSON-serializable values in `context.datastore` if checkpoint/resume
        is expected to work. Non-serializable values will raise at checkpoint time.
        """
        with self._lock:
            snapshot = self._datastore_snapshot_locked()

        json.dumps(snapshot)
        return snapshot

    # ------------------------------------------------------------------
    # Grouped state snapshots (read-only copies)
    # ------------------------------------------------------------------

    @property
    def frames(self) -> FrameState:
        with self._lock:
            return self._frames

    @property
    def game(self) -> GameProgress:
        with self._lock:
            return self._game

    @property
    def metrics(self) -> MetricsState:
        with self._lock:
            return self._metrics

    @property
    def history(self) -> HistoryState:
        with self._lock:
            return self._history

    @property
    def checkpoint_id(self) -> Optional[str]:
        with self._lock:
            return self._checkpoint.checkpoint_id

    @checkpoint_id.setter
    def checkpoint_id(self, value: Optional[str]) -> None:
        with self._lock:
            self._checkpoint = replace(self._checkpoint, checkpoint_id=value)
            self._checkpoint_manager = None  # Invalidate cached manager

    @property
    def checkpoint_dir(self) -> str:
        with self._lock:
            return self._checkpoint.checkpoint_dir

    @checkpoint_dir.setter
    def checkpoint_dir(self, value: Optional[str]) -> None:
        with self._lock:
            self._checkpoint = replace(
                self._checkpoint,
                checkpoint_dir=value or CheckpointManager.DEFAULT_CHECKPOINT_DIR,
            )
            self._checkpoint_manager = None  # Invalidate cached manager

    def _get_checkpoint_manager(self) -> CheckpointManager:
        """Get or create the cached CheckpointManager."""
        with self._lock:
            if self._checkpoint_manager is None:
                checkpoint_id = self._checkpoint.checkpoint_id
                if not checkpoint_id:
                    raise ValueError(
                        "SessionContext.checkpoint_id is required to save a checkpoint"
                    )
                self._checkpoint_manager = CheckpointManager(
                    checkpoint_id, checkpoint_dir=self._checkpoint.checkpoint_dir
                )
            return self._checkpoint_manager

    def get_state(self, extra_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get all context state in a single lock acquisition for efficient checkpointing.

        Returns organized state structure matching the checkpoint format.
        """
        with self._lock:
            datastore_snapshot = self._datastore_snapshot_locked()
            frames = self._frames
            game = self._game
            metrics = self._metrics
            history = self._history

        state = {
            "frames": {
                "frame_grids": list(frames.frame_grids),
            },
            "game": {
                "game_id": game.game_id,
                "guid": game.guid,
                "current_score": game.current_score,
                "current_state": game.current_state,
                "previous_score": game.previous_score,
                "play_num": game.play_num,
                "play_action_counter": game.play_action_counter,
                "action_counter": game.action_counter,
                "available_actions": list(game.available_actions),
            },
            "metrics": {
                "total_cost": metrics.total_cost.model_dump() if metrics.total_cost else None,
                "total_usage": metrics.total_usage.model_dump() if metrics.total_usage else None,
            },
            "history": {
                "action_history": [action.model_dump() for action in history.actions],
                "model_calls": [call.model_dump(mode="json") for call in history.model_calls],
            },
            "datastore": datastore_snapshot,
        }

        if extra_metadata:
            state["metadata"] = dict(extra_metadata)

        json.dumps(state)
        return state

    def save_checkpoint_state(self, state: Dict[str, Any]) -> None:
        mgr = self._get_checkpoint_manager()
        mgr.save_state(state)

    @classmethod
    def restore_from_checkpoint(
        cls,
        checkpoint_id: str,
        checkpoint_dir: Optional[str] = None,
        datastore: Optional[Datastore] = None,
    ) -> SessionContext:
        """
        Restore a SessionContext from checkpoint storage.

        This is a pure factory: it does not require an agent instance.
        """
        mgr = CheckpointManager(checkpoint_id, checkpoint_dir=checkpoint_dir)
        state = mgr.load_state()

        # Extract from organized structure
        frames_dict = state.get("frames", {})
        game_dict = state.get("game", {})
        metrics_dict = state.get("metrics", {})
        history_dict = state.get("history", {})
        datastore_snapshot = state.get("datastore", {})

        # Restore datastore snapshot (exact, JSON-only contract)
        ds = datastore or Datastore()
        if not isinstance(datastore_snapshot, dict):
            raise TypeError(f"Checkpoint datastore must be a dict; got {type(datastore_snapshot)}")
        for k, v in datastore_snapshot.items():
            if not isinstance(k, str):
                raise TypeError(f"Checkpoint datastore key must be str; got {type(k)}")
            ds[k] = v

        # Build state objects from checkpoint data
        frames = FrameState(
            frame_grids=tuple(frames_dict.get("frame_grids", [])),
            previous_grids=(),
        )

        game = GameProgress(
            game_id=game_dict.get("game_id", ""),
            guid=game_dict.get("guid"),
            current_score=game_dict.get("current_score", 0),
            current_state=game_dict.get("current_state", "IN_PROGRESS"),
            previous_score=game_dict.get("previous_score", 0),
            play_num=game_dict.get("play_num", 1),
            play_action_counter=game_dict.get("play_action_counter", 0),
            action_counter=game_dict.get("action_counter", 0),
            available_actions=tuple(game_dict.get("available_actions", [])),
        )

        total_cost_payload = metrics_dict.get("total_cost")
        if total_cost_payload is None:
            total_cost = Cost(prompt_cost=0.0, completion_cost=0.0, total_cost=0.0)
        elif isinstance(total_cost_payload, Cost):
            total_cost = total_cost_payload
        elif isinstance(total_cost_payload, dict):
            total_cost = Cost(**total_cost_payload)
        else:
            raise TypeError(f"Invalid Cost payload: {type(total_cost_payload)}")

        total_usage_payload = metrics_dict.get("total_usage")
        if total_usage_payload is None:
            total_usage = Usage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                completion_tokens_details=CompletionTokensDetails(),
            )
        elif isinstance(total_usage_payload, Usage):
            total_usage = total_usage_payload
        elif isinstance(total_usage_payload, dict):
            total_usage = Usage(**total_usage_payload)
        else:
            raise TypeError(f"Invalid Usage payload: {type(total_usage_payload)}")
        metrics = MetricsState(
            total_cost=total_cost,
            total_usage=total_usage,
        )

        action_history_payload = history_dict.get("action_history", [])
        actions = tuple(
            record if isinstance(record, GameActionRecord) else GameActionRecord(**record)
            for record in action_history_payload
        )
        model_calls_payload = history_dict.get("model_calls", [])
        model_calls = tuple(
            record if isinstance(record, ModelCallRecord) else ModelCallRecord(**record)
            for record in model_calls_payload
        )
        history = HistoryState(actions=actions, model_calls=model_calls)

        return cls(
            checkpoint_id=checkpoint_id,
            checkpoint_dir=checkpoint_dir,
            datastore=ds,
            frames=frames,
            game=game,
            metrics=metrics,
            history=history,
        )

    # ---------------------------------------------------------------------
    # Invocation-scoped metrics/history
    # ---------------------------------------------------------------------

    def append_action_record(self, record: GameActionRecord) -> None:
        with self._lock:
            self._history = replace(self._history, actions=self._history.actions + (record,))

    def append_model_call(self, record: ModelCallRecord) -> None:
        with self._lock:
            call_num = record.call_num
            if call_num is None:
                call_num = len(self._history.model_calls) + 1
                record = record.model_copy(update={"call_num": call_num})
            self._history = replace(
                self._history,
                model_calls=self._history.model_calls + (record,),
            )

    def metrics_snapshot(self) -> Cost:
        with self._lock:
            return Cost(**self._metrics.total_cost.model_dump())

    def add_usage_and_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_tokens: int = 0,
        pricing: Optional[Any] = None,
    ) -> None:
        """
        Append token usage + dollar cost to this invocation context.

        pricing is expected to have .input and .output fields expressed as $ per 1M tokens.
        """
        prompt_tokens = int(prompt_tokens)
        completion_tokens = int(completion_tokens)
        reasoning_tokens = int(reasoning_tokens)

        with self._lock:
            usage = self._metrics.total_usage
            details = usage.completion_tokens_details
            if reasoning_tokens > 0:
                details = details or CompletionTokensDetails()
                details = CompletionTokensDetails(
                    reasoning_tokens=details.reasoning_tokens + reasoning_tokens,
                    accepted_prediction_tokens=details.accepted_prediction_tokens,
                    rejected_prediction_tokens=details.rejected_prediction_tokens,
                )
            new_usage = Usage(
                prompt_tokens=usage.prompt_tokens + prompt_tokens,
                completion_tokens=usage.completion_tokens + completion_tokens,
                total_tokens=usage.total_tokens + prompt_tokens + completion_tokens,
                completion_tokens_details=details,
            )

            cost = self._metrics.total_cost
            if pricing is None:
                new_cost = cost
            else:
                input_cost_per_token = float(getattr(pricing, "input")) / 1_000_000
                output_cost_per_token = float(getattr(pricing, "output")) / 1_000_000

                prompt_cost_delta = prompt_tokens * input_cost_per_token
                completion_cost_delta = completion_tokens * output_cost_per_token
                reasoning_cost_delta = (
                    reasoning_tokens * output_cost_per_token if reasoning_tokens > 0 else 0.0
                )

                new_reasoning_cost = (
                    (cost.reasoning_cost or 0.0) + reasoning_cost_delta
                    if reasoning_tokens > 0
                    else cost.reasoning_cost
                )
                new_cost = Cost(
                    prompt_cost=cost.prompt_cost + prompt_cost_delta,
                    completion_cost=cost.completion_cost + completion_cost_delta,
                    reasoning_cost=new_reasoning_cost,
                    total_cost=cost.total_cost
                    + prompt_cost_delta
                    + completion_cost_delta
                    + reasoning_cost_delta,
                )

            self._metrics = replace(self._metrics, total_cost=new_cost, total_usage=new_usage)

    def set_available_actions(self, actions: Sequence[str]) -> None:
        with self._lock:
            self._game = replace(self._game, available_actions=tuple(actions))

    def set_counters(
        self,
        *,
        play_action_counter: int | object = _UNSET,
        action_counter: int | object = _UNSET,
    ) -> None:
        updates: Dict[str, Any] = {}
        if play_action_counter is not _UNSET:
            updates["play_action_counter"] = int(play_action_counter)
        if action_counter is not _UNSET:
            updates["action_counter"] = int(action_counter)
        if not updates:
            return
        with self._lock:
            self._game = replace(self._game, **updates)

    def set_play_num(self, play_num: int) -> None:
        with self._lock:
            self._game = replace(self._game, play_num=int(play_num))

    def set_game_identity(
        self,
        *,
        game_id: Optional[str] | object = _UNSET,
        guid: Optional[str] | object = _UNSET,
    ) -> None:
        updates: Dict[str, Any] = {}
        if game_id is not _UNSET:
            updates["game_id"] = game_id or ""
        if guid is not _UNSET:
            updates["guid"] = guid
        if not updates:
            return
        with self._lock:
            self._game = replace(self._game, **updates)

    def update(
        self,
        frame_grids: FrameGridSequence,
        current_score: int,
        current_state: str,
        guid: Optional[str] = None,
    ) -> None:
        """
        Update context with new game state from the game client.

        This is the simplest update - only accepts what comes from the game client
        after executing an action: frame grids, score, state, and optional guid.
        Frame images are automatically generated from frame grids.
        """
        new_grids = tuple(frame_grids) if frame_grids else ()
        with self._lock:
            frames = self._frames
            game = self._game
            self._frames = replace(
                frames,
                previous_grids=frames.frame_grids,
                frame_grids=new_grids,
            )
            self._game = replace(
                game,
                # Derived fields: previous_score is set from current_score here only.
                previous_score=game.current_score,
                current_score=current_score,
                current_state=current_state,
                guid=guid if guid is not None else game.guid,
            )

    @property
    def is_won(self) -> bool:
        with self._lock:
            return self._game.current_state == "WIN"

    @property
    def is_game_over(self) -> bool:
        with self._lock:
            return self._game.current_state in _TERMINAL_GAME_STATES

    @property
    def score_increased(self) -> bool:
        with self._lock:
            return self._game.current_score > self._game.previous_score

    @property
    def frame_images(self) -> FrameImageSequence:
        grids: Tuple[FrameGrid, ...]
        with self._lock:
            grids = self._frames.frame_grids
        if not grids:
            return []
        return [grid_to_image(frame) for frame in grids]

    def get_frame_images(self, resize: Resize = None) -> FrameImageSequence:
        images = self.frame_images
        if resize is None:
            size = None
        elif isinstance(resize, int):
            size = (resize, resize)
        else:
            size = resize
        if size is None:
            return images
        return [img.resize(size, Image.Resampling.LANCZOS) for img in images]

    @property
    def previous_images(self) -> FrameImageSequence:
        grids: Tuple[FrameGrid, ...]
        with self._lock:
            grids = self._frames.previous_grids
        if not grids:
            return []
        return [grid_to_image(frame) for frame in grids]

    def last_frame_image(self, resize: Resize = None) -> Optional[Image.Image]:
        last_grid: Optional[FrameGrid]
        with self._lock:
            last_grid = self._frames.frame_grids[-1] if self._frames.frame_grids else None
        if last_grid is None:
            return None
        img = grid_to_image(last_grid)
        if resize is None:
            size = None
        elif isinstance(resize, int):
            size = (resize, resize)
        else:
            size = resize
        if size is None:
            return img
        return img.resize(size, Image.Resampling.LANCZOS)

    @property
    def last_frame_grid(self) -> Optional[FrameGrid]:
        with self._lock:
            return self._frames.frame_grids[-1] if self._frames.frame_grids else None
