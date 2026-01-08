from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import json

from PIL import Image

from threadsafe_datastore import Datastore
from arcagi3.utils.image import grid_to_image
from arcagi3.schemas import Cost, Usage, CompletionTokensDetails, GameActionRecord
from arcagi3.checkpoint import CheckpointManager
from arcagi3.types import FrameGrid, FrameGridSequence, FrameImageSequence

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
        frame_images: Optional[FrameImageSequence] = None,
        frame_grids: Optional[FrameGridSequence] = None,
        current_score: int = 0,
        current_state: str = "IN_PROGRESS",
        game_id: Optional[str] = None,
        play_num: int = 1,
        play_action_counter: int = 0,
        action_counter: int = 0,
        guid: Optional[str] = None,
        previous_score: int = 0,
        previous_images: Optional[FrameImageSequence] = None,
        previous_grids: Optional[FrameGridSequence] = None,
        available_actions: Optional[List[str]] = None,
        total_cost: Optional[Cost] = None,
        total_usage: Optional[Usage] = None,
        action_history: Optional[List[GameActionRecord]] = None,
    ):
        """
        Initialize the session context with all optional parameters.
        
        All parameters have defaults, allowing the context to be created
        with minimal or no arguments and then updated as needed.
        
        Args:
            datastore: Thread-safe datastore instance. If None, creates a new one
            frame_images: Current frame images. Defaults to empty list
            frame_grids: Current frame grids. Defaults to empty list
            current_score: Current game score. Defaults to 0
            current_state: Current game state (WIN, GAME_OVER, IN_PROGRESS). Defaults to "IN_PROGRESS"
            game_id: Game identifier. Defaults to empty string
            play_num: Current play number. Defaults to 1
            play_action_counter: Action counter within this play. Defaults to 0
            action_counter: Global action counter. Defaults to 0
            guid: Game session identifier. Defaults to None
            previous_score: Previous score (for comparison). Defaults to 0
            previous_images: Previous frame images (for comparison). Defaults to empty list
            previous_grids: Previous frame grids (for comparison). Defaults to empty list
            available_actions: List of available actions for this game. Defaults to empty list
        """
        # Create datastore if not provided
        if datastore is None:
            datastore = Datastore()

        # Checkpoint targeting (invocation-scoped)
        self._checkpoint_id = checkpoint_id
        self._checkpoint_dir = checkpoint_dir or CheckpointManager.CHECKPOINT_DIR
        
        self._datastore = datastore
        # Store only grids - images are generated on-demand
        # If frame_images are provided, we still store grids (they're the source of truth)
        self._frame_grids = frame_grids or []
        # If frame_images provided but no grids, we can't generate grids from images
        # So we accept frame_images for backward compatibility but prefer grids
        if frame_images and not frame_grids:
            # If images provided without grids, we can't convert back, so store images
            # This is a fallback for initialization - update() will use grids
            self._frame_images_cache = frame_images
        else:
            self._frame_images_cache = None
        self._current_score = current_score
        self._current_state = current_state
        self._game_id = game_id or ""
        self._play_num = play_num
        self._play_action_counter = play_action_counter
        self._action_counter = action_counter
        self._guid = guid
        self._previous_score = previous_score
        # Previous images are generated on-demand from previous_grids
        self._previous_grids = previous_grids or []
        self._available_actions = available_actions or []

        # Metrics
        self._total_cost = total_cost or Cost(prompt_cost=0.0, completion_cost=0.0, total_cost=0.0)
        self._total_usage = total_usage or Usage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            completion_tokens_details=CompletionTokensDetails(),
        )
        self._action_history = action_history or []
    
    @property
    def datastore(self) -> Datastore:
        """Thread-safe datastore instance for this invocation."""
        return self._datastore

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
        snapshot: Dict[str, Any] = {}
        for k, v in self._datastore.items():
            if not isinstance(k, str):
                raise TypeError(f"Datastore key must be str for checkpointing; got {type(k)}")
            snapshot[k] = v

        # Validate JSON-serializability eagerly (fail-fast before writing partial checkpoints)
        json.dumps(snapshot)
        return snapshot

    @property
    def checkpoint_id(self) -> Optional[str]:
        return self._checkpoint_id

    @checkpoint_id.setter
    def checkpoint_id(self, value: Optional[str]) -> None:
        self._checkpoint_id = value

    @property
    def checkpoint_dir(self) -> str:
        return self._checkpoint_dir

    @checkpoint_dir.setter
    def checkpoint_dir(self, value: Optional[str]) -> None:
        self._checkpoint_dir = value or CheckpointManager.CHECKPOINT_DIR

    def save_checkpoint_state(self, state: Dict[str, Any]) -> None:
        if not self._checkpoint_id:
            raise ValueError("SessionContext.checkpoint_id is required to save a checkpoint")
        mgr = CheckpointManager(self._checkpoint_id, checkpoint_dir=self._checkpoint_dir)
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

        metadata = state["metadata"]
        metrics_state = state["metrics"]

        # Restore datastore snapshot (exact, JSON-only contract)
        ds = datastore or Datastore()
        snapshot = metadata.get("datastore") or {}
        if not isinstance(snapshot, dict):
            raise TypeError(f"Checkpoint datastore must be a dict; got {type(snapshot)}")
        for k, v in snapshot.items():
            if not isinstance(k, str):
                raise TypeError(f"Checkpoint datastore key must be str; got {type(k)}")
            ds[k] = v

        return cls(
            checkpoint_id=checkpoint_id,
            checkpoint_dir=checkpoint_dir,
            datastore=ds,
            game_id=metadata.get("game_id", ""),
            play_num=metadata.get("current_play", 1),
            play_action_counter=metadata.get("play_action_counter", 0),
            action_counter=metadata.get("action_counter", 0),
            guid=metadata.get("guid"),
            current_score=metadata.get("current_score", 0),
            current_state=metadata.get("current_state", "IN_PROGRESS"),
            previous_score=metadata.get("previous_score", 0),
            frame_grids=metadata.get("frame_grids", []),
            available_actions=metadata.get("available_actions", []),
            total_cost=metrics_state.get("total_cost"),
            total_usage=metrics_state.get("total_usage"),
            action_history=metrics_state.get("action_history", []),
        )

    # ---------------------------------------------------------------------
    # Invocation-scoped metrics
    # ---------------------------------------------------------------------

    @property
    def total_cost(self) -> Cost:
        return self._total_cost

    @property
    def total_usage(self) -> Usage:
        return self._total_usage

    @property
    def action_history(self) -> List[GameActionRecord]:
        return self._action_history

    def append_action_record(self, record: GameActionRecord) -> None:
        self._action_history.append(record)

    def metrics_snapshot(self) -> Cost:
        return Cost(
            prompt_cost=self._total_cost.prompt_cost,
            completion_cost=self._total_cost.completion_cost,
            reasoning_cost=self._total_cost.reasoning_cost,
            total_cost=self._total_cost.total_cost,
        )

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
        # Update usage
        self._total_usage.prompt_tokens += int(prompt_tokens)
        self._total_usage.completion_tokens += int(completion_tokens)
        self._total_usage.total_tokens += int(prompt_tokens) + int(completion_tokens)

        if reasoning_tokens and reasoning_tokens > 0:
            if self._total_usage.completion_tokens_details is None:
                self._total_usage.completion_tokens_details = CompletionTokensDetails()
            self._total_usage.completion_tokens_details.reasoning_tokens += int(reasoning_tokens)

        # Update cost if pricing is provided
        if pricing is None:
            return

        input_cost_per_token = float(getattr(pricing, "input")) / 1_000_000
        output_cost_per_token = float(getattr(pricing, "output")) / 1_000_000

        prompt_cost = int(prompt_tokens) * input_cost_per_token
        completion_cost = int(completion_tokens) * output_cost_per_token
        reasoning_cost = int(reasoning_tokens) * output_cost_per_token if reasoning_tokens and reasoning_tokens > 0 else 0.0

        self._total_cost.prompt_cost += prompt_cost
        self._total_cost.completion_cost += completion_cost
        if reasoning_tokens and reasoning_tokens > 0:
            if self._total_cost.reasoning_cost is None:
                self._total_cost.reasoning_cost = 0.0
            self._total_cost.reasoning_cost += reasoning_cost
        self._total_cost.total_cost += prompt_cost + completion_cost + reasoning_cost
    
    @property
    def frame_images(self) -> FrameImageSequence:
        """
        Current frame images, generated on-demand from frame grids.
        
        Images are generated fresh each time this property is accessed.
        This avoids storing large image objects in memory when grids are sufficient.
        """
        # Generate images from grids on-demand
        if not self._frame_grids:
            return []
        return [grid_to_image(frame) for frame in self._frame_grids]
    
    def get_frame_images(self, resize: Optional[Union[int, tuple]] = None) -> FrameImageSequence:
        """
        Get current frame images, optionally resized.
        
        Images are generated on-demand from frame grids.
        
        Args:
            resize: Optional resize parameter. If int, resizes to (resize, resize).
                    If tuple, resizes to (width, height). If None, returns original size.
        """
        images = self.frame_images  # Generate from grids
        if resize is None:
            return images
        
        # Handle resize parameter
        if isinstance(resize, int):
            size = (resize, resize)
        else:
            size = resize
        
        return [img.resize(size, Image.Resampling.LANCZOS) for img in images]
    
    @property
    def frame_grids(self) -> FrameGridSequence:
        """Current frame grids."""
        return self._frame_grids
    
    @property
    def current_score(self) -> int:
        """Current game score."""
        return self._current_score
    
    @property
    def current_state(self) -> str:
        """Current game state (WIN, GAME_OVER, IN_PROGRESS)."""
        return self._current_state
    
    @property
    def game_id(self) -> str:
        """Game identifier."""
        return self._game_id

    @game_id.setter
    def game_id(self, value: str) -> None:
        self._game_id = value or ""
    
    @property
    def play_num(self) -> int:
        """Current play number."""
        return self._play_num

    @play_num.setter
    def play_num(self, value: int) -> None:
        self._play_num = int(value)
    
    @property
    def play_action_counter(self) -> int:
        """Action counter within this play."""
        return self._play_action_counter

    @play_action_counter.setter
    def play_action_counter(self, value: int) -> None:
        self._play_action_counter = int(value)
    
    @property
    def action_counter(self) -> int:
        """Global action counter across all plays."""
        return self._action_counter

    @action_counter.setter
    def action_counter(self, value: int) -> None:
        self._action_counter = int(value)
    
    @property
    def guid(self) -> Optional[str]:
        """Game session identifier."""
        return self._guid

    @guid.setter
    def guid(self, value: Optional[str]) -> None:
        self._guid = value
    
    @property
    def previous_score(self) -> int:
        """Previous score (for comparison)."""
        return self._previous_score
    
    @property
    def previous_images(self) -> FrameImageSequence:
        """
        Previous frame images (for comparison), generated on-demand from previous grids.
        
        Images are generated fresh each time this property is accessed.
        """
        if not self._previous_grids:
            return []
        return [grid_to_image(frame) for frame in self._previous_grids]
    
    @property
    def previous_grids(self) -> FrameGridSequence:
        """Previous frame grids (for comparison)."""
        return self._previous_grids
    
    @property
    def available_actions(self) -> List[str]:
        """List of available actions for this game."""
        return self._available_actions

    @available_actions.setter
    def available_actions(self, value: Optional[List[str]]) -> None:
        self._available_actions = value or []
    
    @property
    def is_won(self) -> bool:
        """Whether the game has been won."""
        return self._current_state == "WIN"
    
    @property
    def is_game_over(self) -> bool:
        """Whether the game is over (won or lost)."""
        return self._current_state in ["WIN", "GAME_OVER"]
    
    @property
    def score_increased(self) -> bool:
        """Whether the score increased from the previous step."""
        return self._current_score > self._previous_score
    
    def last_frame_image(self, resize: Optional[Union[int, tuple]] = None) -> Optional[Image.Image]:
        """
        The last frame image (most recent frame), generated on-demand from frame grids.
        
        Args:
            resize: Optional resize parameter. If int, resizes to (resize, resize).
                    If tuple, resizes to (width, height). If None, returns original size.
        """
        if not self._frame_grids:
            return None
        
        # Generate image from last grid on-demand
        img = grid_to_image(self._frame_grids[-1])
        if resize is None:
            return img
        
        # Handle resize parameter
        if isinstance(resize, int):
            size = (resize, resize)
        else:
            size = resize
        
        return img.resize(size, Image.Resampling.LANCZOS)
    
    @property
    def last_frame_grid(self) -> Optional[FrameGrid]:
        """The last frame grid (most recent frame)."""
        return self._frame_grids[-1] if self._frame_grids else None
    
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
        
        Args:
            frame_grids: Current frame grids from game state
            current_score: Current game score
            current_state: Current game state (WIN, GAME_OVER, IN_PROGRESS)
            guid: Optional game session identifier
        """
        # Store previous state before updating (only grids, images generated on-demand)
        self._previous_score = self._current_score
        self._previous_grids = self._frame_grids.copy() if self._frame_grids else []
        
        # Update current state - only store grids, images generated on-demand
        self._frame_grids = frame_grids
        self._frame_images_cache = None  # Clear any cached images
        self._current_score = current_score
        self._current_state = current_state
        if guid is not None:
            self._guid = guid

