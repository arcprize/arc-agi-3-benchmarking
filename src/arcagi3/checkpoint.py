"""
Checkpoint functionality for saving and loading agent state.

This allows for resuming runs after crashes or interruptions.
"""
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _require_dict(value: Any, name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    raise TypeError(f"{name} must be a dict; got {type(value)}")


class CheckpointManager:
    """Manages checkpointing of agent state"""

    DEFAULT_CHECKPOINT_DIR = ".checkpoint"

    def __init__(self, card_id: str, checkpoint_dir: Optional[str] = None):
        """
        Initialize checkpoint manager.

        Args:
            card_id: Scorecard ID to use as checkpoint directory name
            checkpoint_dir: Base directory for checkpoints (defaults to CHECKPOINT_DIR)
        """
        self.card_id = card_id
        base_dir = checkpoint_dir or self.DEFAULT_CHECKPOINT_DIR
        self.checkpoint_path = Path(base_dir) / card_id

    def save_state(self, state: Dict[str, Any]):
        """
        Save the current agent state to a checkpoint file.

        Args:
            state: Dictionary containing organized state (metadata, frames, game, metrics, history, datastore).
        """
        logger.info(f"Saving checkpoint to {self.checkpoint_path}")

        # Create checkpoint directory
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Extract from organized structure
        metadata_dict = state.get("metadata", {})
        frames_dict = state.get("frames", {})
        game_dict = state.get("game", {})
        metrics_dict = state.get("metrics", {})
        history_dict = state.get("history", {})
        datastore = state.get("datastore", {})

        # Extract metadata fields
        config = metadata_dict.get("config")
        checkpoint_id = metadata_dict.get("checkpoint_id")
        max_actions = metadata_dict.get("max_actions")
        retry_attempts = metadata_dict.get("retry_attempts")  # Legacy field
        num_plays = metadata_dict.get("num_plays")
        max_episode_actions = metadata_dict.get("max_episode_actions", 0)

        # Extract frame fields
        frame_grids = frames_dict.get("frame_grids", [])

        # Extract game fields
        game_id = game_dict.get("game_id", "")
        guid = game_dict.get("guid")
        action_counter = game_dict.get("action_counter", 0)
        current_play = game_dict.get("play_num", 1)
        play_action_counter = game_dict.get("play_action_counter", 0)
        current_score = game_dict.get("current_score", 0)
        current_state = game_dict.get("current_state", "IN_PROGRESS")
        previous_score = game_dict.get("previous_score", 0)
        available_actions = game_dict.get("available_actions", [])

        # Extract metrics fields
        total_cost = _require_dict(metrics_dict.get("total_cost"), "metrics.total_cost")
        total_usage = _require_dict(metrics_dict.get("total_usage"), "metrics.total_usage")

        # Extract history fields
        action_history = history_dict.get("action_history", [])
        if not isinstance(action_history, list):
            raise TypeError(f"history.action_history must be a list; got {type(action_history)}")
        for idx, action in enumerate(action_history):
            if not isinstance(action, dict):
                raise TypeError(f"history.action_history[{idx}] must be a dict; got {type(action)}")
        model_calls = history_dict.get("model_calls", [])
        if not isinstance(model_calls, list):
            raise TypeError(f"history.model_calls must be a list; got {type(model_calls)}")
        for idx, call in enumerate(model_calls):
            if not isinstance(call, dict):
                raise TypeError(f"history.model_calls[{idx}] must be a dict; got {type(call)}")

        # Save metadata
        metadata = {
            "card_id": self.card_id,
            "checkpoint_id": checkpoint_id,
            "config": config,
            "game_id": game_id,
            "guid": guid,
            "max_actions": max_actions,
            "retry_attempts": retry_attempts,
            "num_plays": num_plays,
            "max_episode_actions": max_episode_actions,
            "action_counter": action_counter,
            "current_play": current_play,
            "play_action_counter": play_action_counter,
            "current_score": current_score,
            "current_state": current_state,
            "previous_score": previous_score,
            "frame_grids": frame_grids,
            "available_actions": available_actions,
            "datastore": datastore,
            "checkpoint_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Validate JSON-serializability eagerly (fail-fast before writing partial checkpoints)
        json.dumps(metadata)

        with open(self.checkpoint_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save per-action datastore snapshots as JSONL so you can inspect how the
        # agent state evolves over time (similar to action_history/model_completion),
        # without rewriting a growing JSON array.
        #
        # Append-only file:
        #   .checkpoint/<CARD_ID>/datastore_history.jsonl
        #
        # This is not used for resume; resume uses the latest snapshot in
        # metadata.json.
        try:
            action_num = int(action_counter)
        except Exception:
            action_num = 0
        if action_num > 0:
            record = {
                "action_num": action_num,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "datastore": datastore,
            }
            json.dumps(record)
            with open(self.checkpoint_path / "datastore_history.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")

        # Save costs and usage
        costs = {
            "total_cost": total_cost,
            "total_usage": total_usage,
        }

        with open(self.checkpoint_path / "costs.json", "w") as f:
            json.dump(costs, f, indent=2)

        # Save action history
        with open(self.checkpoint_path / "action_history.json", "w") as f:
            json.dump(action_history, f, indent=2)

        # Save model completions separately for easier inspection
        with open(self.checkpoint_path / "model_completion.json", "w") as f:
            json.dump(model_calls, f, indent=2)

        logger.info("Checkpoint saved successfully")

    def write_error(self, payload: Dict[str, Any]) -> Path:
        """
        Persist a human-readable error payload alongside checkpoint data.
        """
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        error_path = self.checkpoint_path / "error.json"

        json.dumps(payload)
        with open(error_path, "w") as f:
            json.dump(payload, f, indent=2)

        return error_path

    def load_state(self) -> Dict[str, Any]:
        """
        Load agent state from checkpoint.

        Returns:
            Dictionary containing all saved state

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            ValueError: If checkpoint is invalid or incomplete
        """
        logger.info(f"Loading checkpoint from {self.checkpoint_path}")

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        # Load metadata
        metadata_path = self.checkpoint_path / "metadata.json"
        if not metadata_path.exists():
            raise ValueError("Checkpoint missing metadata.json")

        with open(metadata_path) as f:
            metadata = json.load(f)

        # Load datastore snapshot (prefer datastore.json; fall back to metadata.json for older checkpoints)
        datastore = metadata.get("datastore", {})

        # Load costs (raw dicts)
        costs_path = self.checkpoint_path / "costs.json"
        if costs_path.exists():
            with open(costs_path) as f:
                costs_data = json.load(f)
                total_cost = costs_data.get("total_cost", {})
                total_usage = costs_data.get("total_usage", {})
        else:
            total_cost = {}
            total_usage = {}

        # Load action history (raw dicts)
        action_history = []
        model_calls = []
        action_history_path = self.checkpoint_path / "action_history.json"
        if action_history_path.exists():
            with open(action_history_path) as f:
                action_history_data = json.load(f)
                if isinstance(action_history_data, dict):
                    action_history = action_history_data.get("action_history", [])
                    model_calls = action_history_data.get("model_calls", [])
                else:
                    action_history = action_history_data

        model_completion_path = self.checkpoint_path / "model_completion.json"
        if model_completion_path.exists():
            with open(model_completion_path) as f:
                model_calls = json.load(f)

        logger.info("Checkpoint loaded successfully")

        return {
            "metadata": {
                "config": metadata.get("config"),
                "checkpoint_id": metadata.get("checkpoint_id"),
                "max_actions": metadata.get("max_actions"),
                "num_plays": metadata.get("num_plays"),
                "max_episode_actions": metadata.get("max_episode_actions", 0),
            },
            "frames": {
                "frame_grids": metadata.get("frame_grids", []),
            },
            "game": {
                "game_id": metadata.get("game_id", ""),
                "guid": metadata.get("guid"),
                "current_score": metadata.get("current_score", 0),
                "current_state": metadata.get("current_state", "IN_PROGRESS"),
                "previous_score": metadata.get("previous_score", 0),
                "play_num": metadata.get("play_num", metadata.get("current_play", 1)),
                "play_action_counter": metadata.get("play_action_counter", 0),
                "action_counter": metadata.get("action_counter", 0),
                "available_actions": metadata.get("available_actions", []),
            },
            "metrics": {
                "total_cost": total_cost,
                "total_usage": total_usage,
            },
            "history": {
                "action_history": action_history,
                "model_calls": model_calls,
            },
            "datastore": datastore,
        }

    def checkpoint_exists(self) -> bool:
        """Check if checkpoint exists for this card_id"""
        return self.checkpoint_path.exists() and (self.checkpoint_path / "metadata.json").exists()

    def delete_checkpoint(self):
        """Delete the checkpoint directory"""
        if self.checkpoint_path.exists():
            shutil.rmtree(self.checkpoint_path)
            logger.info(f"Deleted checkpoint: {self.checkpoint_path}")

    @staticmethod
    def list_checkpoints() -> List[str]:
        """List all available checkpoint card_ids"""
        checkpoint_dir = Path(CheckpointManager.DEFAULT_CHECKPOINT_DIR)
        if not checkpoint_dir.exists():
            return []

        checkpoints = []
        for card_dir in checkpoint_dir.iterdir():
            if card_dir.is_dir() and (card_dir / "metadata.json").exists():
                checkpoints.append(card_dir.name)

        return sorted(checkpoints)

    @staticmethod
    def get_checkpoint_info(card_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata about a checkpoint"""
        checkpoint_path = Path(CheckpointManager.DEFAULT_CHECKPOINT_DIR) / card_id
        metadata_path = checkpoint_path / "metadata.json"

        if not metadata_path.exists():
            return None

        with open(metadata_path) as f:
            return json.load(f)
