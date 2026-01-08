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

from arcagi3.schemas import CompletionTokensDetails, Cost, GameActionRecord, Usage

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpointing of agent state"""
    
    CHECKPOINT_DIR = ".checkpoint"
    
    def __init__(self, card_id: str, checkpoint_dir: Optional[str] = None):
        """
        Initialize checkpoint manager.
        
        Args:
            card_id: Scorecard ID to use as checkpoint directory name
            checkpoint_dir: Base directory for checkpoints (defaults to CHECKPOINT_DIR)
        """
        self.card_id = card_id
        base_dir = checkpoint_dir or self.CHECKPOINT_DIR
        self.checkpoint_path = Path(base_dir) / card_id
        
    def save_state(self, state: Dict[str, Any]):
        """
        Save the current agent state to a checkpoint file.

        Args:
            state: Dictionary containing state to be saved (expects metadata + metrics).
        """
        logger.info(f"Saving checkpoint to {self.checkpoint_path}")
        
        # Create checkpoint directory
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Expect nested structure: {metadata, metrics}
        metadata_dict = state["metadata"]
        metrics_dict = state["metrics"]
        
        # Extract metadata fields
        config = metadata_dict.get("config")
        game_id = metadata_dict.get("game_id")
        guid = metadata_dict.get("guid")
        max_actions = metadata_dict.get("max_actions")
        retry_attempts = metadata_dict.get("retry_attempts")
        num_plays = metadata_dict.get("num_plays")
        max_episode_actions = metadata_dict.get("max_episode_actions", 0)  # Backward compatibility
        action_counter = metadata_dict.get("action_counter")
        current_play = metadata_dict.get("current_play", 1)
        play_action_counter = metadata_dict.get("play_action_counter", 0)
        current_score = metadata_dict.get("current_score", 0)
        current_state = metadata_dict.get("current_state", "IN_PROGRESS")
        previous_score = metadata_dict.get("previous_score", 0)
        frame_grids = metadata_dict.get("frame_grids", [])
        available_actions = metadata_dict.get("available_actions", [])
        datastore = metadata_dict.get("datastore", {})
        
        # Extract metrics fields
        total_cost = metrics_dict.get("total_cost")
        total_usage = metrics_dict.get("total_usage")
        action_history = metrics_dict.get("action_history", [])
        
        # Save metadata
        metadata = {
            "card_id": self.card_id,
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
        
        # Save costs and usage
        costs = {
            "total_cost": total_cost.model_dump() if total_cost else {},
            "total_usage": total_usage.model_dump() if total_usage else {},
        }
        
        with open(self.checkpoint_path / "costs.json", "w") as f:
            json.dump(costs, f, indent=2)
        
        # Save action history
        action_history_data = [action.model_dump() for action in action_history]
        with open(self.checkpoint_path / "action_history.json", "w") as f:
            json.dump(action_history_data, f, indent=2)

        logger.info(f"Checkpoint saved successfully")
    
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
        
        # Load costs
        costs_path = self.checkpoint_path / "costs.json"
        if costs_path.exists():
            with open(costs_path) as f:
                costs_data = json.load(f)
                total_cost = Cost(**costs_data["total_cost"])
                total_usage = Usage(**costs_data["total_usage"])
        else:
            # Default values if costs file is missing
            total_cost = Cost(prompt_cost=0.0, completion_cost=0.0, total_cost=0.0)
            total_usage = Usage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                completion_tokens_details=CompletionTokensDetails()
            )
        
        # Load action history
        action_history = []
        action_history_path = self.checkpoint_path / "action_history.json"
        if action_history_path.exists():
            with open(action_history_path) as f:
                action_history_data = json.load(f)
                action_history = [GameActionRecord(**action) for action in action_history_data]
        
        logger.info(f"Checkpoint loaded successfully")

        return {
            "metadata": metadata,
            "metrics": {
                "total_cost": total_cost,
                "total_usage": total_usage,
                "action_history": action_history,
            },
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
        checkpoint_dir = Path(CheckpointManager.CHECKPOINT_DIR)
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
        checkpoint_path = Path(CheckpointManager.CHECKPOINT_DIR) / card_id
        metadata_path = checkpoint_path / "metadata.json"
        
        if not metadata_path.exists():
            return None
        
        with open(metadata_path) as f:
            return json.load(f)

