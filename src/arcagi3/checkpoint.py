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

from PIL import Image

from .schemas import CompletionTokensDetails, Cost, GameActionRecord, Usage

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpointing of agent state"""
    
    CHECKPOINT_DIR = ".checkpoint"
    
    def __init__(self, card_id: str):
        """
        Initialize checkpoint manager.
        
        Args:
            card_id: Scorecard ID to use as checkpoint directory name
        """
        self.card_id = card_id
        self.checkpoint_path = Path(self.CHECKPOINT_DIR) / card_id
        
    def save_state(self, state: Dict[str, Any]):
        """
        Save the current agent state to a checkpoint file.

        Args:
            state: Dictionary containing all state to be saved (with metadata, memory, metrics keys)
        """
        logger.info(f"Saving checkpoint to {self.checkpoint_path}")
        
        # Create checkpoint directory
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Extract components from nested structure
        metadata_dict = state.get("metadata", {})
        memory_dict = state.get("memory", {})
        metrics_dict = state.get("metrics", {})
        
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
        previous_score = metadata_dict.get("previous_score", 0)
        
        # Extract memory fields
        memory_prompt = memory_dict.get("prompt", "")
        previous_action = memory_dict.get("previous_action")
        previous_images = memory_dict.get("previous_images", [])
        previous_grids = memory_dict.get("previous_grids")
        current_grids = memory_dict.get("current_grids")
        available_actions = memory_dict.get("available_actions", [])
        
        # Extract metrics fields
        total_cost = metrics_dict.get("total_cost")
        total_usage = metrics_dict.get("total_usage")
        action_history = metrics_dict.get("action_history", [])
        
        # Determine use_vision (not in current get_state, but needed for backward compat)
        use_vision = len(previous_images) > 0

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
            "previous_score": previous_score,
            "use_vision": use_vision,
            "checkpoint_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
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
        
        # Save memory
        with open(self.checkpoint_path / "memory.txt", "w") as f:
            f.write(memory_prompt if memory_prompt else "")
        
        # Save previous action
        if previous_action:
            with open(self.checkpoint_path / "previous_action.json", "w") as f:
                json.dump(previous_action, f, indent=2)
        
        # Save previous images
        # Clear existing images directory to prevent stale files from accumulating
        # (e.g., if previous checkpoint had 3 frames but current has 1)
        images_dir = self.checkpoint_path / "previous_images"
        if images_dir.exists():
            shutil.rmtree(images_dir)
        
        if previous_images:
            images_dir.mkdir(exist_ok=True)
            for i, img in enumerate(previous_images):
                img.save(images_dir / f"frame_{i}.png")

        # Save previous grids (for text-only mode)
        if previous_grids:
            with open(self.checkpoint_path / "previous_grids.json", "w") as f:
                json.dump(previous_grids, f)

        # Save current grids (for resuming with correct state)
        if current_grids:
            with open(self.checkpoint_path / "current_grids.json", "w") as f:
                json.dump(current_grids, f)

        # Save available actions
        if available_actions:
            with open(self.checkpoint_path / "available_actions.json", "w") as f:
                json.dump(available_actions, f)

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
        
        # Load memory
        memory_prompt = ""
        memory_path = self.checkpoint_path / "memory.txt"
        if memory_path.exists():
            with open(memory_path) as f:
                memory_prompt = f.read()
        
        # Load previous action
        previous_action = None
        previous_action_path = self.checkpoint_path / "previous_action.json"
        if previous_action_path.exists():
            with open(previous_action_path) as f:
                previous_action = json.load(f)
        
        # Load previous images
        previous_images = []
        images_dir = self.checkpoint_path / "previous_images"
        if images_dir.exists():
            image_files = sorted(images_dir.glob("frame_*.png"), key=lambda p: int(p.stem.split("_")[1]))
            for img_path in image_files:
                with Image.open(img_path) as img:
                    # Create a copy to fully decouple from file handle
                    previous_images.append(img.copy())

        # Load previous grids (for text-only mode)
        previous_grids = []
        grids_path = self.checkpoint_path / "previous_grids.json"
        if grids_path.exists():
            with open(grids_path) as f:
                previous_grids = json.load(f)

        # Load current grids (for resuming)
        current_grids = []
        current_grids_path = self.checkpoint_path / "current_grids.json"
        if current_grids_path.exists():
            with open(current_grids_path) as f:
                current_grids = json.load(f)

        # Load available actions
        available_actions = []
        available_actions_path = self.checkpoint_path / "available_actions.json"
        if available_actions_path.exists():
            with open(available_actions_path) as f:
                available_actions = json.load(f)

        logger.info(f"Checkpoint loaded successfully")

        return {
            "metadata": metadata,
            "total_cost": total_cost,
            "total_usage": total_usage,
            "action_history": action_history,
            "memory_prompt": memory_prompt,
            "previous_action": previous_action,
            "previous_images": previous_images,
            "previous_grids": previous_grids,
            "current_grids": current_grids,
            "available_actions": available_actions,
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

