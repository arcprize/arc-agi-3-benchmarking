"""
Multimodal Agent for playing ARC-AGI-3 games.

Adapted from the original multimodal agent to use provider adapters.
"""
import json
import logging
import time
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from PIL import Image
from .adapters import create_provider
from .game_client import GameClient
from .utils.image import grid_to_image, image_to_base64, make_image_block, image_diff
from .prompts import PromptManager, PromptName, PromptSource
from .schemas import (
    GameResult,
    GameActionRecord,
    ActionData,
    Cost,
    Usage,
    CompletionTokensDetails
)
from .utils import load_hints, find_hints_file
from .utils.formatting import grid_to_text_matrix
from .utils.parsing import extract_json_from_response
from .checkpoint import CheckpointManager


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases for clarity
# ---------------------------------------------------------------------------

# Single frame grid (rows x cols of ints) and sequences of frames.
FrameGrid = List[List[int]]
FrameGridSequence = List[FrameGrid]

# Image sequences for a step.
FrameImageSequence = List[Image.Image]


# Map game actions to human-readable descriptions
HUMAN_ACTIONS = {
    "ACTION1": "Move Up",
    "ACTION2": "Move Down",
    "ACTION3": "Move Left",
    "ACTION4": "Move Right",
    "ACTION5": "Perform Action",
    "ACTION6": "Click object on screen (describe object and relative position)",
    "ACTION7": "Undo",
}


HUMAN_ACTIONS_LIST = list(HUMAN_ACTIONS.keys())


class MultimodalAgent:
    """Agent that plays ARC-AGI-3 games using multimodal LLMs"""
    def __init__(
        self,
        config: str,
        game_client: GameClient,
        card_id: str,
        max_actions: int = 40,
        retry_attempts: int = 3,
        num_plays: int = 1,
        max_episode_actions: int = 0,
        show_images: bool = False,
        use_vision: bool = True,
        memory_word_limit: Optional[int] = None,
        checkpoint_frequency: int = 1,
        checkpoint_card_id: Optional[str] = None,
        prompt_overrides: Optional[Dict[str, PromptSource]] = None,
    ):
        """
        Initialize the multimodal agent.

        Args:
            config: Model configuration name from models.yml
            game_client: GameClient for API communication
            card_id: Scorecard identifier for API calls
            max_actions: Maximum actions for entire run across all games/plays (0 = no limit)
            retry_attempts: Number of retry attempts for failed API calls
            num_plays: Number of times to play the game (0 = infinite, continues session with memory)
            max_episode_actions: Maximum actions per game/episode (0 = no limit)
            show_images: Whether to display game frames in the terminal
            use_vision: Whether to use vision (images) or text-only mode
            memory_word_limit: Maximum number of words allowed in memory scratchpad (default: from config or 500)
            checkpoint_frequency: Save checkpoint every N actions (default: 1, 0 to disable)
            checkpoint_card_id: Optional card_id for checkpoint directory (defaults to card_id if not provided)
        """
        self.config = config
        self.game_client = game_client
        self.card_id = card_id
        self.max_actions = max_actions
        self.retry_attempts = retry_attempts
        self.num_plays = num_plays
        self.max_episode_actions = max_episode_actions
        self.show_images = show_images
        
        # Initialize provider adapter (needed to access model config)
        self.provider = create_provider(config)
        
        # Set memory_word_limit: explicit parameter > model config > default (500)
        if memory_word_limit is not None:
            self.memory_word_limit = memory_word_limit
        else:
            self.memory_word_limit = self.provider.model_config.kwargs.get("memory_word_limit", 500)
        
        self.checkpoint_frequency = checkpoint_frequency

        # Prompt manager (handles default templates + optional overrides)
        self.prompt_manager = PromptManager(prompt_overrides)

        self.hints_file = find_hints_file()
        self.current_game_id: Optional[str] = None
        self.current_hint: Optional[str] = None
        
        # Vision support already determined from provider initialized earlier
        self._model_supports_vision = bool(getattr(self.provider.model_config, "is_multimodal", False))
        self._use_vision = bool(use_vision and self._model_supports_vision)

        if not self._model_supports_vision:
            if use_vision:
                logger.warning(
                    "Model config `%s` does not support multimodal; continuing without vision.",
                    self.config,
                )
            else:
                logger.info(
                    "Model config `%s` is text-only; vision disabled.",
                    self.config,
                )
        elif self._use_vision:
            logger.info("Vision is enabled for this agent. Images will be used.")
        else:
            logger.warning("Vision is disabled for this agent. Only text will be used.")
        
        # Tracking variables
        self.action_counter = 0
        self.total_cost = Cost(prompt_cost=0.0, completion_cost=0.0, total_cost=0.0)
        self.total_usage = Usage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            completion_tokens_details=CompletionTokensDetails()
        )
        self.action_history: List[GameActionRecord] = []
        
        # Memory for the agent
        self._available_actions: List[str] = []
        self._memory_prompt = ""
        self._previous_action: Optional[Dict[str, Any]] = None
        self._previous_images: FrameImageSequence = []
        self._previous_grids: FrameGridSequence = []  # Store raw grids for text-based providers
        self._current_grids: FrameGridSequence = []  # Current game state after last action; used for checkpoint restoration
        self._previous_score = 0

        self._previous_prompt = ""

        # Checkpoint manager - use checkpoint_card_id if provided, otherwise use card_id
        # This allows resuming from original checkpoint even when scorecard changes
        effective_checkpoint_id = checkpoint_card_id if checkpoint_card_id else card_id
        self.checkpoint_manager = CheckpointManager(effective_checkpoint_id)

        # Current play tracking (for checkpoint restoration)
        self._current_play = 1
        self._play_action_counter = 0
        self._current_guid: Optional[str] = None

    def _get_system_prompt(self) -> str:
        """
        Get the system prompt, prepending any hint for the current game if available.
        
        Returns:
            System prompt with hint prepended if available
        """
        # Base system prompt from templates
        system_prompt = self.prompt_manager.render(PromptName.SYSTEM)
        
        # Prepend hint if available for current game
        if self.current_hint:
            hint = self.current_hint.strip()
            if hint:
                system_prompt = f"{system_prompt}\n\n ALSO USE these hints in order to complete the game: \n {hint} "
                logger.info(f"Using hint for game {self.current_game_id}")
        
        return system_prompt
        
    def _get_memory_word_count(self) -> int:
        """Get the word count of the current memory"""
        return len(self._memory_prompt.split(" ")) if self._memory_prompt else 0
    
    def _compress_memory(self) -> str:
        """Ask LLM to compress memory if it exceeds the limit"""
        if not self._memory_prompt:
            return ""
        
        current_word_count = self._get_memory_word_count()
        compress_prompt = self.prompt_manager.render(
            PromptName.COMPRESS_MEMORY,
            {
                "current_word_count": current_word_count,
                "memory_limit": self.memory_word_limit,
                "memory_text": self._memory_prompt,
            },
        )
        
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {
                "role": "user",
                "content": compress_prompt,
            },
        ]
        
        try:
            response = self.provider.call_provider(messages)
            prompt_tokens, completion_tokens, reasoning_tokens = self.provider.extract_usage(response)
            self._update_costs(prompt_tokens, completion_tokens, reasoning_tokens)
            
            compressed = self.provider.extract_content(response).strip()
            compressed_word_count = len(compressed.split(" ")) if compressed else 0
            logger.info(f"Compressed memory from {current_word_count} to {compressed_word_count} words")
            return compressed
        except Exception as e:
            logger.error(f"Failed to compress memory: {e}")
            # Fallback to truncation
            return self._truncate_memory()
    
    def _truncate_memory(self) -> str:
        """Truncate memory to word limit by keeping the first N words"""
        if not self._memory_prompt:
            return ""
        
        words = self._memory_prompt.split(" ")
        if len(words) <= self.memory_word_limit:
            return self._memory_prompt
        
        truncated = " ".join(words[:self.memory_word_limit])
        logger.warning(f"Truncated memory from {len(words)} to {self.memory_word_limit} words")
        return truncated
    
    def _enforce_memory_limit(self):
        """Check memory size and compress or truncate if it exceeds the limit"""
        word_count = self._get_memory_word_count()
        if word_count <= self.memory_word_limit:
            return
        
        logger.info(f"Memory exceeds limit ({word_count} > {self.memory_word_limit} words). Attempting compression...")
        # Try compression first, fallback to truncation if it fails
        self._memory_prompt = self._compress_memory()
        
        # If compression didn't work or still exceeds limit, truncate
        if self._get_memory_word_count() > self.memory_word_limit:
            self._memory_prompt = self._truncate_memory()
    
    def _validate_action(self, action_name: str) -> bool:
        """
        Validate that action is in available actions set.
        
        Args:
            action_name: Action name like "ACTION1", "ACTION2", etc.
            
        Returns:
            True if action is valid, False otherwise
        """
        if not action_name or not action_name.startswith("ACTION"):
            logger.warning(f"Invalid action format: {action_name}")
            return False
        
        try:
            # Extract action number from ACTION1, ACTION2, etc.
            action_num = action_name.replace("ACTION", "")
            # Normalize available actions to string numbers for comparison
            normalized_available = {str(a) for a in self._available_actions}
            is_valid = action_num in normalized_available
            
            if not is_valid:
                logger.warning(
                    f"Action {action_name} (number {action_num}) not in available actions: "
                    f"{sorted(normalized_available)}"
                )
            
            return is_valid
        except Exception as e:
            logger.error(f"Error validating action {action_name}: {e}")
            return False

    def save_checkpoint(self):
        """Save current state to checkpoint"""
        if self.checkpoint_frequency == 0:
            return
            
        try:
            state = self.get_state()
            self.checkpoint_manager.save_state(state)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}", exc_info=True)

    def restore_from_checkpoint(self):
        """Restore agent state from checkpoint"""
        logger.info(f"Restoring agent state from checkpoint: {self.checkpoint_manager.card_id}")

        try:
            state = self.checkpoint_manager.load_state()

            # Restore metadata
            metadata = state["metadata"]
            self.current_game_id = metadata["game_id"]
            self._current_guid = metadata.get("guid")
            self.max_actions = metadata["max_actions"]
            self.retry_attempts = metadata["retry_attempts"]
            self.num_plays = metadata["num_plays"]
            self.max_episode_actions = metadata.get("max_episode_actions", 0)  # Backward compatibility
            self.action_counter = metadata["action_counter"]
            self._current_play = metadata.get("current_play", 1)
            self._play_action_counter = metadata.get("play_action_counter", 0)
            self._previous_score = metadata.get("previous_score", 0)

            # Restore costs and usage
            self.total_cost = state["total_cost"]
            self.total_usage = state["total_usage"]

            # Restore action history
            self.action_history = state["action_history"]

            # Restore memory and state
            self._memory_prompt = state["memory_prompt"]
            self._previous_action = state["previous_action"]
            self._previous_images = state["previous_images"]
            self._previous_grids = state.get("previous_grids", [])
            self._current_grids = state.get("current_grids", [])
            self._available_actions = state.get("available_actions", [])

            logger.info(
                f"Restored checkpoint: game_id={self.current_game_id}, "
                f"action_counter={self.action_counter}, "
                f"play={self._current_play}/{self.num_plays}, "
                f"guid={self._current_guid}"
            )

            return True
        except Exception as e:
            logger.error(f"Failed to restore checkpoint: {e}", exc_info=True)
            return False

    def _update_costs(self, prompt_tokens: int, completion_tokens: int, reasoning_tokens: int = 0):
        """Update cost and usage tracking"""
        # Get pricing from model config
        input_cost_per_token = self.provider.model_config.pricing.input / 1_000_000
        output_cost_per_token = self.provider.model_config.pricing.output / 1_000_000
        
        prompt_cost = prompt_tokens * input_cost_per_token
        completion_cost = completion_tokens * output_cost_per_token
        
        # Reasoning tokens are billed at the output rate but tracked separately
        reasoning_cost = reasoning_tokens * output_cost_per_token if reasoning_tokens > 0 else 0.0
        
        self.total_cost.prompt_cost += prompt_cost
        self.total_cost.completion_cost += completion_cost
        if reasoning_tokens > 0:
            if self.total_cost.reasoning_cost is None:
                self.total_cost.reasoning_cost = 0.0
            self.total_cost.reasoning_cost += reasoning_cost
        self.total_cost.total_cost += prompt_cost + completion_cost
        
        self.total_usage.prompt_tokens += prompt_tokens
        self.total_usage.completion_tokens += completion_tokens
        self.total_usage.total_tokens += prompt_tokens + completion_tokens
        
        # Update reasoning token details
        if reasoning_tokens > 0:
            if not self.total_usage.completion_tokens_details:
                self.total_usage.completion_tokens_details = CompletionTokensDetails()
            self.total_usage.completion_tokens_details.reasoning_tokens += reasoning_tokens

    # ------------------------------------------------------------------
    # Substep methods – primary extension points for subclasses
    # ------------------------------------------------------------------

    def analyze_outcome_step(
        self,
        current_frame_images: FrameImageSequence,
        current_frame_grids: FrameGridSequence,
        current_score: int,
    ) -> str:
        """
        Substep: Analyze the outcome of the previous action and update memory.

        Default implementation:
          - Compares the new score to the previous score.
          - Builds the analysis prompt (including memory and hints).
          - Calls the provider to get analysis and updated memory.
          - Enforces the memory size limit.

        Subclasses can override this method to change analysis behavior
        (e.g. different prompts, memory strategies) while keeping the
        surrounding workflow intact.
        """
        if not self._previous_action:
            return "no previous action"

        level_complete = ""
        if current_score > self._previous_score:
            level_complete = "NEW LEVEL!!!! - Whatever you did must have been good!"

        analyze_instruct = self.prompt_manager.render(
            PromptName.ANALYZE_INSTRUCT,
            {"memory_limit": self.memory_word_limit},
        )
        analyze_prompt = f"{level_complete}\n\n{analyze_instruct}\n\n{self._memory_prompt}"
        if self._model_supports_vision and self._use_vision:
            # For multimodal providers, use images
            all_imgs = [
                self._previous_images[-1],
                *current_frame_images,
                image_diff(self._previous_images[-1], current_frame_images[-1]),
            ]

            # Build message with images
            msg_parts = [
                make_image_block(image_to_base64(img))
                for img in all_imgs
            ] + [{"type": "text", "text": analyze_prompt}]
        else:
            # For text-only providers, use text matrices
            msg_parts = []

            # Previous frame
            msg_parts.append({
                "type": "text",
                "text": f"Frame 0 (before action):\n{grid_to_text_matrix(self._previous_grids[-1])}"
            })

            # Current frames
            for i, grid in enumerate(current_frame_grids):
                msg_parts.append({
                    "type": "text",
                    "text": f"Frame {i+1} (after action):\n{grid_to_text_matrix(grid)}"
                })

            # Add the prompt
            msg_parts.append({"type": "text", "text": analyze_prompt})

        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {
                "role": "user",
                "content": [{"type": "text", "text": self._previous_prompt}],
            },
            {
                "role": "assistant",
                "content": f"```json\n{json.dumps(self._previous_action)}\n```",
            },
            {
                "role": "user",
                "content": msg_parts,
            },
        ]

        response = self.provider.call_provider(messages)

        # Track costs - handle different response formats
        prompt_tokens, completion_tokens, reasoning_tokens = self.provider.extract_usage(response)
        self._update_costs(prompt_tokens, completion_tokens, reasoning_tokens)

        # Extract analysis and update memory
        analysis_message = self.provider.extract_content(response)
        logger.info(f"Analysis: {analysis_message[:200]}...")
        before, _, after = analysis_message.partition("---")
        analysis = before.strip()
        if after.strip():
            self._memory_prompt = after.strip()
            word_count = self._get_memory_word_count()
            logger.info(f"Memory updated ({word_count} words):\n{self._memory_prompt}")
            # Enforce memory word limit
            self._enforce_memory_limit()
            # Log memory again after enforcement (in case it was compressed/truncated)
            final_word_count = self._get_memory_word_count()
            if final_word_count != word_count:
                logger.info(f"Memory after enforcement ({final_word_count} words):\n{self._memory_prompt}")
        return analysis

    def decide_human_action_step(
        self,
        frame_images: FrameImageSequence,
        frame_grids: FrameGridSequence,
        analysis: str,
    ) -> Dict[str, Any]:
        """
        Substep: Decide the next human-level action given the current frames
        and the analysis of the previous step.

        Default implementation:
          - Builds a human-readable list of available actions and example actions.
          - Injects those into the ACTION_INSTRUCT prompt via placeholders.
          - Builds a prompt that includes the current analysis and memory.
          - Calls the provider to get a JSON `human_action` description.

        Override this to change how the agent chooses high-level actions.
        """
        # Format available actions for the prompt (with fallback)
        if self._available_actions:
            # Normalize action indices to integers (handles "6" and 6)
            indices = [int(str(a)) for a in self._available_actions]
            action_descriptions = [
                HUMAN_ACTIONS[HUMAN_ACTIONS_LIST[i - 1]]
                for i in indices
            ]
        else:
            # Fallback to all actions (shouldn't happen)
            action_descriptions = list(HUMAN_ACTIONS.values())

        # Build the bullet list shown to the model
        available_actions_list = "\n".join(f"  • {desc}" for desc in action_descriptions)

        # Prepare example actions for the prompt (up to three examples)
        example_actions = ", ".join(f'"{desc}"' for desc in action_descriptions[:3])
        json_example_action = f'"{action_descriptions[0]}"' if action_descriptions else "Move Up"

        action_instruct = self.prompt_manager.render(
            PromptName.ACTION_INSTRUCT,
            {
                "available_actions_list": available_actions_list,
                "example_actions": example_actions,
                "json_example_action": json_example_action,
            },
        )

        if len(analysis) > 20:
            self._previous_prompt = f"{analysis}\n\n{self._memory_prompt}\n\n{action_instruct}"
        else:
            self._previous_prompt = f"{self._memory_prompt}\n\n{action_instruct}"
        if self._model_supports_vision and self._use_vision:
            # For multimodal providers, use images
            content = [
                *[make_image_block(image_to_base64(img)) for img in frame_images],
            ]
        else:
            # For text-only providers, use text matrices
            content = []
            for i, grid in enumerate(frame_grids):
                content.append({
                    "type": "text",
                    "text": f"Frame {i}:\n{grid_to_text_matrix(grid)}"
                })
        content.append({"type": "text", "text": self._previous_prompt})

        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {
                "role": "user",
                "content": content,
            },
        ]

        response = self.provider.call_provider(messages)

        # Track costs
        prompt_tokens, completion_tokens, reasoning_tokens = self.provider.extract_usage(response)
        self._update_costs(prompt_tokens, completion_tokens, reasoning_tokens)

        action_message = self.provider.extract_content(response)

        logger.info(f"Human action: {action_message[:200]}...")

        try:
            return extract_json_from_response(action_message)
        except ValueError as e:
            logger.error(f"Failed to extract JSON from response: {e}")
            logger.debug(f"Full response: {action_message}")
            # Re-raise to be caught by game loop
            raise

    def convert_human_to_game_action_step(
        self,
        human_action: str,
        last_frame_image: Image.Image,
        last_frame_grid: FrameGrid,
    ) -> Dict[str, Any]:
        """
        Substep: Convert a natural-language human action into a concrete
        game action dictionary suitable for execution.

        Default implementation:
          - Builds a prompt that lists available actions and the desired action.
          - Includes the set of valid action codes allowed.
          - Calls the provider to obtain a concrete game action JSON.

        This is the most common substep to override when mapping decisions into
        different action spaces.
        """
        # Format available actions and valid codes for the prompt
        if self._available_actions:
            indices = [int(str(a)) for a in self._available_actions]
            available_actions_text = "\n".join(
                f"{HUMAN_ACTIONS_LIST[i - 1]} = {HUMAN_ACTIONS[HUMAN_ACTIONS_LIST[i - 1]]}"
                for i in indices
            )
            valid_actions = ", ".join(HUMAN_ACTIONS_LIST[i - 1] for i in indices)
        else:
            # Fallback to all actions
            available_actions_text = "\n".join(
                f"{name} = {desc}" for name, desc in HUMAN_ACTIONS.items()
            )
            valid_actions = ", ".join(HUMAN_ACTIONS_LIST)

        find_action_instruct = self.prompt_manager.render(
            PromptName.FIND_ACTION_INSTRUCT,
            {"action_list": available_actions_text, "valid_actions": valid_actions},
        )

        content = []
        if self._model_supports_vision and self._use_vision:
            # For multimodal providers, use image
            content.append(
                make_image_block(image_to_base64(last_frame_image)),
            )
        else:
            # For text-only providers, use text matrix
            content.append(
                {
                    "type": "text",
                    "text": f"Current frame:\n{grid_to_text_matrix(last_frame_grid)}"
                },
            )
        content.append({
            "type": "text",
            "text": human_action + "\n\n" + find_action_instruct,
        })

        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {
                "role": "user",
                "content": content,
            },
        ]

        response = self.provider.call_provider(messages)

        # Track costs
        prompt_tokens, completion_tokens, reasoning_tokens = self.provider.extract_usage(response)
        self._update_costs(prompt_tokens, completion_tokens, reasoning_tokens)

        action_message = self.provider.extract_content(response)
        logger.info(f"Game action: {action_message[:200]}...")

        try:
            return extract_json_from_response(action_message)
        except ValueError as e:
            logger.error(f"Failed to extract JSON from game action response: {e}")
            logger.debug(f"Full response: {action_message}")
            raise
    
    def step(
        self,
        frame_images: FrameImageSequence,
        frame_grids: FrameGridSequence,
        current_score: int
    ) -> Dict[str, Any]:
        """
        Perform one cognitive step: Analyze -> Decide -> Convert.
        
        Args:
            frame_images: List of images for the current state
            frame_grids: List of grid matrices for the current state
            current_score: Current game score
            
        Returns:
            Game action dictionary ready for execution
        """
        # 1. Analyze previous action results
        analysis = self.analyze_outcome_step(
            frame_images, frame_grids, current_score
        )
        
        # 2. Decide on next human action
        human_action_dict = self.decide_human_action_step(
            frame_images, frame_grids, analysis
        )
        human_action = human_action_dict.get("human_action")
        
        if not human_action:
            raise ValueError("No human_action in response")
            
        # 3. Convert to game action
        game_action_dict = self.convert_human_to_game_action_step(
            human_action, frame_images[-1], frame_grids[-1]
        )
        
        # Merge reasoning from human action into game action for tracking
        game_action_dict["human_action_dict"] = human_action_dict
        game_action_dict["analysis"] = analysis
        
        return game_action_dict

    
    def _execute_game_action(
        self,
        action_name: str,
        action_data: Optional[Dict[str, Any]],
        game_id: str,
        guid: Optional[str],
        reasoning: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute action via game client"""
        data = {"game_id": game_id}
        if guid:
            data["guid"] = guid
        if action_data:
            data.update(action_data)
        if reasoning:
            data["reasoning"] = reasoning
        
        return self.game_client.execute_action(action_name, data)
    
    def get_state(self) -> Dict[str, Any]:
        """Returns serializable agent state for checkpointing."""
        return {
            "metadata": {
                "config": self.config,
                "game_id": self.current_game_id,
                "guid": self._current_guid,
                "max_actions": self.max_actions,
                "retry_attempts": self.retry_attempts,
                "num_plays": self.num_plays,
                "max_episode_actions": self.max_episode_actions,
                "action_counter": self.action_counter,
                "current_play": self._current_play,
                "play_action_counter": self._play_action_counter,
                "previous_score": self._previous_score,
            },
            "memory": {
                "prompt": self._memory_prompt,
                "previous_action": self._previous_action,
                "previous_images": self._previous_images,
                "previous_grids": self._previous_grids,
                "current_grids": self._current_grids,
                "available_actions": self._available_actions,
            },
            "metrics": {
                "total_cost": self.total_cost,
                "total_usage": self.total_usage,
                "action_history": self.action_history,
            }
        }

    def play_game(self, game_id: str, resume_from_checkpoint: bool = False) -> GameResult:
        """
        Play a complete game and return results.

        Args:
            game_id: Game identifier to play
            resume_from_checkpoint: If True, resume from existing checkpoint

        Returns:
            GameResult with complete game information (best result if multiple plays)
        """
        # Restore from checkpoint if requested
        if resume_from_checkpoint:
            if not self.checkpoint_manager.checkpoint_exists():
                logger.warning(f"No checkpoint found for {self.checkpoint_manager.card_id}, starting fresh")
                resume_from_checkpoint = False
            else:
                if not self.restore_from_checkpoint():
                    logger.error("Failed to restore checkpoint, starting fresh")
                    resume_from_checkpoint = False
                else:
                    # Use the restored game_id if available
                    if self.current_game_id:
                        game_id = self.current_game_id
                        logger.info(f"Resuming game {game_id} from checkpoint")

        # Store current game ID
        self.current_game_id = game_id

        logger.info(f"Starting game {game_id} with config {self.config} ({self.num_plays} play(s))")
        overall_start_time = time.time()

        # Load hint for this specific game if available (only if not resuming, since hint is already in memory)
        if not resume_from_checkpoint:
            if self.hints_file:
                hints = load_hints(self.hints_file, game_id=game_id)
                self.current_hint = hints.get(game_id) if hints else None
                if self.current_hint:
                    logger.info(f"Found hint for game {game_id}")
                else:
                    logger.debug(f"No hint found for game {game_id}")
            else:
                self.current_hint = None

        best_result: Optional[GameResult] = None
        guid: Optional[str] = self._current_guid if resume_from_checkpoint else None

        # Determine starting play number
        start_play = self._current_play if resume_from_checkpoint else 1
        play_num = start_play

        while True:
            # Check if we should continue based on num_plays
            # If num_plays == 0, continue indefinitely (until max_actions or WIN)
            # If num_plays > 0, continue while play_num <= num_plays
            if self.num_plays > 0 and play_num > self.num_plays:
                break

            # Check global max_actions limit before starting new play
            if self.max_actions > 0 and self.action_counter >= self.max_actions:
                logger.info(f"Global max_actions ({self.max_actions}) reached. Stopping.")
                break

            self._current_play = play_num
            play_start_time = time.time()

            # Log play start message
            if play_num > 1:
                if self.num_plays == 0:
                    logger.info(f"Starting play {play_num}")
                else:
                    logger.info(f"Starting play {play_num}/{self.num_plays}")

            # Initialize session state
            session_restored = False
            state = {}

            # Skip reset if resuming from checkpoint in the middle of a play
            if resume_from_checkpoint and play_num == start_play and self._play_action_counter > 0:
                logger.info(f"Resuming play {play_num} at action {self._play_action_counter}")

                if self._current_guid:
                    guid = self._current_guid
                    current_score = self._previous_score
                    current_state = "IN_PROGRESS"
                    session_restored = True
                    # Create a minimal state structure using previous grids
                    state = {
                        "guid": guid,
                        "score": current_score,
                        "state": current_state,
                        "frame": self._current_grids if self._current_grids else []
                    }
                    logger.info(f"Continuing session with guid: {guid}, score: {current_score}")

                if not session_restored:
                    logger.info("No GUID found, starting new game session with restored memory...")
                    state = self.game_client.reset_game(self.card_id, game_id, guid=None)
                    guid = state.get("guid")
                    current_score = state.get("score", 0)
                    current_state = state.get("state", "IN_PROGRESS")

                    # This reset is not the "first ever" reset for this session (we are resuming),
                    # so count it like the server does.
                    self.action_counter += 1
                    self.action_history.append(
                        GameActionRecord(
                            action_num=self.action_counter,
                            action="RESET",
                            action_data=None,
                            reasoning={"system": "reset_game (checkpoint recovery)"},
                            result_score=current_score,
                            result_state=current_state,
                            cost=Cost(prompt_cost=0.0, completion_cost=0.0, reasoning_cost=0.0, total_cost=0.0),
                        )
                    )

                # If we had to re-reset (checkpoint recovery), that RESET counts as the first action
                # of this play; otherwise keep the restored counter.
                play_action_counter = self._play_action_counter if session_restored else 1
                resume_from_checkpoint = False
            else:
                # Reset game
                state = self.game_client.reset_game(self.card_id, game_id, guid=guid)
                guid = state.get("guid")
                current_score = state.get("score", 0)
                current_state = state.get("state", "IN_PROGRESS")

                # Initialize memory on first play
                if play_num == 1 and not self._memory_prompt:
                    self._available_actions = state.get("available_actions", list(HUMAN_ACTIONS.keys()))
                    self._memory_prompt = ""  # Initialize memory as empty

                # Per the server: the very first RESET of a fresh run (play 1) is free and does
                # not count as an action; any other RESET counts.
                count_reset = resume_from_checkpoint or play_num > 1
                if count_reset:
                    self.action_counter += 1
                    self.action_history.append(
                        GameActionRecord(
                            action_num=self.action_counter,
                            action="RESET",
                            action_data=None,
                            reasoning={"system": f"reset_game (start play {play_num})"},
                            result_score=current_score,
                            result_state=current_state,
                            cost=Cost(prompt_cost=0.0, completion_cost=0.0, reasoning_cost=0.0, total_cost=0.0),
                        )
                    )
                    play_action_counter = 1
                else:
                    play_action_counter = 0

            # Store guid
            self._current_guid = guid
            self._play_action_counter = play_action_counter

            # --- Run Game Session ---
            # We pass the initial state to the session runner
            session_result = self._run_session_loop(
                game_id=game_id,
                initial_state=state,
                play_num=play_num,
                start_action_counter=play_action_counter
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
                total_cost=self.total_cost,
                usage=self.total_usage,
                actions=play_action_history,
                final_memory=self._memory_prompt,
                timestamp=datetime.now(timezone.utc),
                scorecard_url=scorecard_url,
                card_id=self.card_id
            )

            # Log play completion message
            if self.num_plays == 0:
                logger.info(
                    f"Play {play_num} completed: {current_state}, "
                    f"Score: {current_score}, Actions: {play_action_counter}"
                )
            else:
                logger.info(
                    f"Play {play_num}/{self.num_plays} completed: {current_state}, "
                    f"Score: {current_score}, Actions: {play_action_counter}"
                )

            # Track best result
            if best_result is None:
                best_result = play_result
            elif current_state == "WIN" and best_result.final_state != "WIN":
                best_result = play_result
            elif current_state == "WIN" and best_result.final_state == "WIN":
                if current_score > best_result.final_score:
                    best_result = play_result
            elif current_score > best_result.final_score:
                best_result = play_result

            # Save checkpoint after play
            self.save_checkpoint()

            if current_state == "WIN":
                logger.info(f"Game won on play {play_num}! Stopping early.")
                break

            # Check global max_actions limit after play
            if self.max_actions > 0 and self.action_counter >= self.max_actions:
                logger.info(f"Global max_actions ({self.max_actions}) reached. Stopping.")
                break

            # Log continuation message (only for finite plays that aren't the last)
            if self.num_plays > 0 and play_num < self.num_plays:
                logger.info(f"Play {play_num} ended ({current_state}). Continuing to next play...")
            elif self.num_plays == 0:
                logger.info(f"Play {play_num} ended ({current_state}). Continuing to next play...")

            play_num += 1
        
        overall_duration = time.time() - overall_start_time
        
        # Update best result with overall stats
        best_result.actions_taken = self.action_counter
        best_result.duration_seconds = overall_duration
        
        logger.info(
            f"All plays completed. Best: {best_result.final_state}, "
            f"Score: {best_result.final_score}, Total Actions: {self.action_counter}, "
            f"Cost: ${self.total_cost.total_cost:.4f}"
        )

        if best_result.final_state == "WIN":
            logger.info("Game won, deleting checkpoint")
            self.checkpoint_manager.delete_checkpoint()

        return best_result

    def _run_session_loop(
        self,
        game_id: str,
        initial_state: Dict[str, Any],
        play_num: int,
        start_action_counter: int
    ) -> Dict[str, Any]:
        """
        Run the inner game loop for a single session.
        """
        state = initial_state
        guid = state.get("guid")
        current_score = state.get("score", 0)
        current_state = state.get("state", "IN_PROGRESS")
        play_action_counter = start_action_counter
        
        play_action_history: List[GameActionRecord] = []
        
        # Reconstruct history if resuming
        if guid and play_action_counter > 0:
            start_action_num = self.action_counter - play_action_counter + 1
            end_action_num = self.action_counter
            play_action_history = [
                action for action in self.action_history
                if start_action_num <= action.action_num <= end_action_num
            ]
        
        while (
            current_state not in ["WIN", "GAME_OVER"]
            and (self.max_episode_actions == 0 or play_action_counter < self.max_episode_actions)
            and (self.max_actions == 0 or self.action_counter < self.max_actions)
        ):
            try:
                frames = state.get("frame", [])
                if not frames:
                    logger.warning("No frames in state, breaking")
                    break
                
                # Store raw grids and convert to images
                frame_grids = frames
                frame_images = [grid_to_image(frame) for frame in frames]
                
                # Track cost before this action to calculate per-action cost
                cost_before = Cost(
                    prompt_cost=self.total_cost.prompt_cost,
                    completion_cost=self.total_cost.completion_cost,
                    reasoning_cost=self.total_cost.reasoning_cost,
                    total_cost=self.total_cost.total_cost
                )
                
                # A single step of the game
                try:
                    game_action_dict = self.step(frame_images, frame_grids, current_score)
                except ValueError as e:
                    logger.error(f"Step failed: {e}")
                    break
                
                action_name = game_action_dict.get("action")
                if not action_name:
                    logger.error("No action name in response")
                    break
                
                # Extract action data
                action_data_dict = {}
                if action_name == "ACTION6":
                    x = game_action_dict.get("x", 0)
                    y = game_action_dict.get("y", 0)
                    action_data_dict = {
                        "x": max(0, min(x, 127)) // 2,
                        "y": max(0, min(y, 127)) // 2,
                    }
                
                # Prepare reasoning for API
                human_action_dict = game_action_dict.get("human_action_dict", {})
                analysis = game_action_dict.get("analysis", "")
                human_action = human_action_dict.get("human_action", "")
                
                action_field = action_name
                if action_name == "ACTION6" and action_data_dict:
                    action_field = f"{action_name}: [{action_data_dict}]"

                reasoning_for_api = {
                    "analysis": analysis[:1000] if len(analysis) > 1000 else analysis,
                    "action": action_field,
                    "human_action": human_action,
                    "reasoning": (human_action_dict.get("reasoning", "") or "")[:300],
                    "expected": (human_action_dict.get("expected_result", "") or "")[:300],
                    "tokens:": [self.total_usage.prompt_tokens, self.total_usage.completion_tokens],
                }
                
                # Execute action - this returns the NEW state after the action
                state = self._execute_game_action(action_name, action_data_dict, game_id, guid, reasoning_for_api)
                guid = state.get("guid", guid)
                new_score = state.get("score", current_score)
                current_state = state.get("state", "IN_PROGRESS")
                
                # Calculate cost for this action only
                action_cost = Cost(
                    prompt_cost=self.total_cost.prompt_cost - cost_before.prompt_cost,
                    completion_cost=self.total_cost.completion_cost - cost_before.completion_cost,
                    reasoning_cost=(self.total_cost.reasoning_cost or 0) - (cost_before.reasoning_cost or 0),
                    total_cost=self.total_cost.total_cost - cost_before.total_cost
                )
                
                # Update tracking
                self.action_counter += 1
                action_record = GameActionRecord(
                    action_num=self.action_counter,
                    action=action_name,
                    action_data=ActionData(**action_data_dict) if action_data_dict else None,
                    reasoning={
                        "human_action": human_action,
                        "reasoning": human_action_dict.get("reasoning", ""),
                        "expected": human_action_dict.get("expected_result", ""),
                        "analysis": analysis[:500] if len(analysis) > 500 else analysis,
                    },
                    result_score=new_score,
                    result_state=current_state,
                    cost=action_cost,
                )
                play_action_history.append(action_record)
                self.action_history.append(action_record)
                
                # Update previous state for the NEXT iteration's analysis
                # The next iteration will compare:
                #   - self._previous_images/grids (current frames, after THIS action)
                #   - vs next_frames (frames after NEXT action)
                # This way each analysis compares before/after for one action
                self._previous_action = human_action_dict
                self._previous_images = frame_images
                self._previous_grids = frame_grids
                self._previous_score = current_score
                
                # Track current state: _previous_grids holds frames BEFORE this action,
                # _current_grids holds frames AFTER this action (for checkpoint save/restore)
                self._current_grids = state.get("frame", [])
                
                # Update current variables for next iteration
                current_score = new_score
                play_action_counter += 1
                self._play_action_counter = play_action_counter
                self._current_guid = guid

                logger.info(
                    f"Play {play_num}, Action {play_action_counter}: {action_name}, "
                    f"Score: {current_score}, State: {current_state}"
                )

                # Check limits after incrementing counters
                if self.max_actions > 0 and self.action_counter >= self.max_actions:
                    logger.info(f"Global max_actions ({self.max_actions}) reached. Stopping session.")
                    break
                if self.max_episode_actions > 0 and play_action_counter >= self.max_episode_actions:
                    logger.info(f"Episode max_episode_actions ({self.max_episode_actions}) reached. Stopping session.")
                    break

                # Save checkpoint periodically
                if self.checkpoint_frequency > 0 and play_action_counter % self.checkpoint_frequency == 0:
                    logger.info(f"Saving checkpoint at action {play_action_counter}")
                    self.save_checkpoint()

            except Exception as e:
                logger.error(f"Error during game loop: {e}", exc_info=True)
                break
        
        return {
            "score": current_score,
            "state": current_state,
            "actions_taken": play_action_counter,
            "action_history": play_action_history
        }

