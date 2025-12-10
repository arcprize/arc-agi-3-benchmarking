import logging
from typing import Optional
from arcagi3.utils import read_models_config
from arcagi3.game_client import GameClient
from arcagi3.utils import generate_scorecard_tags
from arcagi3.agent import MultimodalAgent
from arcagi3.checkpoint import CheckpointManager
from arcagi3.schemas import GameResult
from arcagi3.utils import save_result
import uuid

logger = logging.getLogger(__name__)

class ARC3Tester:
    """Main tester class for running ARC-AGI-3 games"""
    
    def __init__(
        self,
        config: str,
        save_results_dir: Optional[str] = None,
        overwrite_results: bool = False,
        max_actions: int = 40,
        retry_attempts: int = 3,
        api_retries: int = 3,
        num_plays: int = 1,
        show_images: bool = False,
        use_vision: bool = True,
        checkpoint_frequency: int = 1,
        close_on_exit: bool = False,
        memory_word_limit: Optional[int] = None,
        submit_scorecard: bool = True,
    ):
        """
        Initialize the tester.

        Args:
            config: Model configuration name from models.yml
            save_results_dir: Directory to save results (None to skip saving)
            overwrite_results: Whether to overwrite existing results
            max_actions: Maximum actions per game
            retry_attempts: Number of retry attempts for API failures
            api_retries: Number of retry attempts for ARC-AGI-3 API calls
            num_plays: Number of times to play the game (continues session with memory)
            show_images: Whether to display game frames in the terminal
            use_vision: Whether to use vision (images) or text-only mode
            checkpoint_frequency: Save checkpoint every N actions (default: 1, 0 to disable)
            close_on_exit: Close scorecard on exit even if not won (prevents checkpoint resume)
            memory_word_limit: Memory scratchpad word limit (overrides model config, default: from config or 500)
        """
        self.config = config
        self.model_config = read_models_config(config)
        self.save_results_dir = save_results_dir
        self.overwrite_results = overwrite_results
        self.max_actions = max_actions
        self.retry_attempts = retry_attempts
        self.num_plays = num_plays
        self.show_images = show_images
        self.use_vision = use_vision
        self.checkpoint_frequency = checkpoint_frequency
        self.close_on_exit = close_on_exit
        self.submit_scorecard = submit_scorecard
        
        # Determine memory limit: CLI > Config > Default (500)
        if memory_word_limit is not None:
            self.memory_word_limit = memory_word_limit
        else:
            # Check if defined in model config kwargs
            self.memory_word_limit = self.model_config.kwargs.get("memory_word_limit", 500)
            
        # Initialize game client
        self.game_client = GameClient(max_retries=api_retries)
        
        logger.info(f"Initialized ARC3Tester with config: {config}")
        logger.info(f"Model: {self.model_config.model_name}, Provider: {self.model_config.provider}")
    
    def play_game(self, game_id: str, card_id: Optional[str] = None, resume_from_checkpoint: bool = False) -> GameResult:
        """
        Play a single game.

        Args:
            game_id: Game identifier
            card_id: Optional scorecard ID (generated if not provided, or loaded from checkpoint)
            resume_from_checkpoint: If True, resume from existing checkpoint

        Returns:
            GameResult with complete game information
        """
        # If resuming from checkpoint, try to load the card_id and game_id
        checkpoint_card_id = None  # Track original checkpoint card_id
        if resume_from_checkpoint and card_id:
            checkpoint_mgr = CheckpointManager(card_id)
            if checkpoint_mgr.checkpoint_exists():
                checkpoint_info = CheckpointManager.get_checkpoint_info(card_id)
                if checkpoint_info:
                    game_id = checkpoint_info.get("game_id", game_id)
                    checkpoint_card_id = card_id  # Preserve original checkpoint card_id
                    logger.info(f"Resuming from checkpoint: card_id={card_id}, game_id={game_id}")
            else:
                logger.warning(f"No checkpoint found for card_id={card_id}, starting fresh")
                resume_from_checkpoint = False

        # Note: Results are saved with unique timestamps, so multiple runs are allowed
        # Each run creates a new file: {game_id}_{config}_{timestamp}.json

        # Check if scorecard still exists on server when resuming
        if resume_from_checkpoint and card_id:
            try:
                # Try to get the scorecard to verify it still exists
                self.game_client.get_scorecard(card_id)
                logger.info(f"Verified existing scorecard: {card_id}")
            except Exception as e:
                logger.warning(f"Scorecard {card_id} no longer exists on server: {e}")
                if self.submit_scorecard:
                    logger.info("Opening new scorecard for checkpoint recovery...")
                    # Open a new scorecard with the same card_id (API will reject, so we get a new one)
                    tags = generate_scorecard_tags(self.model_config)
                    scorecard_response = self.game_client.open_scorecard(
                        [game_id], card_id=None, tags=tags
                    )
                    old_card_id = card_id
                    card_id = scorecard_response.get("card_id", card_id)
                    logger.info(f"Created new scorecard: {card_id} (old: {old_card_id})")
                    logger.info(
                        f"Checkpoint will continue using original card_id: {checkpoint_card_id}"
                    )
                    # Note: We keep resume_from_checkpoint=True to restore agent state,
                    # but the game will need to reset since the old session is gone
                else:
                    logger.info(
                        "submit_scorecard is disabled; continuing without creating a new "
                        "scorecard on the server."
                    )
        else:
            # Decide whether to open a real scorecard or run in local-only mode.
            local_only = not self.submit_scorecard and not card_id

            if local_only:
                # Generate a local-only card_id used purely for checkpointing and
                # game commands. No scorecard is explicitly opened or closed.
                if not card_id:
                    card_id = f"local-{uuid.uuid4()}"
                logger.info(
                    f"Running in local-only mode without opening a scorecard (card_id={card_id})"
                )
            else:
                # Generate tags from model config for scorecard tracking
                tags = generate_scorecard_tags(self.model_config)
                scorecard_response = self.game_client.open_scorecard(
                    [game_id], card_id=card_id, tags=tags
                )
                card_id = scorecard_response.get("card_id", card_id)
        
        
        try:
            from arcagi3.utils import load_hints, find_hints_file
            
            hints_file = find_hints_file()
            hint_found = False
            if hints_file:
                hints = load_hints(hints_file, game_id=game_id)
                hint_found = game_id in hints
            
            if hint_found:
                logger.info(f"✓ Hint found for game {game_id}")
            else:
                logger.debug(f"⊘ No hint found for game {game_id}")
            
            # Create agent
            # Use checkpoint_card_id for checkpoint management if resuming, otherwise use card_id
            agent = MultimodalAgent(
                config=self.config,
                game_client=self.game_client,
                card_id=card_id,
                max_actions=self.max_actions,
                retry_attempts=self.retry_attempts,
                num_plays=self.num_plays,
                show_images=self.show_images,
                use_vision=self.use_vision,
                checkpoint_frequency=self.checkpoint_frequency,
                checkpoint_card_id=checkpoint_card_id,
                memory_word_limit=self.memory_word_limit,
            )

            # Play game (with checkpoint support)
            result = agent.play_game(game_id, resume_from_checkpoint=resume_from_checkpoint)
            
            # Save result if directory provided
            if self.save_results_dir:
                result_file = save_result(self.save_results_dir, result)
                logger.info(f"Saved result to {result_file}")
            
            # Determine the actual checkpoint card_id for logging
            actual_checkpoint_id = checkpoint_card_id if checkpoint_card_id else card_id

            # Determine whether we should close the scorecard. When no card_id
            # existed initially and submit_scorecard is False, we never opened a
            # scorecard, so there is nothing to close.
            if self.submit_scorecard or checkpoint_card_id or not card_id.startswith("local-"):
                if result.final_state == "WIN" or self.close_on_exit:
                    try:
                        self.game_client.close_scorecard(card_id)
                        logger.info(f"Closed scorecard {card_id}")
                    except Exception as e:
                        logger.debug(f"Could not close scorecard: {e}")
                else:
                    logger.info(
                        f"Scorecard {card_id} left open for potential checkpoint resume"
                    )
                    logger.info(f"Checkpoint saved at: .checkpoint/{actual_checkpoint_id}")
            else:
                logger.info(
                    f"Local-only run completed; no scorecard was opened. "
                    f"Checkpoint (if any) is under .checkpoint/{actual_checkpoint_id}"
                )

            return result

        except KeyboardInterrupt:
            # Determine the actual checkpoint card_id for logging
            actual_checkpoint_id = checkpoint_card_id if checkpoint_card_id else card_id
            logger.info("Interrupted by user (Ctrl-C)")
            logger.info(f"Checkpoint saved at: .checkpoint/{actual_checkpoint_id}")
            logger.info(f"Resume with: python main.py --checkpoint {actual_checkpoint_id}")
            raise
        except Exception as e:
            # Determine the actual checkpoint card_id for logging
            actual_checkpoint_id = checkpoint_card_id if checkpoint_card_id else card_id
            logger.error(f"Error during game execution: {e}")
            logger.info(f"Checkpoint may be available at: .checkpoint/{actual_checkpoint_id}")
            raise
