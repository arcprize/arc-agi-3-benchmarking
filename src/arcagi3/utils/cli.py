
import logging
import os
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from arcagi3.checkpoint import CheckpointManager
from arcagi3.game_client import GameClient
from typing import List, Optional, Tuple
from arcagi3.arc3tester import ARC3Tester
from arcagi3.schemas import GameResult

logger = logging.getLogger(__name__)

# Thread-local storage for game ID
_thread_local = threading.local()

# ============================================================================
# CLI Arguments
# ============================================================================

def _bool_env(env_var: str, default: str = "false") -> bool:
    """Helper to parse boolean environment variable."""
    return os.getenv(env_var, default).lower() in ("true", "1", "yes")

def _int_env(env_var: str, default: int) -> int:
    """Helper to parse integer environment variable."""
    val = os.getenv(env_var)
    return int(val) if val else default

def _str_env(env_var: str, default: str = None) -> str:
    """Helper to parse string environment variable."""
    return os.getenv(env_var, default)

def configure_args(parser):
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default=_str_env("CONFIG"),
        help="Model configuration name from models.yml. Not required when using --checkpoint. Can be set via CONFIG env var."
    )
    parser.add_argument(
        "--save_results_dir",
        type=str,
        default=_str_env("SAVE_RESULTS_DIR"),
        help="Directory to save results (default: results/<config>). Can be set via SAVE_RESULTS_DIR env var."
    )
    parser.add_argument(
        "--overwrite_results",
        action="store_true",
        help="Overwrite existing result files. Can be set via OVERWRITE_RESULTS env var (true/1/yes)."
    )
    parser.add_argument(
        "--max_actions",
        type=int,
        default=_int_env("MAX_ACTIONS", 40),
        help="Maximum actions for entire run across all games/plays (default: 40, 0 = no limit). Can be set via MAX_ACTIONS env var."
    )
    parser.add_argument(
        "--retry_attempts",
        type=int,
        default=_int_env("RETRY_ATTEMPTS", 3),
        help="Number of retry attempts for API failures (default: 3). Can be set via RETRY_ATTEMPTS env var."
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=_int_env("RETRIES", 3),
        help="Number of retry attempts for ARC-AGI-3 API calls (default: 3). Can be set via RETRIES env var."
    )
    parser.add_argument(
        "--num_plays",
        type=int,
        default=_int_env("NUM_PLAYS", 0),
        help="Number of times to play each game (0 = infinite, default: 0, continues session with memory on subsequent plays). Can be set via NUM_PLAYS env var."
    )
    parser.add_argument(
        "--max_episode_actions",
        type=int,
        default=_int_env("MAX_EPISODE_ACTIONS", 0),
        help="Maximum actions per game/episode (default: 0 = no limit). Can be set via MAX_EPISODE_ACTIONS env var."
    )

    # Display
    parser.add_argument(
        "--show-images",
        action="store_true",
        help="Display game frames in the terminal. Can be set via SHOW_IMAGES env var (true/1/yes)."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=_str_env("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO). Can be set via LOG_LEVEL env var."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (DEBUG level for app, WARNING for libraries). Can be set via VERBOSE env var (true/1/yes)."
    )
    parser.add_argument(
        "--memory-limit",
        type=int,
        default=int(os.getenv("MEMORY_LIMIT")) if os.getenv("MEMORY_LIMIT") else None,
        help="Maximum number of words allowed in memory scratchpad (overrides model config). Can be set via MEMORY_LIMIT env var."
    )
    parser.add_argument(
        "--use_vision",
        action="store_true",
        help="Use vision to play the game (default: True). Can be set via USE_VISION env var (true/1/yes)."
    )
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=_int_env("CHECKPOINT_FREQUENCY", 1),
        help="Save checkpoint every N actions (default: 1, 0 to disable periodic checkpoints). Can be set via CHECKPOINT_FREQUENCY env var."
    )
    parser.add_argument(
        "--close-on-exit",
        action="store_true",
        help="Close scorecard on exit even if game not won (prevents checkpoint resume). Can be set via CLOSE_ON_EXIT env var (true/1/yes)."
    )
    parser.add_argument(
        "--no-scorecard-submission",
        action="store_true",
        help="Do not open or close scorecards on the ARC server; run in local-only mode when no existing card_id is provided."
    )

def configure_cli_args(parser):
    # Game selection (mutually exclusive)
    game_group = parser.add_mutually_exclusive_group(required=False)
    game_group.add_argument(
        "--games",
        type=str,
        help="Comma-separated list of game IDs"
    )
    game_group.add_argument(
        "--all-games",
        action="store_true",
        help="Run all available games from the API"
    )
    game_group.add_argument(
        "--list-games",
        action="store_true",
        help="List all available games and exit"
    )

def configure_main_args(parser):
    # Checkpoint options
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--checkpoint",
        type=str,
        metavar="CARD_ID",
        default=_str_env("CHECKPOINT"),
        help="Resume from existing checkpoint using the specified scorecard ID. Can be set via CHECKPOINT env var."
    )
    checkpoint_group.add_argument(
        "--list-checkpoints",
        action="store_true",
        help="List all available checkpoints and exit. Can be set via LIST_CHECKPOINTS env var (true/1/yes)."
    )
    checkpoint_group.add_argument(
        "--close-scorecard",
        type=str,
        metavar="CARD_ID",
        default=_str_env("CLOSE_SCORECARD"),
        help="Close a scorecard by ID and exit. Can be set via CLOSE_SCORECARD env var."
    )

    parser.add_argument(
        "--game_id",
        type=str,
        default=_str_env("GAME_ID"),
        help="Game ID to play (e.g., 'ls20-016295f7601e'). Not required when using --checkpoint. Can be set via GAME_ID env var."
    )

# ============================================================================
# CLI Configurers
# ============================================================================

def apply_env_vars_to_args(args):
    """
    Apply environment variables to parsed arguments.
    This is needed for boolean flags since argparse's store_true action
    doesn't respect default values from environment variables.
    """
    # Boolean flags that can be set via env vars
    # Only override if env var is set (allows CLI flags to take precedence)
    if os.getenv("OVERWRITE_RESULTS"):
        args.overwrite_results = _bool_env("OVERWRITE_RESULTS")
    if os.getenv("SHOW_IMAGES"):
        args.show_images = _bool_env("SHOW_IMAGES")
    if os.getenv("VERBOSE"):
        args.verbose = _bool_env("VERBOSE")
    # use_vision defaults to True, so check env var if set
    if os.getenv("USE_VISION"):
        args.use_vision = _bool_env("USE_VISION", "true")
    elif not args.use_vision:  # If flag wasn't set, default to True
        args.use_vision = True
    if os.getenv("CLOSE_ON_EXIT"):
        args.close_on_exit = _bool_env("CLOSE_ON_EXIT")
    if os.getenv("LIST_CHECKPOINTS"):
        args.list_checkpoints = _bool_env("LIST_CHECKPOINTS")
    
    return args

def validate_args(args, parser):
    if args.checkpoint:
        # When resuming from checkpoint, config and game_id are optional (loaded from checkpoint)
        checkpoint_info = CheckpointManager.get_checkpoint_info(args.checkpoint)
        if not checkpoint_info:
            print(f"Error: Checkpoint '{args.checkpoint}' not found.")
            print("Use --list-checkpoints to see available checkpoints.")
            return

        # Use checkpoint values if not provided
        if not args.config:
            args.config = checkpoint_info.get("config")
            print(f"Using config from checkpoint: {args.config}")
        if not args.game_id:
            args.game_id = checkpoint_info.get("game_id")
            print(f"Using game_id from checkpoint: {args.game_id}")
    else:
        # When not using checkpoint, both are required
        if not args.game_id or not args.config:
            parser.error("--game_id and --config are required unless using --checkpoint")

def configure_logging(args):
    if args.verbose:
        # Verbose mode: Show DEBUG for our code, WARNING+ for libraries
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Set library loggers to WARNING
        library_loggers = [
            'openai', 'httpx', 'httpcore', 'urllib3', 'requests',
            'anthropic', 'google', 'pydantic'
        ]
        for lib_logger in library_loggers:
            logging.getLogger(lib_logger).setLevel(logging.WARNING)
        
        # Keep our application loggers at DEBUG
        logging.getLogger('arcagi3').setLevel(logging.DEBUG)
        logging.getLogger('__main__').setLevel(logging.DEBUG)
        
        logger.info("Verbose mode enabled")
    else:
        # Normal mode: Use the specified log level
        logging.basicConfig(
            level=getattr(logging, args.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

# ============================================================================
# CLI Handlers
# ============================================================================

def list_available_games(game_client: GameClient) -> List[dict]:
    """List all available games from the API"""
    try:
        games = game_client.list_games()
        return games
    except Exception as e:
        logger.error(f"Failed to list games: {e}")
        return []

def handle_list_games(game_client: GameClient):
    games = list_available_games(game_client)
    if games:
        logger.info("\nAvailable Games:")
        logger.info("=" * 60)
        for game in games:
            logger.info(f"  {game['game_id']:<30} {game['title']}")
        logger.info("=" * 60)
        logger.info(f"Total: {len(games)} games\n")
    else:
        logger.warning("No games available or failed to fetch games.")

def handle_list_checkpoints():
    checkpoints = CheckpointManager.list_checkpoints()
    if checkpoints:
        logger.info("\nAvailable Checkpoints:")
        logger.info("=" * 80)
        for card_id in checkpoints:
            info = CheckpointManager.get_checkpoint_info(card_id)
            if info:
                logger.info(f"  Card ID: {card_id}")
                logger.info(f"    Game: {info.get('game_id', 'N/A')}")
                logger.info(f"    Config: {info.get('config', 'N/A')}")
                logger.info(f"    Actions: {info.get('action_counter', 0)}")
                logger.info(f"    Play: {info.get('current_play', 1)}/{info.get('num_plays', 1)}")
                logger.info(f"    Timestamp: {info.get('checkpoint_timestamp', 'N/A')}")
                logger.info("")
        logger.info("=" * 80)
        logger.info(f"Total: {len(checkpoints)} checkpoint(s)\n")
    else:
        logger.info("No checkpoints found.\n")

def handle_close_scorecard(args):
    card_id = args.close_scorecard
    logger.info(f"\nClosing scorecard: {card_id}")
    try:
        game_client = GameClient()
        response = game_client.close_scorecard(card_id)
        logger.info(f"✓ Successfully closed scorecard {card_id}")
        logger.info(f"Response: {response}")

        # Optionally delete local checkpoint
        checkpoint_mgr = CheckpointManager(card_id)
        if checkpoint_mgr.checkpoint_exists():
            logger.info(f"\nLocal checkpoint still exists at: .checkpoint/{card_id}")
            logger.info(f"To delete it, run: rm -rf .checkpoint/{card_id}")
    except Exception as e:
        logger.error(f"✗ Failed to close scorecard: {e}", exc_info=True)

def print_result(result):
    logger.info(f"\n{'='*60}")
    logger.info(f"Game Result: {result.game_id}")
    logger.info(f"{'='*60}")
    logger.info(f"Final Score: {result.final_score}")
    logger.info(f"Final State: {result.final_state}")
    logger.info(f"Actions Taken: {result.actions_taken}")
    logger.info(f"Duration: {result.duration_seconds:.2f}s")
    logger.info(f"Total Cost: ${result.total_cost.total_cost:.4f}")
    logger.info(f"Total Tokens: {result.usage.total_tokens}")
    logger.info(f"\nView your scorecard online: {result.scorecard_url}")
    logger.info(f"{'='*60}\n")


class GameLogFilter(logging.Filter):
    """Filter that only allows logs from a specific game (thread)."""
    def __init__(self, game_id: str):
        super().__init__()
        self.game_id = game_id
    
    def filter(self, record):
        # Only log if this record is from the thread with matching game_id
        return getattr(_thread_local, 'game_id', None) == self.game_id


def _setup_game_logger(game_id: str, config: str, log_dir: Path) -> logging.FileHandler:
    """
    Set up a file handler for a specific game using thread-local storage.
    
    Args:
        game_id: Game ID
        config: Config name
        log_dir: Directory where log files should be created
        
    Returns:
        File handler for this game
    """
    # Set thread-local game_id so filter knows which thread this is
    _thread_local.game_id = game_id
    
    # Create log directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file path
    log_file = log_dir / f"{game_id}.log"
    
    # Create file handler with filter
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    file_handler.addFilter(GameLogFilter(game_id))
    
    # Add handler to root logger to catch all logs from this thread
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    return file_handler


def _teardown_game_logger(file_handler: logging.FileHandler):
    """
    Remove the file handler and clean up thread-local storage.
    
    Args:
        file_handler: The file handler to remove
    """
    root_logger = logging.getLogger()
    try:
        root_logger.removeHandler(file_handler)
        file_handler.close()
    except (ValueError, AttributeError):
        pass
    
    # Clear thread-local game_id
    if hasattr(_thread_local, 'game_id'):
        delattr(_thread_local, 'game_id')


def _run_single_game(
    game_id: str,
    config: str,
    save_results_dir: Optional[str],
    overwrite_results: bool,
    max_actions: int,
    retry_attempts: int,
    api_retries: int,
    num_plays: int,
    max_episode_actions: int,
    show_images: bool,
    memory_word_limit: Optional[int],
    checkpoint_frequency: int,
    close_on_exit: bool,
    use_vision: bool,
    game_index: int,
    total_games: int,
    log_dir: Optional[Path] = None,
    submit_scorecard: bool = True,
) -> Tuple[str, Optional[GameResult], Optional[Exception]]:
    """
    Run a single game and return the result.
    
    Returns:
        Tuple of (game_id, result, exception)
    """
    # Set up per-game logging if log_dir is provided
    file_handler = None
    if log_dir:
        try:
            file_handler = _setup_game_logger(game_id, config, log_dir)
        except Exception as e:
            logger.warning(f"Failed to set up per-game logging for {game_id}: {e}")
    
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Game {game_index}/{total_games}: {game_id}")
        logger.info(f"{'='*60}")
        
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
        
        # Create a separate tester instance for each game to ensure thread safety
        tester = ARC3Tester(
            config=config,
            save_results_dir=save_results_dir,
            overwrite_results=overwrite_results,
            max_actions=max_actions,
            retry_attempts=retry_attempts,
            api_retries=api_retries,
            num_plays=num_plays,
            max_episode_actions=max_episode_actions,
            show_images=show_images,
            memory_word_limit=memory_word_limit,
            checkpoint_frequency=checkpoint_frequency,
            close_on_exit=close_on_exit,
            use_vision=use_vision,
            submit_scorecard=submit_scorecard,
        )
        
        try:
            result = tester.play_game(game_id)
            if result:
                logger.info(
                    f"✓ [{game_id}] Completed: {result.final_state}, "
                    f"Score: {result.final_score}, "
                    f"Cost: ${result.total_cost.total_cost:.4f}"
                )
                if result.scorecard_url:
                    logger.info(f"View your scorecard online: {result.scorecard_url}")
                return (game_id, result, None)
            else:
                logger.info(f"⊘ [{game_id}] Skipped (result already exists)")
                return (game_id, None, None)
        except Exception as e:
            logger.error(f"✗ [{game_id}] Failed: {e}", exc_info=True)
            return (game_id, None, e)
    finally:
        # Clean up per-game logging
        if file_handler:
            try:
                _teardown_game_logger(file_handler)
            except Exception as e:
                logger.warning(f"Failed to teardown logging for {game_id}: {e}")


def run_batch_games(
    game_ids: List[str],
    config: str,
    save_results_dir: Optional[str] = None,
    overwrite_results: bool = False,
    max_actions: int = 40,
    retry_attempts: int = 3,
    api_retries: int = 3,
    num_plays: int = 1,
    max_episode_actions: int = 0,
    show_images: bool = False,
    memory_word_limit: Optional[int] = None,
    checkpoint_frequency: int = 1,
    close_on_exit: bool = False,
    use_vision: bool = True,
    submit_scorecard: bool = True,
):
    """
    Run multiple games concurrently in parallel.
    
    Args:
        game_ids: List of game IDs to run
        config: Model configuration name
        save_results_dir: Directory to save results
        overwrite_results: Whether to overwrite existing results
        max_actions: Maximum actions for entire run across all games/plays (0 = no limit)
        retry_attempts: Number of retry attempts for API failures
        api_retries: Number of retry attempts for ARC-AGI-3 API calls
        num_plays: Number of times to play each game (0 = infinite, continues session with memory)
        max_episode_actions: Maximum actions per game/episode (0 = no limit)
        show_images: Whether to display game frames in the terminal
        memory_word_limit: Maximum number of words allowed in memory scratchpad
        checkpoint_frequency: Save checkpoint every N actions (default: 1, 0 to disable periodic checkpoints)
        close_on_exit: Close scorecard on exit even if game not won
        use_vision: Use vision to play the game
    """
    logger.info(f"Running {len(game_ids)} games concurrently with config {config}")
    
    # Create log directory for this concurrent run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / config / f"concurrent_{timestamp}"
    logger.info(f"Log files will be saved to: {log_dir}")
    
    # Track results
    results = []
    successes = 0
    failures = 0
    skipped = 0
    
    # Run games concurrently using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(game_ids)) as executor:
        # Submit all games
        future_to_game = {
            executor.submit(
                _run_single_game,
                game_id,
                config,
                save_results_dir,
                overwrite_results,
                max_actions,
                retry_attempts,
                api_retries,
                num_plays,
                max_episode_actions,
                show_images,
                memory_word_limit,
                checkpoint_frequency,
                close_on_exit,
                use_vision,
                i + 1,
                len(game_ids),
                log_dir,
                submit_scorecard,
            ): game_id
            for i, game_id in enumerate(game_ids)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_game):
            game_id, result, exception = future.result()
            if exception:
                failures += 1
            elif result:
                results.append(result)
                successes += 1
            else:
                skipped += 1
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Batch Run Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Total Games: {len(game_ids)}")
    logger.info(f"Successes: {successes}")
    logger.info(f"Failures: {failures}")
    logger.info(f"Skipped: {skipped}")
    
    if results:
        total_cost = sum(r.total_cost.total_cost for r in results)
        total_actions = sum(r.actions_taken for r in results)
        total_duration = sum(r.duration_seconds for r in results)
        
        logger.info(f"\nAggregates:")
        logger.info(f"  Total Cost: ${total_cost:.4f}")
        logger.info(f"  Total Actions: {total_actions}")
        logger.info(f"  Total Duration: {total_duration:.2f}s")
        logger.info(f"  Avg Cost per Game: ${total_cost/len(results):.4f}")
        logger.info(f"  Avg Actions per Game: {total_actions/len(results):.1f}")
        
        # Show scorecard URLs
        logger.info(f"\nScorecard Links:")
        for result in results:
            if result.scorecard_url:
                logger.info(f"  {result.game_id}: {result.scorecard_url}")
    
    # Show log file locations
    logger.info(f"\nLog Files:")
    for game_id in game_ids:
        log_file = log_dir / f"{game_id}.log"
        if log_file.exists():
            logger.info(f"  {game_id}: {log_file}")
    
    logger.info(f"{'='*60}\n")
