
import logging
import os
from arcagi3.checkpoint import CheckpointManager
from arcagi3.game_client import GameClient
from typing import List

logger = logging.getLogger(__name__)

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

    # Breakpoints
    parser.add_argument(
        "--breakpoints",
        action="store_true",
        help="Enable breakpoint UI integration. Can be set via BREAKPOINTS_ENABLED env var (true/1/yes)."
    )
    parser.add_argument(
        "--breakpoint-ws-url",
        type=str,
        default=_str_env("BREAKPOINT_WS_URL", "ws://localhost:8765/ws"),
        help="WebSocket URL for breakpoint server (default: ws://localhost:8765/ws). Can be set via BREAKPOINT_WS_URL env var."
    )
    parser.add_argument(
        "--breakpoint-schema",
        type=str,
        default=_str_env("BREAKPOINT_SCHEMA"),
        help="Path to breakpoint schema JSON file. Can be set via BREAKPOINT_SCHEMA env var."
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
    if os.getenv("BREAKPOINTS_ENABLED"):
        args.breakpoints = _bool_env("BREAKPOINTS_ENABLED")
    
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
