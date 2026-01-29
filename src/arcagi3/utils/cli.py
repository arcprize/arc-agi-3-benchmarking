import logging
import os
import re
from typing import Any, Dict, List, Optional

import yaml

from arcagi3.checkpoint import CheckpointManager
from arcagi3.game_client import GameClient
from arcagi3.utils.api_tests import (
    test_anthropic,
    test_arc_api_key,
    test_deepseek,
    test_fireworks,
    test_gemini,
    test_groq,
    test_huggingface,
    test_openai,
    test_openrouter,
    test_provider_api_key,
    test_xai,
)

logger = logging.getLogger(__name__)

_GAME_ID_WITH_HASH_RE = re.compile(r"^(?P<base>[a-zA-Z]{2}\d{2})-[0-9a-fA-F]{7,40}$")

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
        help="Model configuration name from models.yml. Not required when using --checkpoint. Can be set via CONFIG env var.",
    )
    parser.add_argument(
        "--save_results_dir",
        type=str,
        default=_str_env("SAVE_RESULTS_DIR"),
        help="Directory to save results (default: results/<config>). Can be set via SAVE_RESULTS_DIR env var.",
    )
    parser.add_argument(
        "--overwrite_results",
        action="store_true",
        help="Overwrite existing result files. Can be set via OVERWRITE_RESULTS env var (true/1/yes).",
    )
    parser.add_argument(
        "--max_actions",
        type=int,
        default=_int_env("MAX_ACTIONS", 40),
        help="Maximum actions for entire run across all games/plays (default: 40, 0 = no limit). Can be set via MAX_ACTIONS env var.",
    )
    parser.add_argument(
        "--retry_attempts",
        type=int,
        default=_int_env("RETRY_ATTEMPTS", 3),
        help="Number of retry attempts for API failures (default: 3). Can be set via RETRY_ATTEMPTS env var.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=_int_env("RETRIES", 3),
        help="Number of retry attempts for ARC-AGI-3 API calls (default: 3). Can be set via RETRIES env var.",
    )
    parser.add_argument(
        "--num_plays",
        type=int,
        default=_int_env("NUM_PLAYS", 0),
        help="Number of times to play each game (0 = infinite, default: 0, continues session with memory on subsequent plays). Can be set via NUM_PLAYS env var.",
    )
    parser.add_argument(
        "--max_episode_actions",
        type=int,
        default=_int_env("MAX_EPISODE_ACTIONS", 0),
        help="Maximum actions per game/episode (default: 0 = no limit). Can be set via MAX_EPISODE_ACTIONS env var.",
    )

    # Display
    parser.add_argument(
        "--show-images",
        action="store_true",
        help="Display game frames in the terminal. Can be set via SHOW_IMAGES env var (true/1/yes).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=_str_env("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO). Can be set via LOG_LEVEL env var.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (DEBUG level for app, WARNING for libraries). Can be set via VERBOSE env var (true/1/yes).",
    )
    parser.add_argument(
        "--memory-limit",
        type=int,
        default=int(os.getenv("MEMORY_LIMIT")) if os.getenv("MEMORY_LIMIT") else None,
        help="Maximum number of words allowed in memory scratchpad (overrides model config). Can be set via MEMORY_LIMIT env var.",
    )
    parser.add_argument(
        "--use_vision",
        action="store_true",
        help="Use vision to play the game (default: True). Can be set via USE_VISION env var (true/1/yes).",
    )
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=_int_env("CHECKPOINT_FREQUENCY", 1),
        help="Save checkpoint every N actions (default: 1, 0 to disable periodic checkpoints). Can be set via CHECKPOINT_FREQUENCY env var.",
    )
    parser.add_argument(
        "--close-on-exit",
        action="store_true",
        help="Close scorecard on exit even if game not won (prevents checkpoint resume). Can be set via CLOSE_ON_EXIT env var (true/1/yes).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run in offline mode: no scoredcards are created on the ARC server; run in local-only mode when no existing card_id is provided. Can be set via OFFLINE env var (true/1/yes).",
    )

    # Breakpoints
    parser.add_argument(
        "--breakpoints",
        action="store_true",
        help="Enable breakpoint UI integration. Can be set via BREAKPOINTS_ENABLED env var (true/1/yes).",
    )
    parser.add_argument(
        "--breakpoint-ws-url",
        type=str,
        default=_str_env("BREAKPOINT_WS_URL", "ws://localhost:8765/ws"),
        help="WebSocket URL for breakpoint server (default: ws://localhost:8765/ws). Can be set via BREAKPOINT_WS_URL env var.",
    )
    parser.add_argument(
        "--breakpoint-schema",
        type=str,
        default=_str_env("BREAKPOINT_SCHEMA"),
        help="Path to breakpoint schema JSON file. Can be set via BREAKPOINT_SCHEMA env var.",
    )


def configure_main_args(parser):
    # Checkpoint options
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--checkpoint",
        type=str,
        metavar="CARD_ID",
        default=_str_env("CHECKPOINT"),
        help="Resume from existing checkpoint using the specified scorecard ID. Can be set via CHECKPOINT env var.",
    )
    checkpoint_group.add_argument(
        "--list-checkpoints",
        action="store_true",
        help="List all available checkpoints and exit. Can be set via LIST_CHECKPOINTS env var (true/1/yes).",
    )
    checkpoint_group.add_argument(
        "--close-scorecard",
        type=str,
        metavar="CARD_ID",
        default=_str_env("CLOSE_SCORECARD"),
        help="Close a scorecard by ID and exit. Can be set via CLOSE_SCORECARD env var.",
    )

    parser.add_argument(
        "--game_id",
        type=str,
        default=_str_env("GAME_ID"),
        help="Game ID to play (e.g., 'ls20-016295f7601e'). Not required when using --checkpoint. Can be set via GAME_ID env var.",
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
    if os.getenv("OFFLINE"):
        args.offline = _bool_env("OFFLINE")
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
            level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Set library loggers to WARNING
        library_loggers = [
            "openai",
            "httpx",
            "httpcore",
            "urllib3",
            "requests",
            "anthropic",
            "google",
            "pydantic",
        ]
        for lib_logger in library_loggers:
            logging.getLogger(lib_logger).setLevel(logging.WARNING)

        # Keep our application loggers at DEBUG
        logging.getLogger("arcagi3").setLevel(logging.DEBUG)
        logging.getLogger("__main__").setLevel(logging.DEBUG)

        logger.info("Verbose mode enabled")
    else:
        # Normal mode: Use the specified log level
        logging.basicConfig(
            level=getattr(logging, args.log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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


def _strip_game_hash(game_id: str) -> str:
    if not isinstance(game_id, str):
        return game_id
    match = _GAME_ID_WITH_HASH_RE.match(game_id)
    return match.group("base") if match else game_id


def _normalize_game_ids(games: List[dict]) -> List[dict]:
    normalized = []
    for game in games:
        if not isinstance(game, dict):
            normalized.append(game)
            continue
        game_id = game.get("game_id")
        if not game_id:
            normalized.append(game)
            continue
        stripped_id = _strip_game_hash(game_id)
        if stripped_id != game_id:
            game = dict(game)
            game["game_id"] = stripped_id
        normalized.append(game)
    return normalized


def handle_list_games(game_client: GameClient, json_output: bool = False):
    """List available games, optionally in JSON format."""
    import json

    games = _normalize_game_ids(list_available_games(game_client))
    if not games:
        if json_output:
            print(json.dumps([], indent=2))
        else:
            print("No games available or failed to fetch games.")
        return

    if json_output:
        print(json.dumps(games, indent=2))
        return

    # Create a nice table format
    # Calculate column widths
    max_id_len = max(len(game["game_id"]) for game in games)
    max_title_len = max(len(game.get("title", "")) for game in games)

    # Set minimum widths
    id_width = max(max_id_len, 20)
    title_width = max(max_title_len, 30)

    # Calculate total width
    total_width = id_width + title_width + 7

    # Print header
    print("\n" + "=" * total_width)
    print(f"{'Game ID':<{id_width}}  {'Title':<{title_width}}")
    print("=" * total_width)

    # Print games (sorted by game_id)
    for game in sorted(games, key=lambda x: x["game_id"]):
        game_id = game["game_id"]
        title = game.get("title", "N/A")
        print(f"{game_id:<{id_width}}  {title:<{title_width}}")

    # Print footer
    print("=" * total_width)
    print(f"\nTotal: {len(games)} game{'s' if len(games) != 1 else ''}\n")


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


def handle_list_models():
    """List all models for enabled providers."""
    from arcagi3.utils.api_tests import (
        test_anthropic,
        test_deepseek,
        test_fireworks,
        test_gemini,
        test_groq,
        test_huggingface,
        test_openai,
        test_openrouter,
        test_provider_api_key,
        test_xai,
    )

    # Provider mapping: maps provider_id to (display_name, env_var, test_func)
    provider_mapping = {
        "openai": ("OpenAI", "OPENAI_API_KEY", test_openai),
        "anthropic": ("Anthropic", "ANTHROPIC_API_KEY", test_anthropic),
        "gemini": ("Google Gemini", "GOOGLE_API_KEY", test_gemini),
        "openrouter": ("OpenRouter", "OPENROUTER_API_KEY", test_openrouter),
        "fireworks": ("Fireworks", "FIREWORKS_API_KEY", test_fireworks),
        "groq": ("Groq", "GROQ_API_KEY", test_groq),
        "deepseek": ("DeepSeek", "DEEPSEEK_API_KEY", test_deepseek),
        "xai": ("xAI", "XAI_API_KEY", test_xai),
        "grok": ("xAI", "XAI_API_KEY", test_xai),  # grok maps to xai
        "huggingfacefireworks": ("Hugging Face", "HUGGING_FACE_API_KEY", test_huggingface),
    }

    # Test providers and get enabled ones
    enabled_providers = set()
    provider_display_names = {}

    for provider_id, (provider_name, env_var, test_func) in provider_mapping.items():
        status, _ = test_provider_api_key(provider_name, env_var, test_func)
        if status is True:
            enabled_providers.add(provider_id)
            # Also add display name mapping
            provider_display_names[provider_id] = provider_name
            # If this is grok, also enable xai (they're the same)
            if provider_id == "grok":
                enabled_providers.add("xai")
                provider_display_names["xai"] = provider_name

    if not enabled_providers:
        print("\n" + "=" * 80)
        print("No providers are enabled. Run --check to see provider status.")
        print("=" * 80 + "\n")
        return

    # Load models
    base_dir = os.path.dirname(os.path.dirname(__file__))
    models_file = os.path.join(base_dir, "models.yml")
    models_private_file = os.path.join(base_dir, "models_private.yml")

    all_models = []

    # Load main models.yml
    if os.path.exists(models_file):
        with open(models_file, "r") as f:
            config_data = yaml.safe_load(f)
            if config_data and "models" in config_data:
                all_models.extend(config_data["models"])

    # Load private models.yml if it exists
    if os.path.exists(models_private_file):
        with open(models_private_file, "r") as f:
            private_config_data = yaml.safe_load(f)
            if private_config_data and "models" in private_config_data:
                all_models.extend(private_config_data["models"])

    # Filter models to only enabled providers
    enabled_models = []
    for model in all_models:
        provider = model.get("provider", "").lower()
        # Check if provider is enabled (handle grok -> xai mapping)
        if provider in enabled_providers or (provider == "grok" and "xai" in enabled_providers):
            enabled_models.append(model)

    if not enabled_models:
        print("\n" + "=" * 80)
        print("No models found for enabled providers.")
        print("=" * 80 + "\n")
        return

    # Group models by provider
    models_by_provider: Dict[str, List[Dict[str, Any]]] = {}
    for model in enabled_models:
        provider = model.get("provider", "").lower()
        # Normalize grok to xai for grouping
        if provider == "grok":
            provider = "xai"
        if provider not in models_by_provider:
            models_by_provider[provider] = []
        models_by_provider[provider].append(model)

    # Display models
    print("\n" + "=" * 80)
    print("Available Models (for enabled providers)")
    print("=" * 80)

    for provider_id in sorted(models_by_provider.keys()):
        provider_name = provider_display_names.get(provider_id, provider_id.title())
        models = models_by_provider[provider_id]

        print(f"\n{provider_name} ({len(models)} model{'s' if len(models) != 1 else ''}):")
        print("-" * 80)

        # Sort models by name
        models.sort(key=lambda m: m.get("name", ""))

        for model in models:
            name = model.get("name", "N/A")
            pricing = model.get("pricing", {})
            input_price = pricing.get("input", 0)
            output_price = pricing.get("output", 0)
            is_multimodal = model.get("is_multimodal", False)

            # Build info string
            info_parts = []
            if is_multimodal:
                info_parts.append("multimodal")

            # Check for reasoning/thinking features
            if "reasoning" in model or "reasoning_effort" in model:
                info_parts.append("reasoning")
            if "thinking" in model or "thinking_config" in model:
                info_parts.append("thinking")

            info_str = ", ".join(info_parts) if info_parts else "standard"

            print(
                f"  {name:<40} {info_str:<20} ${input_price:.2f}/${output_price:.2f} per 1M tokens"
            )

    print("\n" + "=" * 80)
    print(f"Total: {len(enabled_models)} model{'s' if len(enabled_models) != 1 else ''} available")
    print("=" * 80 + "\n")


def handle_check():
    """Check environment variables and test API keys."""
    results: List[Dict[str, Any]] = []
    games_list: Optional[List[Dict[str, str]]] = None

    # Test ARC API key
    arc_status, arc_message, games_list = test_arc_api_key()
    results.append(
        {
            "name": "ARC-AGI-3 API",
            "env_var": "ARC_API_KEY",
            "status": arc_status,
            "message": arc_message,
        }
    )

    # Test provider API keys with provider mapping
    provider_mapping = {
        "openai": ("OpenAI", "OPENAI_API_KEY", test_openai),
        "anthropic": ("Anthropic", "ANTHROPIC_API_KEY", test_anthropic),
        "gemini": ("Google Gemini", "GOOGLE_API_KEY", test_gemini),
        "openrouter": ("OpenRouter", "OPENROUTER_API_KEY", test_openrouter),
        "fireworks": ("Fireworks", "FIREWORKS_API_KEY", test_fireworks),
        "groq": ("Groq", "GROQ_API_KEY", test_groq),
        "deepseek": ("DeepSeek", "DEEPSEEK_API_KEY", test_deepseek),
        "xai": ("xAI", "XAI_API_KEY", test_xai),
        "huggingfacefireworks": ("Hugging Face", "HUGGING_FACE_API_KEY", test_huggingface),
    }

    working_provider = None
    for provider_id, (provider_name, env_var, test_func) in provider_mapping.items():
        status, message = test_provider_api_key(provider_name, env_var, test_func)
        results.append(
            {
                "name": provider_name,
                "env_var": env_var,
                "status": status,
                "message": message,
                "provider_id": provider_id,
            }
        )
        if status is True and working_provider is None:
            working_provider = provider_id

    # Print results table
    print("\n" + "=" * 80)
    print(f"{'Service':<25} {'Environment Variable':<30} {'Status':<25}")
    print("=" * 80)

    for result in results:
        status_display = result["message"]
        print(f"{result['name']:<25} {result['env_var']:<30} {status_display:<25}")

    print("=" * 80)

    # Determine ready status
    arc_passed = results[0]["status"] is True
    provider_passed = any(r["status"] is True for r in results[1:])

    ready = arc_passed and provider_passed

    print(f"\n{'='*80}")
    if ready:
        print("✓ READY TO BENCHMARK")
        print("  - ARC-AGI-3 API: ✓ Connected")
        print(
            f"  - Provider APIs: {sum(1 for r in results[1:] if r['status'] is True)} configured and working"
        )

        # Generate example command
        if games_list and working_provider:
            # Get first game
            example_game = games_list[0]["game_id"]

            # Find a model config for the working provider
            try:
                import os

                import yaml

                # Load models.yml (same path logic as task_utils)
                base_dir = os.path.dirname(os.path.dirname(__file__))
                models_file = os.path.join(base_dir, "models.yml")

                if os.path.exists(models_file):
                    with open(models_file, "r") as f:
                        models_data = yaml.safe_load(f)

                    # Find first model for this provider
                    example_config = None
                    for model in models_data.get("models", []):
                        if model.get("provider") == working_provider:
                            example_config = model.get("name")
                            break

                    if example_config:
                        print("\n  Example command:")
                        print("  uv run python -m arcagi3.runner \\")
                        print("    --agent adcr \\")
                        print(f"    --game_id {example_game} \\")
                        print(f"    --config {example_config} \\")
                        print("    --max_actions 3")
            except Exception:
                # If we can't find a model, just show a generic example
                example_game = games_list[0]["game_id"]
                print("\n  Example command:")
                print("  uv run python -m arcagi3.runner \\")
                print("    --agent adcr \\")
                print(f"    --game_id {example_game} \\")
                print("    --config <model-config> \\")
                print("    --max_actions 3")
    else:
        print("✗ NOT READY TO BENCHMARK")
        if not arc_passed:
            print("  - ARC-AGI-3 API: ✗ Not connected")
        if not provider_passed:
            print("  - Provider APIs: ✗ No working provider configured")
    print("=" * 80 + "\n")
