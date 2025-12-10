"""
Batch CLI for running multiple ARC-AGI-3 games.

Usage:
    # List available games
    python -m arcagi3.cli --list-games
    
    # Run specific games
    python -m arcagi3.cli \
        --games "ls20-016295f7601e,ft09-16726c5b26ff" \
        --config gpt-4o-2024-11-20
    
    # Run all available games
    python -m arcagi3.cli \
        --all-games \
        --config claude-sonnet-4-5-20250929
"""
import sys
import os
import argparse
import logging
from typing import Optional
from dotenv import load_dotenv
from arcagi3.utils.cli import configure_logging, list_available_games, configure_args, configure_cli_args, handle_list_games, run_batch_games

# Ensure parent directory is in path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from arcagi3.game_client import GameClient


load_dotenv()
logger = logging.getLogger(__name__)


def main_cli(cli_args: Optional[list] = None):
    """Main CLI entry point for batch operations"""
    parser = argparse.ArgumentParser(
        description="Run ARC-AGI-3 benchmarks on multiple games"
    )
    # Configure arguments
    configure_args(parser)
    configure_cli_args(parser)

    # Parse arguments
    args = parser.parse_args(cli_args)
    
    # Configure logging
    configure_logging(args)
    
    # Initialize game client
    game_client = GameClient()
    
    # Handle --list-games
    if args.list_games:
        handle_list_games(game_client)
        return
    
    # Require config for running games
    if not args.config:
        parser.error("--config is required unless using --list-games")
    
    # Determine which games to run
    game_ids = []
    
    if args.all_games:
        logger.info("Fetching all available games...")
        games = list_available_games(game_client)
        game_ids = [g['game_id'] for g in games]
        if not game_ids:
            logger.error("No games available")
            return
        logger.info(f"Found {len(game_ids)} games")
    elif args.games:
        game_ids = [g.strip() for g in args.games.split(',') if g.strip()]
        if not game_ids:
            parser.error("No valid game IDs provided in --games")
    else:
        parser.error("Must specify --games, --all-games, or --list-games")
    
    # Set default save directory
    if not args.save_results_dir:
        args.save_results_dir = f"results/{args.config}"
    
    # Run batch
    run_batch_games(
        game_ids=game_ids,
        config=args.config,
        save_results_dir=args.save_results_dir,
        overwrite_results=args.overwrite_results,
        max_actions=args.max_actions,
        retry_attempts=args.retry_attempts,
        api_retries=args.retries,
        num_plays=args.num_plays,
        show_images=args.show_images,
        memory_word_limit=args.memory_limit,
        checkpoint_frequency=args.checkpoint_frequency,
        close_on_exit=args.close_on_exit,
        use_vision=args.use_vision,
        submit_scorecard=not getattr(args, "no_scorecard_submission", False),
    )


if __name__ == "__main__":
    main_cli()
