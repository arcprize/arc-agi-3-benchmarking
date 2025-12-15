"""
Main CLI for running ARC-AGI-3 benchmarks on single games.

Usage:
    python main.py --game_id ls20-016295f7601e --config gpt-4o-2024-11-20
"""
import sys
import os
import argparse
import logging
from typing import Optional
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


from arcagi3.utils.cli import configure_logging, validate_args, handle_list_checkpoints, handle_close_scorecard, configure_args, configure_main_args, print_result, apply_env_vars_to_args
from arcagi3.arc3tester import ARC3Tester

load_dotenv()
logger = logging.getLogger(__name__)

def main_cli(cli_args: Optional[list] = None):
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Run ARC-AGI-3 benchmark on a single game"
    )
    # Configure arguments
    configure_args(parser)
    configure_main_args(parser)
    
    # Parse arguments
    args = parser.parse_args(cli_args)
    
    # Apply environment variables to args (for boolean flags and overrides)
    args = apply_env_vars_to_args(args)

    # Configure logging
    configure_logging(args)

    # Handle --list-checkpoints
    if args.list_checkpoints:
        handle_list_checkpoints()
        return

    # Handle --close-scorecard
    if args.close_scorecard:
        handle_close_scorecard(args)
        return

    # Validate arguments
    validate_args(args, parser)
   
    # Set default save directory if not provided
    if not args.save_results_dir:
        args.save_results_dir = f"results/{args.config}"
    
    # Create tester
    tester = ARC3Tester(
        config=args.config,
        save_results_dir=args.save_results_dir,
        overwrite_results=args.overwrite_results,
        max_actions=args.max_actions,
        retry_attempts=args.retry_attempts,
        api_retries=args.retries,
        num_plays=args.num_plays,
        max_episode_actions=args.max_episode_actions,
        show_images=args.show_images,
        use_vision=args.use_vision,
        checkpoint_frequency=args.checkpoint_frequency,
        close_on_exit=args.close_on_exit,
        memory_word_limit=args.memory_limit,
        submit_scorecard=not getattr(args, "no_scorecard_submission", False),
    )

    # Play game (with checkpoint support)
    card_id = args.checkpoint if args.checkpoint else None
    resume_from_checkpoint = bool(args.checkpoint)
    result = tester.play_game(
        args.game_id,
        card_id=card_id,
        resume_from_checkpoint=resume_from_checkpoint
    )
    
    if result:
        print_result(result)
    

if __name__ == "__main__":
    main_cli()

