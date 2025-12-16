#!/usr/bin/env python3
"""
Script to run all models in parallel with limited actions to verify they work correctly
and that checkpoints are being created properly.
"""
import asyncio
import os
import sys
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path (go up one level from scripts/ to project root)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)

from arcagi3.arc3tester import ARC3Tester
from arcagi3.checkpoint import CheckpointManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_all_model_names() -> List[str]:
    """Extract all model names from models.yml"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_file = os.path.join(project_root, "src", "arcagi3", "models.yml")
    
    with open(models_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    model_names = []
    if 'models' in config_data:
        for model in config_data['models']:
            if 'name' in model:
                model_names.append(model['name'])
    
    return sorted(model_names)


async def run_model_test(
    model_name: str,
    game_id: str,
    max_actions: int = 3,
    checkpoint_frequency: int = 1
) -> Dict[str, Any]:
    """
    Run a single model test with limited actions and verify checkpoint creation.
    
    Returns:
        Dictionary with test results including checkpoint verification
    """
    result = {
        'model': model_name,
        'success': False,
        'error': None,
        'checkpoint_created': False,
        'checkpoint_path': None,
        'card_id': None,
        'actions_taken': 0,
    }
    
    try:
        logger.info(f"Starting test for model: {model_name}")
        
        # Create tester with limited actions and checkpoint frequency
        tester = ARC3Tester(
            config=model_name,
            save_results_dir=None,  # Don't save results, just test
            overwrite_results=False,
            max_actions=max_actions,
            retry_attempts=2,
            api_retries=2,
            num_plays=1,
            max_episode_actions=0,
            show_images=False,
            use_vision=True,
            checkpoint_frequency=checkpoint_frequency,
            close_on_exit=False,  # Keep scorecard open to preserve checkpoint
        )
        
        # Run the game (this will be wrapped in asyncio.to_thread)
        def run_game():
            return tester.play_game(game_id)
        
        # Run in thread pool to avoid blocking
        game_result = await asyncio.to_thread(run_game)
        
        if game_result:
            result['success'] = True
            result['card_id'] = getattr(game_result, 'card_id', None)
            result['actions_taken'] = len(game_result.actions) if game_result.actions else 0
            
            # Verify checkpoint was created
            if result['card_id']:
                checkpoint_mgr = CheckpointManager(result['card_id'])
                if checkpoint_mgr.checkpoint_exists():
                    result['checkpoint_created'] = True
                    result['checkpoint_path'] = str(checkpoint_mgr.checkpoint_path)
                    
                    # Verify checkpoint has required files
                    required_files = ['metadata.json', 'costs.json', 'action_history.json']
                    missing_files = []
                    for req_file in required_files:
                        if not (checkpoint_mgr.checkpoint_path / req_file).exists():
                            missing_files.append(req_file)
                    
                    if missing_files:
                        result['error'] = f"Checkpoint missing files: {missing_files}"
                        result['checkpoint_created'] = False
                    else:
                        logger.info(f"✓ {model_name}: Checkpoint verified at {result['checkpoint_path']}")
                else:
                    result['error'] = f"Checkpoint not found for card_id: {result['card_id']}"
                    logger.warning(f"⚠ {model_name}: {result['error']}")
            else:
                result['error'] = "No card_id returned from game result (may be None)"
                logger.warning(f"⚠ {model_name}: {result['error']}")
        else:
            result['error'] = "Game result is None"
            
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"✗ {model_name}: {type(e).__name__}: {e}")
    
    return result


async def run_all_models_parallel(
    game_id: str,
    max_actions: int = 3,
    checkpoint_frequency: int = 1,
    max_concurrent: int = 10
) -> List[Dict[str, Any]]:
    """
    Run all models in parallel with rate limiting.
    
    Args:
        game_id: Game ID to test with
        max_actions: Maximum actions per model (default: 3)
        checkpoint_frequency: Save checkpoint every N actions (default: 1)
        max_concurrent: Maximum concurrent model runs (default: 10)
    
    Returns:
        List of test results for each model
    """
    model_names = get_all_model_names()
    logger.info(f"Found {len(model_names)} models to test")
    logger.info(f"Testing with game_id: {game_id}, max_actions: {max_actions}")
    logger.info(f"Max concurrent runs: {max_concurrent}\n")
    
    # Create semaphore to limit concurrent runs
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def run_with_semaphore(model_name: str):
        async with semaphore:
            return await run_model_test(model_name, game_id, max_actions, checkpoint_frequency)
    
    # Run all models in parallel
    tasks = [run_with_semaphore(model_name) for model_name in model_names]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append({
                'model': model_names[i],
                'success': False,
                'error': str(result),
                'checkpoint_created': False,
            })
        else:
            processed_results.append(result)
    
    return processed_results


def print_summary(results: List[Dict[str, Any]]):
    """Print summary of test results"""
    total = len(results)
    successful = sum(1 for r in results if r.get('success', False))
    checkpoints_created = sum(1 for r in results if r.get('checkpoint_created', False))
    failed = total - successful
    
    print("\n" + "="*80)
    print("TEST SUMMARY".center(80))
    print("="*80)
    print(f"\nTotal Models Tested: {total}")
    print(f"Successful Runs: {successful}")
    print(f"Failed Runs: {failed}")
    print(f"Checkpoints Created: {checkpoints_created}")
    print(f"Checkpoint Success Rate: {checkpoints_created/successful*100:.1f}%" if successful > 0 else "N/A")
    
    if failed > 0:
        print(f"\n{'='*80}")
        print("FAILED MODELS:")
        print("="*80)
        for result in results:
            if not result.get('success', False):
                print(f"\n✗ {result['model']}")
                print(f"  Error: {result.get('error', 'Unknown error')}")
    
    if checkpoints_created < successful:
        print(f"\n{'='*80}")
        print("MODELS WITH MISSING CHECKPOINTS:")
        print("="*80)
        for result in results:
            if result.get('success', False) and not result.get('checkpoint_created', False):
                print(f"\n⚠ {result['model']}")
                print(f"  Card ID: {result.get('card_id', 'N/A')}")
                print(f"  Error: {result.get('error', 'Checkpoint not found')}")
    
    print(f"\n{'='*80}")
    print("CHECKPOINT VERIFICATION:")
    print("="*80)
    checkpoint_dir = Path(".checkpoint")
    if checkpoint_dir.exists():
        checkpoint_count = len([d for d in checkpoint_dir.iterdir() if d.is_dir()])
        print(f"Total checkpoints found in .checkpoint/: {checkpoint_count}")
    else:
        print("No .checkpoint directory found")
    
    print("="*80 + "\n")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run all models in parallel to verify they work and checkpoints are created"
    )
    parser.add_argument(
        "--game-id",
        type=str,
        default="ls20-016295f7601e",
        help="Game ID to test with (default: ls20-016295f7601e)"
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        default=3,
        help="Maximum actions per model (default: 3)"
    )
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=1,
        help="Save checkpoint every N actions (default: 1)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent model runs (default: 10)"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting parallel model testing...")
    logger.info(f"Game ID: {args.game_id}")
    logger.info(f"Max Actions: {args.max_actions}")
    logger.info(f"Checkpoint Frequency: {args.checkpoint_frequency}")
    logger.info(f"Max Concurrent: {args.max_concurrent}\n")
    
    # Run all models
    results = asyncio.run(
        run_all_models_parallel(
            game_id=args.game_id,
            max_actions=args.max_actions,
            checkpoint_frequency=args.checkpoint_frequency,
            max_concurrent=args.max_concurrent
        )
    )
    
    # Print summary
    print_summary(results)
    
    # Exit with error code if any failures
    successful = sum(1 for r in results if r.get('success', False))
    checkpoints_created = sum(1 for r in results if r.get('checkpoint_created', False))
    
    if successful < len(results) or checkpoints_created < successful:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

