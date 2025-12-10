"""Parallel execution orchestrator for ARC-AGI-3 benchmarking."""
import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_dir = os.path.join(project_root, 'src')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from arcagi3.utils import (AsyncRequestRateLimiter, generate_execution_map,
                           generate_summary, read_models_config,
                           read_provider_rate_limits,
                           save_result_in_timestamped_structure)
from main import ARC3Tester

logger = logging.getLogger(__name__)

DEFAULT_RATE_LIMIT_RATE = 400
DEFAULT_RATE_LIMIT_PERIOD = 60
DEFAULT_MODEL_CONFIGS_TO_TEST: List[str] = ["gpt-4o-mini-2024-07-18"]

PROVIDER_RATE_LIMITERS: Dict[str, AsyncRequestRateLimiter] = {}
MODEL_CONFIG_CACHE: Dict[str, Any] = {}


def get_model_config(config_name: str):
    if config_name not in MODEL_CONFIG_CACHE:
        MODEL_CONFIG_CACHE[config_name] = read_models_config(config_name)
    return MODEL_CONFIG_CACHE[config_name]


def get_or_create_rate_limiter(provider_name: str, all_provider_limits: Dict) -> AsyncRequestRateLimiter:
    if provider_name not in PROVIDER_RATE_LIMITERS:
        if provider_name not in all_provider_limits:
            rate = DEFAULT_RATE_LIMIT_RATE / DEFAULT_RATE_LIMIT_PERIOD
            capacity = max(1.0, rate)
        else:
            limits = all_provider_limits[provider_name]
            if limits['period'] <= 0:
                rate = float('inf')
                capacity = float('inf')
            else:
                rate = limits['rate'] / limits['period']
                capacity = max(1.0, rate)
        PROVIDER_RATE_LIMITERS[provider_name] = AsyncRequestRateLimiter(rate=rate, capacity=capacity)
    return PROVIDER_RATE_LIMITERS[provider_name]


async def run_single_game_wrapper(
    config_name: str,
    game_id: str,
    limiter: AsyncRequestRateLimiter,
    timestamp_dir: str,
    overwrite_results: bool,
    max_actions: int,
    retry_attempts: int,
    api_retries: int,
    num_plays: int,
    use_vision: bool,
) -> bool:
    def _synchronous_game_execution():
        tester = ARC3Tester(
            config=config_name,
            save_results_dir=None,
            overwrite_results=overwrite_results,
            max_actions=max_actions,
            retry_attempts=retry_attempts,
            api_retries=api_retries,
            num_plays=num_plays,
            use_vision=use_vision,
        )
        result = tester.play_game(game_id)
        if result:
            save_result_in_timestamped_structure(timestamp_dir, result)
        return result is not None

    try:
        async with limiter:
            success = await asyncio.to_thread(_synchronous_game_execution)
        if success:
            logger.info(f"✓ {config_name} / {game_id}")
        return success
    except Exception as e:
        logger.error(f"Failed: {config_name} / {game_id} - {type(e).__name__}: {e}")
        return False


async def main(
    game_list_file: Optional[str],
    model_configs_to_test: List[str],
    results_root: str,
    overwrite_results: bool,
    max_actions: int,
    retry_attempts: int,
    api_retries: int,
    num_plays: int,
    use_vision: bool,
) -> int:
    start_time = time.perf_counter()
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    timestamp_dir = os.path.join(results_root, timestamp_str)
    os.makedirs(timestamp_dir, exist_ok=True)

    try:
        with open(game_list_file, 'r') as f:
            game_ids = [line.strip() for line in f if line.strip()]
        if not game_ids:
            logger.error(f"No game IDs found in {game_list_file}")
            return 1
    except FileNotFoundError:
        logger.error(f"Game list file not found: {game_list_file}")
        return 1
    except Exception as e:
        logger.error(f"Error loading games: {e}")
        return 1

    all_jobs_to_run = [(config_name, game_id) for config_name in model_configs_to_test for game_id in game_ids]
    if not all_jobs_to_run:
        logger.warning("No jobs to run")
        return 1

    try:
        all_provider_limits = read_provider_rate_limits()
    except Exception:
        all_provider_limits = {}

    async_tasks_to_execute = []
    for config_name, game_id in all_jobs_to_run:
        try:
            model_config_obj = get_model_config(config_name)
            limiter = get_or_create_rate_limiter(model_config_obj.provider, all_provider_limits)
            async_tasks_to_execute.append(
                run_single_game_wrapper(
                    config_name, game_id, limiter, timestamp_dir,
                    overwrite_results, max_actions, retry_attempts, api_retries, num_plays, use_vision
                )
            )
        except Exception as e:
            logger.error(f"Skipping {config_name}/{game_id}: {e}")

    if not async_tasks_to_execute:
        logger.warning("No tasks to execute")
        return 1

    logger.info(f"\nRunning {len(async_tasks_to_execute)} game executions in parallel...\n")
    results = await asyncio.gather(*async_tasks_to_execute, return_exceptions=True)

    successful_runs = sum(1 for r in results if r is True)
    orchestrator_level_failures = sum(1 for r in results if r is False or isinstance(r, Exception))

    summary = None
    try:
        execution_map = generate_execution_map(timestamp_dir)
        with open(os.path.join(timestamp_dir, "execution_map.json"), 'w') as f:
            json.dump(execution_map, f, indent=2)
        summary = generate_summary(timestamp_dir)
        with open(os.path.join(timestamp_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
    except Exception as e:
        logger.error(f"Error generating summary: {e}")

    total_duration = time.perf_counter() - start_time

    logger.info("\n" + "="*80)
    logger.info("EXECUTION SUMMARY".center(80))
    logger.info("="*80)
    logger.info(f"\nExecution Info:")
    logger.info(f"   • Started: {summary.get('execution_start', 'N/A') if summary else 'N/A'}")
    logger.info(f"   • Duration: {total_duration:.2f}s")
    logger.info(f"   • Models tested: {', '.join(model_configs_to_test)}")
    logger.info(f"   • Games: {summary.get('total_games', 0) if summary else 0}")
    logger.info(f"   • Total executions: {successful_runs}/{len(results)}")
    
    if summary and summary.get('stats_by_model'):
        logger.info(f"\nResults by Model:")
        for model, stats in summary['stats_by_model'].items():
            logger.info(f"\n   {model}:")
            logger.info(f"      Games: {stats.get('total_games', 0)}  |  Wins: {stats.get('wins', 0)}  |  Win Rate: {stats.get('win_rate', 0) * 100:.1f}%")
            logger.info(f"      Avg Score: {stats.get('avg_score', 0):.0f}  |  Cost: ${stats.get('total_cost', 0):.4f}")
    
    if summary and summary.get('overall_stats'):
        overall = summary['overall_stats']
        logger.info(f"\nOverall Stats:")
        logger.info(f"   • Total Cost: ${overall.get('total_cost', 0):.4f}")
        logger.info(f"   • Total Tokens: {overall.get('total_tokens', 0):,}")
        logger.info(f"   • Wins: {overall.get('wins', 0)}")
        logger.info(f"   • Game Overs: {overall.get('game_overs', 0)}")
        logger.info(f"   • Avg Score: {overall.get('avg_score', 0):.0f}")
    
    exit_code = 0
    if orchestrator_level_failures > 0:
        logger.warning(f"\nFailures: {orchestrator_level_failures}")
        for i, res in enumerate(results):
            if res is False or isinstance(res, Exception):
                config, game_id = all_jobs_to_run[i]
                error_type = type(res).__name__ if isinstance(res, Exception) else "Failed"
                logger.error(f"   • {config}/{game_id}: {error_type}")
        exit_code = 1
    else:
        logger.info(f"\nAll executions completed successfully!")
    
    logger.info(f"\nResults saved to: {timestamp_dir}")
    logger.info("="*80 + "\n")

    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ARC-AGI-3 games concurrently. Games can be specified via a game list file."
    )
    parser.add_argument(
        "--game_list_file",
        type=str,
        default=None,
        required=False,
        help="Path to a .txt file containing game IDs, one per line. Required if not using --game_ids.",
    )
    parser.add_argument(
        "--game_ids",
        type=str,
        default=None,
        required=False,
        help="Comma-separated list of game IDs (e.g., 'ls20-016295f7601e,ls20-fa137e247ce6'). "
        "Alternative to --game_list_file.",
    )
    parser.add_argument(
        "--model_configs",
        type=str,
        default=",".join(DEFAULT_MODEL_CONFIGS_TO_TEST),
        help=f"Comma-separated list of model configuration names to test (from models.yml). "
        f"Defaults to: {','.join(DEFAULT_MODEL_CONFIGS_TO_TEST)}",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="results",
        help="Root folder name to save results under. A timestamped subdirectory will be created. "
        "Defaults to 'results'",
    )
    parser.add_argument(
        "--overwrite_results",
        action="store_true",
        help="Overwrite results if they already exist. Defaults to False",
    )
    parser.add_argument(
        "--max_actions",
        type=int,
        default=40,
        help="Maximum actions per game (default: 40)",
    )
    parser.add_argument(
        "--retry_attempts",
        type=int,
        default=3,
        help="Number of retry attempts for API failures (default: 3)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retry attempts for ARC-AGI-3 API calls (default: 3)",
    )
    parser.add_argument(
        "--num_plays",
        type=int,
        default=1,
        help="Number of times to play each game (continues session with memory) (default: 1)",
    )
    parser.add_argument(
        "--use_vision",
        action="store_true",
        help="Use vision to play the game (default: True)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NONE"],
        help="Set the logging level (default: INFO). Use NONE to disable logging.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (shows debug info for arcagi3 only, keeps libraries quiet)",
    )

    args = parser.parse_args()

    if args.log_level == "NONE":
        logging.basicConfig(level=logging.CRITICAL + 1, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    elif args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        for lib_logger in ['openai', 'httpx', 'httpcore', 'urllib3', 'requests', 'anthropic', 'google', 'pydantic', 'transformers']:
            logging.getLogger(lib_logger).setLevel(logging.WARNING)
        logging.getLogger('arcagi3').setLevel(logging.DEBUG)
        logging.getLogger('__main__').setLevel(logging.DEBUG)
    else:
        logging.basicConfig(
            level=getattr(logging, args.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    game_ids_list = []
    if args.game_ids:
        game_ids_list = [g.strip() for g in args.game_ids.split(',') if g.strip()]
    elif not args.game_list_file:
        logger.error("Either --game_list_file or --game_ids must be provided.")
        sys.exit(1)

    model_configs_list = [m.strip() for m in args.model_configs.split(',') if m.strip()]
    if not model_configs_list:
        model_configs_list = DEFAULT_MODEL_CONFIGS_TO_TEST

    game_list_file = args.game_list_file
    temp_file = None
    if game_ids_list and not game_list_file:
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        temp_file.write('\n'.join(game_ids_list))
        temp_file.close()
        game_list_file = temp_file.name

    try:
        exit_code = asyncio.run(
            main(
                game_list_file=game_list_file,
                model_configs_to_test=model_configs_list,
                results_root=args.results_root,
                overwrite_results=args.overwrite_results,
                max_actions=args.max_actions,
                retry_attempts=args.retry_attempts,
                api_retries=args.retries,
                num_plays=args.num_plays,
                use_vision=args.use_vision,
            )
        )
        sys.exit(exit_code)
    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

