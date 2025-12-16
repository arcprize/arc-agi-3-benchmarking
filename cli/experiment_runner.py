"""
Experiment runner for ARC-AGI-3 benchmarking.

This script lets you:
- Define sweeps over models, games, and core parameters (via YAML or CLI lists)
- Run up to N agents concurrently (default: 5)
- Resume a multi-run from a manifest YAML, leveraging existing checkpoints

Examples:

    # Basic sweep over 2 models x 3 games, max 5 concurrent agents
    python cli/experiment_runner.py \
      --configs gpt-4o-2024-11-20 claude-sonnet-4-5-20250929 \
      --games ls20-fa137e247ce6 ft09-16726c5b26ff ls20-016295f7601e \
      --max-actions 40 \
      --num-plays 1 3 \
      --max-concurrent 5

    # Define complete experiment in YAML (recommended for repeatability)
    python cli/experiment_runner.py --experiment scripts/experiment_example.yml

    # Use YAML config with CLI overrides (CLI args override YAML on conflict)
    python cli/experiment_runner.py \
      --yaml-config scripts/experiment_example.yml \
      --max-concurrent 8

    # Resume from an existing multi-run manifest
    python cli/experiment_runner.py --resume-from results/multirun/my_run.yml
"""

import argparse
import asyncio
import logging
import os
import sys
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv

# Add src to path (same as main.py)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

load_dotenv()

from arcagi3.arc3tester import ARC3Tester  # type: ignore  # noqa: E402
from arcagi3.checkpoint import CheckpointManager  # type: ignore  # noqa: E402
from arcagi3.game_client import GameClient  # type: ignore  # noqa: E402
from arcagi3.utils.cli import configure_logging  # type: ignore  # noqa: E402

# Import report generation helper
import sys
import os
# Add parent directory to path to import scripts
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
from scripts.multirun_report import generate_multirun_report  # type: ignore  # noqa: E402


logger = logging.getLogger(__name__)


# Defaults aligned with existing single-run CLI
DEFAULT_MAX_ACTIONS = 40
DEFAULT_NUM_PLAYS = 0  # 0 = infinite
DEFAULT_CHECKPOINT_FREQUENCY = 1
DEFAULT_MAX_CONCURRENT = 5


@dataclass
class SweepDefinition:
    configs: List[str]
    games: List[str]
    max_actions: List[int]
    num_plays: List[int]
    max_episode_actions: List[int]
    memory_limits: List[Optional[int]]
    checkpoint_frequencies: List[int]
    use_vision: List[bool]
    show_helper_images: List[bool]


@dataclass
class RunOptions:
    save_results_dir: Optional[str]
    overwrite_results: bool
    retry_attempts: int
    retries: int
    show_images: bool
    close_on_exit: bool
    no_scorecard_submission: bool
    max_concurrent: int
    log_level: str
    verbose: bool
    dry_run: bool


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_list(value: Any) -> Optional[List[Any]]:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [value]


def _first_non_empty(*values: Optional[List[Any]]) -> Optional[List[Any]]:
    for v in values:
        if v:
            return v
    return None


# Valid YAML configuration keys (for validation)
VALID_YAML_KEYS = {
    # Sweep parameters
    "configs",
    "games",
    "max_actions",
    "num_plays",
    "max_episode_actions",
    "memory_limits",
    "checkpoint_frequencies",
    "use_vision",
    "show_helper_images",
    # Run options
    "save_results_dir",
    "overwrite_results",
    "retry_attempts",
    "retries",
    "show_images",
    "close_on_exit",
    "no_scorecard_submission",
    "max_concurrent",
    "sweep_repeats",
    "log_level",
    "verbose",
    # Metadata
    "run_name",
    "manifest_path",
}


def validate_yaml_config(yaml_cfg: Dict[str, Any], yaml_path: Optional[str] = None) -> None:
    """
    Validate that all keys in the YAML config are recognized.
    Raises SystemExit with a clear error message if unknown keys are found.
    """
    if not yaml_cfg:
        return
    
    unknown_keys = set(yaml_cfg.keys()) - VALID_YAML_KEYS
    if unknown_keys:
        path_msg = f" in {yaml_path}" if yaml_path else ""
        error_msg = (
            f"Unknown configuration key(s){path_msg}: {', '.join(sorted(unknown_keys))}\n\n"
            f"Valid configuration keys are:\n"
            f"  Sweep parameters: configs, games, max_actions, num_plays, max_episode_actions, "
            f"memory_limits, checkpoint_frequencies, use_vision, show_helper_images\n"
            f"  Run options: save_results_dir, overwrite_results, retry_attempts, retries, "
            f"show_images, close_on_exit, no_scorecard_submission, max_concurrent, sweep_repeats, "
            f"log_level, verbose\n"
            f"  Metadata: run_name, manifest_path\n\n"
            f"See scripts/EXPERIMENT_YAML.md for documentation."
        )
        raise SystemExit(error_msg)


def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML config file not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"YAML config root must be a mapping, got {type(data)}")
        validate_yaml_config(data, str(yaml_path))
        return data


def build_sweep_definition(args: argparse.Namespace, yaml_cfg: Dict[str, Any]) -> SweepDefinition:
    # configs (models)
    yaml_configs = _ensure_list(yaml_cfg.get("configs"))
    configs = _first_non_empty(args.configs, yaml_configs)
    if not configs:
        if yaml_cfg:
            raise SystemExit(
                "At least one config/model is required. "
                "Specify 'configs' in your YAML file or use --configs flag. "
                "See scripts/experiment_example.yml for an example."
            )
        raise SystemExit("At least one config/model is required (via --configs or YAML 'configs').")

    # games
    yaml_games = _ensure_list(yaml_cfg.get("games"))
    games = _first_non_empty(args.games, yaml_games)
    if not games:
        if yaml_cfg:
            raise SystemExit(
                "At least one game_id is required. "
                "Specify 'games' in your YAML file or use --games flag. "
                "See scripts/experiment_example.yml for an example."
            )
        raise SystemExit("At least one game_id is required (via --games/--game-id or YAML 'games').")

    # max_actions
    yaml_max_actions = _ensure_list(yaml_cfg.get("max_actions"))
    max_actions = _first_non_empty(args.max_actions, yaml_max_actions) or [DEFAULT_MAX_ACTIONS]

    # num_plays
    yaml_num_plays = _ensure_list(yaml_cfg.get("num_plays"))
    num_plays = _first_non_empty(args.num_plays, yaml_num_plays) or [DEFAULT_NUM_PLAYS]

    # max_episode_actions (0 = no limit, default: 0)
    yaml_max_episode_actions = _ensure_list(yaml_cfg.get("max_episode_actions"))
    max_episode_actions = _first_non_empty(args.max_episode_actions, yaml_max_episode_actions) or [0]

    # memory_limits (None means: let model config / default decide)
    yaml_memory_limits = _ensure_list(yaml_cfg.get("memory_limits"))
    memory_limits = _first_non_empty(args.memory_limits, yaml_memory_limits) or [None]

    # checkpoint_frequencies
    yaml_checkpoint_freqs = _ensure_list(yaml_cfg.get("checkpoint_frequencies"))
    checkpoint_frequencies = (
        _first_non_empty(args.checkpoint_frequencies, yaml_checkpoint_freqs) or [DEFAULT_CHECKPOINT_FREQUENCY]
    )

    # use_vision
    yaml_use_vision = _ensure_list(yaml_cfg.get("use_vision"))
    # args.use_vision_values is a list[bool] (from --use-vision / --no-vision flags)
    use_vision_values = _first_non_empty(args.use_vision_values, yaml_use_vision) or [True]
    use_vision = [bool(v) for v in use_vision_values]

    # show_helper_images
    yaml_show_helper_images = _ensure_list(yaml_cfg.get("show_helper_images"))
    # args.show_helper_images_values is a list[bool] (from --show-helper-images / --no-helper-images flags)
    show_helper_images_values = _first_non_empty(args.show_helper_images_values, yaml_show_helper_images) or [True]
    show_helper_images = [bool(v) for v in show_helper_images_values]

    # Deduplicate simple dimensions where it is safe
    configs = sorted(set(configs))
    games = sorted(set(games))

    return SweepDefinition(
        configs=configs,
        games=games,
        max_actions=max_actions,
        num_plays=num_plays,
        max_episode_actions=max_episode_actions,
        memory_limits=memory_limits,
        checkpoint_frequencies=checkpoint_frequencies,
        use_vision=use_vision,
        show_helper_images=show_helper_images,
    )


def build_run_options(
    args: argparse.Namespace,
    yaml_cfg: Dict[str, Any],
    run_name: str,
) -> RunOptions:
    # save_results_dir: CLI > YAML > default (results/multirun/<run_name>)
    default_results_dir = os.path.join("results", "multirun", run_name)
    save_results_dir = args.save_results_dir or yaml_cfg.get("save_results_dir") or default_results_dir

    overwrite_results = bool(
        args.overwrite_results
        if args.overwrite_results is not None
        else yaml_cfg.get("overwrite_results", False)
    )

    retry_attempts = args.retry_attempts or yaml_cfg.get("retry_attempts", 3)
    retries = args.retries or yaml_cfg.get("retries", 3)

    show_images = bool(
        args.show_images
        if args.show_images is not None
        else yaml_cfg.get("show_images", False)
    )
    close_on_exit = bool(
        args.close_on_exit
        if args.close_on_exit is not None
        else yaml_cfg.get("close_on_exit", False)
    )
    no_scorecard_submission = bool(
        args.no_scorecard_submission
        if args.no_scorecard_submission is not None
        else yaml_cfg.get("no_scorecard_submission", False)
    )

    max_concurrent = args.max_concurrent or yaml_cfg.get("max_concurrent", DEFAULT_MAX_CONCURRENT)

    # logging options align with arcagi3.utils.cli.configure_logging
    log_level = args.log_level or yaml_cfg.get("log_level", "INFO")
    verbose = bool(args.verbose or yaml_cfg.get("verbose", False))

    dry_run = bool(args.dry_run)

    return RunOptions(
        save_results_dir=save_results_dir,
        overwrite_results=overwrite_results,
        retry_attempts=retry_attempts,
        retries=retries,
        show_images=show_images,
        close_on_exit=close_on_exit,
        no_scorecard_submission=no_scorecard_submission,
        max_concurrent=max_concurrent,
        log_level=log_level,
        verbose=verbose,
        dry_run=dry_run,
    )


def generate_jobs(sweep: SweepDefinition, sweep_repeats: int = 1) -> List[Dict[str, Any]]:
    """
    Generate job definitions for all unique parameter combinations, repeated
    sweep_repeats times.
    
    Filters out invalid combinations where use_vision=False and show_helper_images=True,
    since helper images only affect runs with vision enabled. However, if filtering would
    remove all use_vision=False combinations, converts one to show_helper_images=False
    instead to preserve at least one non-vision run.
    """
    jobs: List[Dict[str, Any]] = []

    combinations = list(
        product(
            sweep.configs,
            sweep.games,
            sweep.max_actions,
            sweep.num_plays,
            sweep.max_episode_actions,
            sweep.memory_limits,
            sweep.checkpoint_frequencies,
            sweep.use_vision,
            sweep.show_helper_images,
        )
    )

    # Deduplicate combinations once (in case of accidental duplicates in input),
    # but then repeat the full unique set sweep_repeats times.
    # Also filter out invalid combinations: vision=False + show_helper_images=True
    # (helper images don't do anything when vision is disabled)
    unique_combos: List[tuple] = []
    seen_keys = set()
    filtered_count = 0
    no_vision_combos: List[tuple] = []  # Track all use_vision=False combinations
    
    for combo in combinations:
        (config, game_id, max_actions, num_plays, max_episode_actions, 
         memory_limit, checkpoint_freq, use_vision, show_helper_image) = combo
        
        # Track non-vision combinations
        if not use_vision:
            no_vision_combos.append(combo)
        
        # Filter: skip combinations where vision=False and show_helper_images=True
        # Helper images don't do anything when vision is disabled
        if not use_vision and show_helper_image:
            filtered_count += 1
            continue
        
        key = combo
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_combos.append(combo)
    
    # Check if we filtered out all non-vision combinations
    remaining_no_vision = [c for c in unique_combos if not c[7]]  # Index 7 is use_vision
    if len(no_vision_combos) > 0 and len(remaining_no_vision) == 0:
        # We filtered out all non-vision runs. Convert one filtered combo to show_helper_images=False
        # to preserve at least one non-vision run (pick the first one for consistency)
        first_no_vision = no_vision_combos[0]
        (config, game_id, max_actions, num_plays, max_episode_actions, 
         memory_limit, checkpoint_freq, use_vision, _) = first_no_vision
        # Convert to show_helper_images=False
        converted_combo = (config, game_id, max_actions, num_plays, max_episode_actions,
                          memory_limit, checkpoint_freq, use_vision, False)
        if converted_combo not in seen_keys:
            unique_combos.append(converted_combo)
            seen_keys.add(converted_combo)
            logger.info(
                f"Converted one filtered combination (vision=False, show_helper_images=True) "
                f"to vision=False, show_helper_images=False to preserve a non-vision run."
            )
    
    if filtered_count > 0:
        logger.info(
            f"Filtered out {filtered_count} invalid combination(s) where vision=False and "
            f"show_helper_images=True (helper images only apply when vision is enabled)."
        )

    job_counter = 1
    for repeat_index in range(1, max(1, sweep_repeats) + 1):
        for (config, game_id, max_actions, num_plays, max_episode_actions, memory_limit, checkpoint_freq, use_vision, show_helper_image) in unique_combos:
            job_id = f"job-{job_counter:04d}"
            job_counter += 1
            jobs.append(
                {
                    "job_id": job_id,
                    "repeat_index": repeat_index,
                    "config": str(config),
                    "game_id": str(game_id),
                    "max_actions": int(max_actions),
                    "num_plays": int(num_plays),
                    "max_episode_actions": int(max_episode_actions),
                    "memory_limit": memory_limit if memory_limit is None else int(memory_limit),
                    "checkpoint_frequency": int(checkpoint_freq),
                    "use_vision": bool(use_vision),
                    "show_helper_image": bool(show_helper_image),
                    "status": "pending",
                    "created_at": _now_iso(),
                    "started_at": None,
                    "ended_at": None,
                    "final_score": None,
                    "final_state": None,
                    "result_file": None,
                    "checkpoint_id": None,
                    "scorecard_url": None,
                    "error": None,
                }
            )

    return jobs


def create_manifest(
    sweep: SweepDefinition,
    run_options: RunOptions,
    run_name: str,
    manifest_path: Path,
    yaml_cfg_snapshot: Dict[str, Any],
    sweep_repeats: int,
) -> Dict[str, Any]:
    manifest = {
        "version": 1,
        "run_id": str(uuid.uuid4()),
        "run_name": run_name,
        "created_at": _now_iso(),
        "max_concurrent": run_options.max_concurrent,
        "sweep": {
            "configs": sweep.configs,
            "games": sweep.games,
            "max_actions": sweep.max_actions,
            "num_plays": sweep.num_plays,
            "max_episode_actions": sweep.max_episode_actions,
            "memory_limits": sweep.memory_limits,
            "checkpoint_frequencies": sweep.checkpoint_frequencies,
            "use_vision": sweep.use_vision,
            "show_helper_images": sweep.show_helper_images,
            "sweep_repeats": sweep_repeats,
        },
        "run_options": asdict(run_options),
        "yaml_config": yaml_cfg_snapshot,
        "jobs": generate_jobs(sweep, sweep_repeats),
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    logger.info(f"Created manifest with {len(manifest['jobs'])} jobs at: {manifest_path}")
    return manifest


def load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Manifest root must be a mapping, got {type(data)}")
        if "jobs" not in data or not isinstance(data["jobs"], list):
            raise ValueError("Manifest must contain a 'jobs' list.")
        return data


def save_manifest(manifest: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)


def get_job_note(use_vision: bool, show_helper_image: bool) -> str:
    """Generate a descriptive note for a job combination."""
    if use_vision and show_helper_image:
        return "Full vision + helpers"
    elif use_vision and not show_helper_image:
        return "Vision only (no helpers)"
    elif not use_vision and not show_helper_image:
        return "Text-only baseline"
    else:
        # This shouldn't happen due to filtering, but handle it
        return "Text-only (helpers ignored)"


def preview_experiments(jobs: List[Dict[str, Any]], sweep: SweepDefinition, run_options: RunOptions, run_name: str) -> None:
    """Display a human-readable preview of all experiments that will run."""
    if not jobs:
        print("\nNo jobs to preview.\n")
        return
    
    # Extract global parameters (same across all jobs)
    first_job = jobs[0]
    global_params = {
        "game": first_job['game_id'],
        "max_actions": first_job['max_actions'],
        "num_plays": first_job['num_plays'],
        "max_episode_actions": first_job.get('max_episode_actions', 0),
        "memory_limit": first_job['memory_limit'],
        "checkpoint_frequency": first_job['checkpoint_frequency'],
    }
    
    # Verify these are truly global (same for all jobs)
    for job in jobs[1:]:
        if job['game_id'] != global_params['game']:
            global_params['game'] = None  # Not global
        if job['max_actions'] != global_params['max_actions']:
            global_params['max_actions'] = None
        if job.get('max_episode_actions', 0) != global_params['max_episode_actions']:
            global_params['max_episode_actions'] = None
        if job['memory_limit'] != global_params['memory_limit']:
            global_params['memory_limit'] = None
        if job['checkpoint_frequency'] != global_params['checkpoint_frequency']:
            global_params['checkpoint_frequency'] = None
    
    # Identify variation axes
    vision_values = sorted(set(job['use_vision'] for job in jobs))
    helper_values = sorted(set(job.get('show_helper_image', True) for job in jobs))
    
    variation_axes = []
    if len(vision_values) > 1:
        vision_str = ", ".join("On" if v else "Off" for v in vision_values)
        variation_axes.append(("Vision", vision_str))
    if len(helper_values) > 1:
        helper_str = ", ".join("On" if v else "Off" for v in helper_values)
        variation_axes.append(("Helper Images", helper_str))
    
    # Count unique combinations per model (should be same for all models)
    if jobs:
        first_model = jobs[0]['config']
        first_model_combos = set()
        for job in jobs:
            if job['config'] == first_model:
                key = (job['use_vision'], job.get('show_helper_image', True))
                first_model_combos.add(key)
        num_combinations = len(first_model_combos)
    else:
        num_combinations = 0
    
    # Group jobs by model
    jobs_by_model: Dict[str, List[Dict[str, Any]]] = {}
    for job in jobs:
        model = job['config']
        if model not in jobs_by_model:
            jobs_by_model[model] = []
        jobs_by_model[model].append(job)
    
    # Sort jobs within each model by vision, then helper images
    for model in jobs_by_model:
        jobs_by_model[model].sort(
            key=lambda j: (not j['use_vision'], not j.get('show_helper_image', True))
        )
    
    # Print preview
    print("\n" + "=" * 80)
    print("EXPERIMENT PREVIEW")
    print("=" * 80)
    print(f"\nRun Name: {run_name}")
    print(f"Results Directory: {run_options.save_results_dir}")
    print(f"\nTotal Jobs: {len(jobs)}")
    print(f"Models: {len(jobs_by_model)}")
    print(f"Max Concurrent Jobs: {run_options.max_concurrent}")
    
    # Global parameters section
    print("\n" + "-" * 80)
    print("GLOBAL PARAMETERS (shared unless overridden)")
    print("-" * 80)
    if global_params['game']:
        print(f"Game:                   {global_params['game']}")
    if global_params['max_actions'] is not None:
        print(f"Max Actions:            {global_params['max_actions']}")
    if global_params['num_plays'] is not None:
        num_plays_str = "unlimited" if global_params['num_plays'] == 0 else str(global_params['num_plays'])
        print(f"Num Plays:              {num_plays_str}")
    if global_params['max_episode_actions'] is not None:
        ep_actions_str = "unlimited" if global_params['max_episode_actions'] == 0 else str(global_params['max_episode_actions'])
        print(f"Max Episode Actions:    {ep_actions_str}")
    if global_params['memory_limit'] is not None:
        memory_str = "default" if global_params['memory_limit'] is None else str(global_params['memory_limit'])
        print(f"Memory Limit:           {memory_str}")
    if global_params['checkpoint_frequency'] is not None:
        print(f"Checkpoint Frequency:   {global_params['checkpoint_frequency']}")
    
    # Variation axes section
    if variation_axes:
        print("\n" + "-" * 80)
        print("VARIATION AXES")
        print("-" * 80)
        for axis_name, axis_values in variation_axes:
            print(f"{axis_name}:{' ' * (15 - len(axis_name))} [{axis_values}]")
        if num_combinations > 0:
            print(f"({num_combinations} combinations per model selected)")
    
    # Per-model sections
    for model_idx, (model, model_jobs) in enumerate(sorted(jobs_by_model.items())):
        separator = "=" * 80 if model_idx == 0 else "-" * 80
        print(f"\n{separator}")
        print(f"MODEL: {model} ({len(model_jobs)} jobs)")
        print(separator)
        print("\nJob ID   Vision   Helper Images   Notes")
        print("------   ------   -------------   " + "-" * 45)
        
        for job in model_jobs:
            # Extract number from job_id (e.g., "job-0001" -> "#01")
            job_num_str = job['job_id'].replace('job-', '')
            job_num = int(job_num_str) if job_num_str.isdigit() else 0
            job_id = f"#{job_num:02d}"
            vision_str = "On" if job['use_vision'] else "Off"
            helper_str = "On" if job.get('show_helper_image', True) else "Off"
            note = get_job_note(job['use_vision'], job.get('show_helper_image', True))
            print(f"{job_id:<7} {vision_str:<7} {helper_str:<15} {note}")
    
    # Summary section
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"• {len(jobs_by_model)} models")
    print(f"• {num_combinations} parameter combinations per model")
    print(f"• {len(jobs)} total experiment jobs")
    print("\nEach job represents a unique combination of:")
    
    combo_parts = ["Model"]
    if len(vision_values) > 1:
        combo_parts.append("Vision Mode")
    if len(helper_values) > 1:
        combo_parts.append("Helper Image Usage")
    
    print(f"  ({' × '.join(combo_parts)})")
    print("\n" + "=" * 80)


def prepare_jobs_for_resume(manifest: Dict[str, Any]) -> List[int]:
    """
    Inspect jobs and checkpoints for a resumed run.

    Rules:
    - completed jobs are skipped
    - jobs with checkpoint_id:
        - if checkpoint exists: resume_from_checkpoint=True
        - if missing: status -> 'removed', not scheduled
    - jobs without checkpoint_id: scheduled as fresh runs
    """
    jobs = manifest.get("jobs", [])
    now = _now_iso()
    indexes_to_run: List[int] = []

    for idx, job in enumerate(jobs):
        status = job.get("status", "pending")

        # Completed or already explicitly removed: do nothing
        if status in {"completed", "removed"}:
            continue

        checkpoint_id = job.get("checkpoint_id")
        if checkpoint_id:
            mgr = CheckpointManager(str(checkpoint_id))
            if mgr.checkpoint_exists():
                job["resume_from_checkpoint"] = True
                job["status"] = "pending"
                job["updated_at"] = now
                indexes_to_run.append(idx)
            else:
                # Checkpoint vanished; treat as not part of the run
                job["status"] = "removed"
                job["removed_reason"] = "checkpoint_missing"
                job["updated_at"] = now
        else:
            job["resume_from_checkpoint"] = False
            job["status"] = "pending"
            job["updated_at"] = now
            indexes_to_run.append(idx)

    return indexes_to_run


def _find_latest_result_file(save_results_dir: str, game_id: str, config: str) -> Optional[str]:
    """
    Best-effort helper to find the most recent result JSON for a (game, config) pair.
    """
    base = Path(save_results_dir)
    if not base.exists():
        return None

    prefix = f"{game_id}_{config}_"
    candidates = [p for p in base.glob("*.json") if p.name.startswith(prefix)]
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)




def _execute_single_job(job: Dict[str, Any], run_options: RunOptions) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Run a single job synchronously and return:
      - result_info: dict with summary fields (or None on failure)
      - checkpoint_id: the checkpoint card_id (if any)
    """
    config = job["config"]
    game_id = job["game_id"]
    max_actions = job["max_actions"]
    num_plays = job["num_plays"]
    max_episode_actions = job.get("max_episode_actions", 0)  # Default to 0 for backward compatibility
    memory_limit = job["memory_limit"]
    checkpoint_frequency = job["checkpoint_frequency"]
    use_vision = job["use_vision"]
    show_helper_image = job.get("show_helper_image", True)  # Default to True for backward compatibility

    resume_from_checkpoint = bool(job.get("resume_from_checkpoint"))
    checkpoint_id = job.get("checkpoint_id")

    logger.info(
        f"Running job {job['job_id']}: game={game_id}, config={config}, "
        f"max_actions={max_actions}, num_plays={num_plays}, max_episode_actions={max_episode_actions}, "
        f"memory_limit={memory_limit}, checkpoint_freq={checkpoint_frequency}, "
        f"use_vision={use_vision}, show_helper_image={show_helper_image}, resume={resume_from_checkpoint}"
    )

    tester = ARC3Tester(
        config=config,
        save_results_dir=run_options.save_results_dir,
        overwrite_results=run_options.overwrite_results,
        max_actions=max_actions,
        retry_attempts=run_options.retry_attempts,
        api_retries=run_options.retries,
        num_plays=num_plays,
        max_episode_actions=max_episode_actions,
        show_images=run_options.show_images,
        use_vision=use_vision,
        show_helper_image=show_helper_image,
        checkpoint_frequency=checkpoint_frequency,
        close_on_exit=run_options.close_on_exit,
        memory_word_limit=memory_limit,
        submit_scorecard=not run_options.no_scorecard_submission,
    )

    try:
        if resume_from_checkpoint and checkpoint_id:
            result = tester.play_game(game_id, card_id=str(checkpoint_id), resume_from_checkpoint=True)
        else:
            result = tester.play_game(game_id)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Job {job['job_id']} failed with exception: {e}", exc_info=True)
        return None, checkpoint_id

    if not result:
        logger.warning(f"Job {job['job_id']} returned no result (possibly skipped).")
        return None, checkpoint_id

    # Determine checkpoint ownership for this job
    effective_checkpoint_id = checkpoint_id
    if not effective_checkpoint_id and result.card_id:
        # Only record if a checkpoint directory actually exists
        mgr = CheckpointManager(result.card_id)
        if mgr.checkpoint_exists():
            effective_checkpoint_id = result.card_id

    # Best-effort result file detection
    result_file = None
    if run_options.save_results_dir:
        result_file = _find_latest_result_file(run_options.save_results_dir, result.game_id, result.config)

    result_info = {
        "final_score": result.final_score,
        "final_state": result.final_state,
        "scorecard_url": result.scorecard_url,
        "card_id": result.card_id,
        "result_file": result_file,
    }

    return result_info, effective_checkpoint_id


async def run_jobs(manifest: Dict[str, Any], manifest_path: Path, run_options: RunOptions, job_indexes: List[int]) -> None:
    sem = asyncio.Semaphore(run_options.max_concurrent)
    lock = asyncio.Lock()
    jobs = manifest.get("jobs", [])

    async def run_single(index: int) -> None:
        job = jobs[index]
        async with sem:
            async with lock:
                now = _now_iso()
                job["status"] = "running"
                job.setdefault("started_at", now)
                job["updated_at"] = now
                save_manifest(manifest, manifest_path)

            result_info: Optional[Dict[str, Any]] = None
            checkpoint_id: Optional[str] = None

            try:
                result_info, checkpoint_id = await asyncio.to_thread(_execute_single_job, job, run_options)
            except Exception as e:  # noqa: BLE001
                logger.error(f"Unexpected error in job {job.get('job_id')}: {e}", exc_info=True)
                result_info = None

            async with lock:
                now = _now_iso()
                job["ended_at"] = now
                job["updated_at"] = now

                if result_info is None:
                    job["status"] = "failed"
                    if job.get("error") is None:
                        job["error"] = "Execution failed or returned no result"
                else:
                    job["status"] = "completed"
                    job["final_score"] = result_info["final_score"]
                    job["final_state"] = result_info["final_state"]
                    job["scorecard_url"] = result_info["scorecard_url"]
                    job["result_file"] = result_info["result_file"]
                    if checkpoint_id:
                        job["checkpoint_id"] = checkpoint_id

                save_manifest(manifest, manifest_path)

    if not job_indexes:
        logger.info("No jobs to run.")
        return

    logger.info(f"Scheduling {len(job_indexes)} job(s) with max_concurrent={run_options.max_concurrent}")
    tasks = [asyncio.create_task(run_single(idx)) for idx in job_indexes]
    await asyncio.gather(*tasks)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-run orchestrator for ARC-AGI-3 games (sweeps over models/games/settings)."
    )

    # Config sources
    cfg_group = parser.add_mutually_exclusive_group(required=False)
    cfg_group.add_argument(
        "--yaml-config",
        "--experiment",
        dest="yaml_config",
        type=str,
        help="Path to YAML file defining experiment/sweep settings (models/games/parameters). "
        "All CLI flags can be specified in YAML for repeatable experiments. "
        "See scripts/experiment_example.yml for a complete example.",
    )
    cfg_group.add_argument(
        "--resume-from",
        type=str,
        help="Path to an existing multi-run manifest YAML to resume.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional name for this multi-run (used for default results/manifest paths).",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Explicit path for the multi-run manifest YAML "
        "(default: results/multirun/<run_name>.yml for new runs, or value of --resume-from).",
    )

    # Sweepable attributes (can be specified as single values or lists)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Model config names (from models.yml). Provide one or more.",
    )
    parser.add_argument(
        "--games",
        "--game-id",
        dest="games",
        nargs="+",
        default=None,
        help="Game IDs to run. Provide one or more.",
    )
    parser.add_argument(
        "--max-actions",
        dest="max_actions",
        nargs="+",
        type=int,
        default=None,
        help="One or more max_actions values to sweep over.",
    )
    parser.add_argument(
        "--num-plays",
        dest="num_plays",
        nargs="+",
        type=int,
        default=None,
        help="One or more num_plays values to sweep over.",
    )
    parser.add_argument(
        "--max-episode-actions",
        dest="max_episode_actions",
        nargs="+",
        type=int,
        default=None,
        help="One or more max_episode_actions values to sweep over (0 = no limit per episode).",
    )
    parser.add_argument(
        "--memory-limit",
        dest="memory_limits",
        nargs="+",
        type=int,
        default=None,
        help="One or more memory limits (word counts) to sweep over.",
    )
    parser.add_argument(
        "--checkpoint-frequency",
        dest="checkpoint_frequencies",
        nargs="+",
        type=int,
        default=None,
        help="One or more checkpoint frequencies to sweep over.",
    )

    # use_vision sweep: allow True/False combinations
    parser.add_argument(
        "--use-vision",
        dest="use_vision_values",
        action="append_const",
        const=True,
        help="Include runs with vision enabled.",
    )
    parser.add_argument(
        "--no-vision",
        dest="use_vision_values",
        action="append_const",
        const=False,
        help="Include runs with vision disabled.",
    )

    # show_helper_images sweep: allow True/False combinations
    parser.add_argument(
        "--show-helper-images",
        dest="show_helper_images_values",
        action="append_const",
        const=True,
        help="Include runs with helper images enabled.",
    )
    parser.add_argument(
        "--no-helper-images",
        dest="show_helper_images_values",
        action="append_const",
        const=False,
        help="Include runs with helper images disabled.",
    )

    # Run-level options (mostly mirror main CLI)
    parser.add_argument(
        "--save-results-dir",
        type=str,
        default=None,
        help="Directory to save results (default: results/multirun/<run_name>).",
    )
    parser.add_argument(
        "--overwrite-results",
        action="store_true",
        default=None,
        help="Overwrite existing result files.",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=None,
        help="Number of retry attempts for provider API failures (default: 3).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=None,
        help="Number of retry attempts for ARC-AGI-3 API calls (default: 3).",
    )
    parser.add_argument(
        "--show-images",
        action="store_true",
        default=None,
        help="Display game frames in the terminal.",
    )
    parser.add_argument(
        "--close-on-exit",
        action="store_true",
        default=None,
        help="Close scorecard on exit even if game not won (prevents checkpoint resume).",
    )
    parser.add_argument(
        "--no-scorecard-submission",
        action="store_true",
        default=None,
        help="Do not open or close scorecards on the ARC server; run in local-only mode when no existing card_id is provided.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Maximum number of agents to run concurrently (default: 5).",
    )

    parser.add_argument(
        "--sweep-repeats",
        type=int,
        default=None,
        help="Number of times to repeat the entire sweep (default: 1).",
    )

    # Logging options
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Base logging level (default: INFO).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging (DEBUG for app, WARNING for libraries).",
    )

    # Utility flags
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated job combinations and exit without running them.",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Skip confirmation prompt and run experiments immediately.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # Configure logging early (before we do any heavy work).
    # We construct a lightweight namespace with log_level/verbose fields expected by configure_logging.
    log_args = argparse.Namespace(log_level=args.log_level or "INFO", verbose=args.verbose)
    configure_logging(log_args)

    # Determine run_name.
    # If not provided, default to a UUID so each unnamed run is unique.
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = str(uuid.uuid4())

    # Decide if we're in explicit or implicit resume mode and what manifest to use.
    resume_manifest_path: Optional[Path] = None

    if args.resume_from:
        # Explicit resume has highest priority.
        resume_manifest_path = Path(args.resume_from)
    else:
        # No explicit resume requested. If a run_name is provided and a manifest for it
        # already exists at the default location, automatically treat this as a resume.
        default_manifest_for_run = Path(
            args.manifest_path or os.path.join("results", "multirun", f"{run_name}.yml")
        )
        if args.run_name and default_manifest_for_run.exists():
            logger.info(
                "Found existing manifest for run '%s' at %s. "
                "Automatically resuming this run. If you intended to start a new run "
                "instead, delete or rename the manifest file or choose a different --run-name.",
                run_name,
                default_manifest_for_run,
            )
            resume_manifest_path = default_manifest_for_run

    # Resume flow (explicit --resume-from or implicit via existing --run-name)
    if resume_manifest_path:
        manifest_path = resume_manifest_path
        yaml_cfg: Dict[str, Any] = {}
        manifest = load_manifest(manifest_path)

        # For resume, we primarily trust run_options stored in manifest,
        # but allow max_concurrent/log changes via CLI if desired.
        stored_run_opts = manifest.get("run_options", {})
        # Merge CLI overrides onto stored options
        merged_run_opts = {
            **stored_run_opts,
            "max_concurrent": args.max_concurrent or stored_run_opts.get("max_concurrent", DEFAULT_MAX_CONCURRENT),
            "log_level": args.log_level or stored_run_opts.get("log_level", "INFO"),
            "verbose": bool(args.verbose or stored_run_opts.get("verbose", False)),
            "dry_run": bool(args.dry_run),
        }
        run_options = RunOptions(**merged_run_opts)

        if run_options.dry_run:
            logger.info("Dry-run with existing manifest:")
            jobs = manifest.get("jobs", [])
            for job in jobs:
                logger.info(
                    f"{job.get('job_id')}: game={job.get('game_id')}, config={job.get('config')}, "
                    f"status={job.get('status')}, checkpoint_id={job.get('checkpoint_id')}"
                )
            return 0

        # Update manifest with checkpoint / resume decisions
        job_indexes = prepare_jobs_for_resume(manifest)
        save_manifest(manifest, manifest_path)

        if not job_indexes:
            logger.info("No resumable or pending jobs found in manifest. Nothing to do.")
            return 0

        asyncio.run(run_jobs(manifest, manifest_path, run_options, job_indexes))

        # Final summary
        jobs = manifest.get("jobs", [])
        completed = sum(1 for j in jobs if j.get("status") == "completed")
        failed = sum(1 for j in jobs if j.get("status") == "failed")
        removed = sum(1 for j in jobs if j.get("status") == "removed")
        logger.info(
            f"Multi-run resume completed. Completed={completed}, Failed={failed}, "
            f"Removed={removed}, Total={len(jobs)}"
        )
        
        # Generate markdown report
        try:
            generate_multirun_report(manifest, manifest_path)
        except Exception as e:
            logger.warning(f"Failed to generate multirun report: {e}", exc_info=True)
        
        return 0

    # New run: load YAML (if any) and build sweep + run options
    yaml_cfg = load_yaml_config(args.yaml_config)
    
    # If YAML config is provided, log that it's being used
    if args.yaml_config:
        logger.info(f"Loading experiment configuration from: {args.yaml_config}")
        if yaml_cfg.get("run_name"):
            logger.info(f"Experiment name from YAML: {yaml_cfg.get('run_name')}")

    run_options = build_run_options(args, yaml_cfg, run_name)

    # Determine how many times to repeat the entire sweep
    yaml_sweep_repeats = yaml_cfg.get("sweep_repeats")
    if args.sweep_repeats is not None:
        sweep_repeats = args.sweep_repeats
    elif isinstance(yaml_sweep_repeats, int):
        sweep_repeats = yaml_sweep_repeats
    else:
        sweep_repeats = 1

    if sweep_repeats < 1:
        raise SystemExit("--sweep-repeats must be >= 1")

    sweep = build_sweep_definition(args, yaml_cfg)

    # For dry-run, we only need the generated jobs (no manifest written)
    jobs_preview = generate_jobs(sweep, sweep_repeats)
    if args.dry_run:
        logger.info(f"Dry-run: would create {len(jobs_preview)} jobs across {sweep_repeats} sweep repeat(s).")
        for job in jobs_preview:
            logger.info(
                f"{job['job_id']}: repeat={job.get('repeat_index', 1)}, "
                f"game={job['game_id']}, config={job['config']}, "
                f"max_actions={job['max_actions']}, num_plays={job['num_plays']}, "
                f"max_episode_actions={job.get('max_episode_actions', 0)}, "
                f"memory_limit={job['memory_limit']}, checkpoint_freq={job['checkpoint_frequency']}, "
                f"use_vision={job['use_vision']}, show_helper_image={job.get('show_helper_image', True)}"
            )
        return 0

    # Show preview and request confirmation (unless --force is passed)
    if not args.force:
        preview_experiments(jobs_preview, sweep, run_options, run_name)
        try:
            response = input("Proceed with running these experiments? [y/N]: ").strip().lower()
            if response not in ('y', 'yes'):
                print("Experiment run cancelled.")
                return 0
        except (EOFError, KeyboardInterrupt):
            print("\nExperiment run cancelled.")
            return 0

    # Initial manifest path
    manifest_path = Path(
        args.manifest_path or os.path.join("results", "multirun", f"{run_name}.yml"),
    )
    if manifest_path.exists():
        raise SystemExit(
            f"Manifest already exists at {manifest_path}. "
            "Use --resume-from for existing runs or choose a different --run-name/--manifest-path."
        )

    manifest = create_manifest(
        sweep=sweep,
        run_options=run_options,
        run_name=run_name,
        manifest_path=manifest_path,
        yaml_cfg_snapshot=yaml_cfg,
        sweep_repeats=sweep_repeats,
    )

    # Determine which jobs to run (all pending for a fresh run)
    job_indexes = [idx for idx, _ in enumerate(manifest.get("jobs", []))]

    asyncio.run(run_jobs(manifest, manifest_path, run_options, job_indexes))

    # Final summary
    jobs = manifest.get("jobs", [])
    completed = sum(1 for j in jobs if j.get("status") == "completed")
    failed = sum(1 for j in jobs if j.get("status") == "failed")
    removed = sum(1 for j in jobs if j.get("status") == "removed")
    logger.info(
        f"Multi-run completed. Completed={completed}, Failed={failed}, "
        f"Removed={removed}, Total={len(jobs)}"
    )
    
    # Generate markdown report
    try:
        generate_multirun_report(manifest, manifest_path)
    except Exception as e:
        logger.warning(f"Failed to generate multirun report: {e}", exc_info=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


