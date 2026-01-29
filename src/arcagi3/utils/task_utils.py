import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from arcagi3.schemas import GameResult, ModelConfig


def sanitize_filename(name: str) -> str:
    """
    Sanitize a string to be safe for use in filenames and directory names.
    
    Replaces invalid characters (/, \\, :, *, ?, ", <, >, |, \\0) with underscores.
    Also removes leading/trailing spaces and dots, and collapses multiple
    consecutive invalid characters into a single underscore.
    
    Args:
        name: The string to sanitize
        
    Returns:
        A sanitized string safe for use in filenames
        
    Examples:
        >>> sanitize_filename("openai/gpt-5.2")
        'openai_gpt-5.2'
        >>> sanitize_filename("model:name")
        'model_name'
        >>> sanitize_filename("  test  ")
        'test'
    """
    # Characters invalid in filenames (cross-platform safe)
    # / \ : * ? " < > | and null byte
    invalid_chars = r'[/\\:*?"<>|\x00]'
    
    # Replace invalid characters with underscore
    sanitized = re.sub(invalid_chars, '_', name)
    
    # Remove leading/trailing spaces and dots (Windows doesn't allow trailing dots/spaces)
    sanitized = sanitized.strip(' .')
    
    # Collapse multiple consecutive underscores into one
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # If the result is empty (e.g., all invalid chars), use a default
    if not sanitized:
        sanitized = 'unknown'
    
    return sanitized


def find_hints_file() -> Optional[str]:
    """Find hints.yml file in current working directory or project root."""
    cwd = Path.cwd()
    project_root = Path(__file__).parent.parent.parent.parent

    for search_dir in [cwd, project_root]:
        potential_file = search_dir / "hints.yml"
        if potential_file.exists():
            return str(potential_file)
    return None


def load_hints(hints_file: Optional[str] = None, game_id: Optional[str] = None) -> Dict[str, str]:
    """
    Load hints from a YAML file.

    The file should contain a mapping of game_id to hint text (markdown string).

    Args:
        hints_file: Path to hints YAML file. If None, returns empty dict.
        game_id: Optional specific game_id to load hint for. If provided, only returns hint for that game.

    Returns:
        Dictionary mapping game_id to hint text (or single entry if game_id specified)

    Example YAML format:
        ls20-fa137e247ce6: |
            This is a hint for game ls20-fa137e247ce6.
            You should look for patterns in the grid.

        ft09-16726c5b26ff: |
            Another hint for a different game.
    """
    if not hints_file or not os.path.exists(hints_file):
        return {}

    with open(hints_file, "r") as f:
        hints_data = yaml.safe_load(f)

    if not isinstance(hints_data, dict):
        raise ValueError(f"Hints file must contain a dictionary/mapping. Got {type(hints_data)}")

    if game_id:
        if game_id in hints_data:
            return {game_id: str(hints_data[game_id])}
        return {}

    return {game_id: str(hint) for game_id, hint in hints_data.items()}


def generate_scorecard_tags(model_config: ModelConfig) -> List[str]:
    """
    Generate scorecard tags from a ModelConfig object.

    Tags are formatted as "key:value" strings for better categorization
    and filtering on the ARC Prize platform.

    Args:
        model_config: ModelConfig object containing model configuration

    Returns:
        List of tag strings in "key:value" format

    Example:
        >>> config = read_models_config("gpt-4o-mini-2024-07-18")
        >>> tags = generate_scorecard_tags(config)
        >>> print(tags)
        ["model:gpt-4o-mini-2024-07-18", "provider:openai", "api_type:responses", ...]
    """

    def flatten_dict(d: Dict[str, Any], parent_key: str = "") -> List[tuple]:
        """Recursively flatten nested dictionaries into key-value pairs."""
        items = []
        for k, v in d.items():
            # Skip pricing info (internal) and name (redundant)
            if k in ["pricing", "name"]:
                continue

            new_key = f"{parent_key}_{k}" if parent_key else k

            if isinstance(v, dict):
                # Recursively flatten nested dicts
                items.extend(flatten_dict(v, new_key))
            elif v is None:
                # Skip None values
                continue
            else:
                # Handle leaf values
                if isinstance(v, bool):
                    tag_value = str(v).lower()
                else:
                    tag_value = str(v)
                items.append((new_key, tag_value))
        return items

    tags = []

    # Add core fields
    tags.append(f"model:{model_config.model_name}")
    tags.append(f"provider:{model_config.provider}")

    if model_config.api_type:
        tags.append(f"api_type:{model_config.api_type}")

    # Process kwargs to extract all config parameters
    if model_config.kwargs:
        flattened = flatten_dict(model_config.kwargs)
        for key, value in flattened:
            tags.append(f"{key}:{value}")

    return tags


def read_models_config(config: str) -> ModelConfig:
    base_dir = os.path.dirname(os.path.dirname(__file__))
    models_file = os.path.join(base_dir, "models.yml")
    models_private_file = os.path.join(base_dir, "models_private.yml")

    with open(models_file, "r") as f:
        config_data = yaml.safe_load(f)

    if os.path.exists(models_private_file):
        with open(models_private_file, "r") as f:
            private_config_data = yaml.safe_load(f)
            if "models" in private_config_data:
                config_data["models"].extend(private_config_data["models"])

    for model in config_data["models"]:
        if model.get("name") == config:
            return ModelConfig(**model)

    raise ValueError(f"No matching configuration found for '{config}'")


def result_exists(save_results_dir: str, game_id: str) -> bool:
    if not save_results_dir or not os.path.exists(save_results_dir):
        return False
    sanitized_game_id = sanitize_filename(game_id)
    return any(
        filename.startswith(f"{sanitized_game_id}_") and filename.endswith(".json")
        for filename in os.listdir(save_results_dir)
    )


def save_result(save_results_dir: str, game_result: GameResult) -> str:
    os.makedirs(save_results_dir, exist_ok=True)
    timestamp_str = (
        game_result.timestamp.strftime("%Y%m%d_%H%M%S") if game_result.timestamp else "unknown"
    )
    sanitized_game_id = sanitize_filename(game_result.game_id)
    sanitized_config = sanitize_filename(game_result.config)
    result_file = os.path.join(
        save_results_dir, f"{sanitized_game_id}_{sanitized_config}_{timestamp_str}.json"
    )
    with open(result_file, "w") as f:
        json.dump(game_result.model_dump(mode="json"), f, indent=2, default=str)
    return result_file


def save_result_in_timestamped_structure(timestamp_dir: str, game_result: GameResult) -> str:
    sanitized_game_id = sanitize_filename(game_result.game_id)
    game_dir = os.path.join(timestamp_dir, sanitized_game_id)
    os.makedirs(game_dir, exist_ok=True)
    timestamp_str = (
        game_result.timestamp.strftime("%Y%m%d_%H%M%S") if game_result.timestamp else "unknown"
    )
    sanitized_config = sanitize_filename(game_result.config)
    result_file = os.path.join(
        game_dir, f"{sanitized_game_id}_{sanitized_config}_{timestamp_str}.json"
    )
    with open(result_file, "w") as f:
        json.dump(game_result.model_dump(mode="json"), f, indent=2, default=str)
    return result_file


def read_provider_rate_limits() -> dict:
    current_file = os.path.abspath(__file__)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
    provider_config_file = os.path.join(base_dir, "provider_config.yml")

    if not os.path.exists(provider_config_file):
        raise FileNotFoundError(f"provider_config.yml not found at {provider_config_file}")

    with open(provider_config_file, "r") as f:
        rate_limits_data = yaml.safe_load(f)
        if not isinstance(rate_limits_data, dict):
            raise yaml.YAMLError("provider_config.yml root should be a dictionary of providers.")
        for provider, limits in rate_limits_data.items():
            if not isinstance(limits, dict) or "rate" not in limits or "period" not in limits:
                raise yaml.YAMLError(f"Provider '{provider}' must have 'rate' and 'period' keys.")
            if not isinstance(limits["rate"], int) or not isinstance(limits["period"], int):
                raise yaml.YAMLError(
                    f"'rate' and 'period' for provider '{provider}' must be integers."
                )
        return rate_limits_data


def generate_execution_map(timestamp_dir: str) -> Dict[str, Any]:
    execution_map = {"execution_start": None, "games": {}}

    dir_name = os.path.basename(timestamp_dir)
    try:
        if len(dir_name) == 15 and dir_name.count("_") == 1:
            date_part, time_part = dir_name.split("_")
            if len(date_part) == 8 and len(time_part) == 6:
                dt = datetime.strptime(f"{date_part}_{time_part}", "%Y%m%d_%H%M%S")
                execution_map["execution_start"] = dt.isoformat() + "Z"
    except Exception:
        pass

    for item in os.listdir(timestamp_dir):
        item_path = os.path.join(timestamp_dir, item)
        if os.path.isdir(item_path) and not item.startswith("."):
            game_id = item
            result_files = []
            models = set()

            for filename in os.listdir(item_path):
                if filename.endswith(".json"):
                    result_files.append(f"{game_id}/{filename}")
                    if filename.startswith(f"{game_id}_") and filename.endswith(".json"):
                        remaining = filename[len(f"{game_id}_") : -len(".json")]
                        if len(remaining) > 15:
                            potential_timestamp = remaining[-15:]
                            if "_" in potential_timestamp:
                                date_part, time_part = potential_timestamp.split("_")
                                if (
                                    len(date_part) == 8
                                    and len(time_part) == 6
                                    and date_part.isdigit()
                                    and time_part.isdigit()
                                ):
                                    model_name = remaining[:-16]
                                    if model_name:
                                        models.add(model_name)
                        else:
                            models.add(remaining)

            if result_files:
                execution_map["games"][game_id] = {
                    "models": sorted(list(models)),
                    "result_files": sorted(result_files),
                }

    return execution_map


def generate_summary(timestamp_dir: str) -> Dict[str, Any]:
    summary = {
        "execution_start": None,
        "execution_end": None,
        "total_games": 0,
        "total_executions": 0,
        "models_tested": [],
        "games_by_model": {},
        "stats_by_model": {},
        "overall_stats": {
            "total_cost": 0.0,
            "total_tokens": 0,
            "total_duration_seconds": 0.0,
            "wins": 0,
            "game_overs": 0,
            "in_progress": 0,
            "avg_score": 0.0,
            "avg_actions": 0.0,
        },
    }

    dir_name = os.path.basename(timestamp_dir)
    try:
        if len(dir_name) == 15 and dir_name.count("_") == 1:
            date_part, time_part = dir_name.split("_")
            if len(date_part) == 8 and len(time_part) == 6:
                dt = datetime.strptime(f"{date_part}_{time_part}", "%Y%m%d_%H%M%S")
                summary["execution_start"] = dt.isoformat() + "Z"
    except Exception:
        pass

    models_tested = set()
    all_results = []
    latest_timestamp = None

    for item in os.listdir(timestamp_dir):
        item_path = os.path.join(timestamp_dir, item)
        if os.path.isdir(item_path) and not item.startswith("."):
            game_id = item
            for filename in os.listdir(item_path):
                if filename.endswith(".json"):
                    file_path = os.path.join(item_path, filename)
                    try:
                        with open(file_path, "r") as f:
                            result_data = json.load(f)
                            all_results.append(result_data)
                            config = result_data.get("config", "unknown")
                            models_tested.add(config)

                            timestamp_str = result_data.get("timestamp")
                            if timestamp_str:
                                try:
                                    result_dt = datetime.fromisoformat(
                                        timestamp_str.replace("Z", "+00:00")
                                    )
                                    if latest_timestamp is None or result_dt > latest_timestamp:
                                        latest_timestamp = result_dt
                                except Exception:
                                    pass

                            if config not in summary["games_by_model"]:
                                summary["games_by_model"][config] = []
                            summary["games_by_model"][config].append(game_id)
                    except Exception:
                        continue

    summary["models_tested"] = sorted(list(models_tested))
    summary["total_games"] = len(
        set(
            os.path.basename(item)
            for item in os.listdir(timestamp_dir)
            if os.path.isdir(os.path.join(timestamp_dir, item)) and not item.startswith(".")
        )
    )
    summary["total_executions"] = len(all_results)

    if latest_timestamp:
        summary["execution_end"] = latest_timestamp.isoformat() + "Z"

    for model in models_tested:
        model_results = [r for r in all_results if r.get("config") == model]
        total_cost = sum(r.get("total_cost", {}).get("total_cost", 0.0) for r in model_results)
        total_tokens = sum(r.get("usage", {}).get("total_tokens", 0) for r in model_results)
        total_duration = sum(r.get("duration_seconds", 0.0) for r in model_results)
        wins = sum(1 for r in model_results if r.get("final_state") == "WIN")
        game_overs = sum(1 for r in model_results if r.get("final_state") == "GAME_OVER")
        in_progress = sum(1 for r in model_results if r.get("final_state") == "IN_PROGRESS")
        scores = [r.get("final_score", 0) for r in model_results]
        actions = [r.get("actions_taken", 0) for r in model_results]

        summary["stats_by_model"][model] = {
            "total_games": len(model_results),
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "total_duration_seconds": total_duration,
            "avg_cost_per_game": total_cost / len(model_results) if model_results else 0.0,
            "avg_tokens_per_game": total_tokens / len(model_results) if model_results else 0.0,
            "avg_duration_per_game": total_duration / len(model_results) if model_results else 0.0,
            "wins": wins,
            "game_overs": game_overs,
            "in_progress": in_progress,
            "win_rate": wins / len(model_results) if model_results else 0.0,
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "avg_actions": sum(actions) / len(actions) if actions else 0.0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
        }

    if all_results:
        summary["overall_stats"]["total_cost"] = sum(
            r.get("total_cost", {}).get("total_cost", 0.0) for r in all_results
        )
        summary["overall_stats"]["total_tokens"] = sum(
            r.get("usage", {}).get("total_tokens", 0) for r in all_results
        )
        summary["overall_stats"]["total_duration_seconds"] = sum(
            r.get("duration_seconds", 0.0) for r in all_results
        )
        summary["overall_stats"]["wins"] = sum(
            1 for r in all_results if r.get("final_state") == "WIN"
        )
        summary["overall_stats"]["game_overs"] = sum(
            1 for r in all_results if r.get("final_state") == "GAME_OVER"
        )
        summary["overall_stats"]["in_progress"] = sum(
            1 for r in all_results if r.get("final_state") == "IN_PROGRESS"
        )
        scores = [r.get("final_score", 0) for r in all_results]
        actions = [r.get("actions_taken", 0) for r in all_results]
        summary["overall_stats"]["avg_score"] = sum(scores) / len(scores) if scores else 0.0
        summary["overall_stats"]["avg_actions"] = sum(actions) / len(actions) if actions else 0.0

    return summary
