from pathlib import Path
from typing import Any

import yaml

MODEL_CONFIG_PATH = Path(__file__).resolve().parent / "model_configs.yaml"


def load_model_configs() -> list[dict[str, Any]]:
    if not MODEL_CONFIG_PATH.exists():
        raise ValueError(f"Model config file not found: {MODEL_CONFIG_PATH}")

    try:
        configs = yaml.safe_load(MODEL_CONFIG_PATH.read_text()) or []
    except OSError as e:
        raise ValueError(f"Failed to read model config file {MODEL_CONFIG_PATH}: {e}") from e

    if not isinstance(configs, list):
        raise ValueError(f"Model config file is invalid: {MODEL_CONFIG_PATH}")

    return configs


def list_model_config_ids() -> list[str]:
    return [
        name
        for entry in load_model_configs()
        if isinstance(entry, dict)
        for name in [entry.get("name")]
        if isinstance(name, str) and name.strip()
    ]


def get_model_config(config_id: str) -> dict[str, Any]:
    for entry in load_model_configs():
        if entry.get("name") == config_id:
            return entry

    available_configs = ", ".join(sorted(list_model_config_ids()))
    raise ValueError(
        f"Model config '{config_id}' not found in {MODEL_CONFIG_PATH}. "
        f"Available configs: {available_configs}"
    )
