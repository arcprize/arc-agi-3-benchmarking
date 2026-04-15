from __future__ import annotations

import argparse
from collections.abc import Callable

from .model_config import list_model_config_ids


def _print_values(title: str, values: list[str]) -> None:
    print(f"{title}:")
    for value in sorted(values):
        if title == "Games":
            print("-", value.split("-")[0])
        else:
            print("-", value)


def print_requested_resource_lists(
    args: argparse.Namespace,
    *,
    root_url: str,
    fetch_available_games: Callable[[str], list[str]],
) -> bool:
    """Print requested list commands and report whether main() should exit early."""
    requested_lists: list[tuple[str, list[str]]] = []

    if args.list_configs:
        requested_lists.append(("Configs", list_model_config_ids()))
    if args.list_games:
        requested_lists.append(("Games", fetch_available_games(root_url)))

    if not requested_lists:
        return False

    for index, (title, values) in enumerate(requested_lists):
        if index > 0:
            print()
        _print_values(title, values)

    return True
