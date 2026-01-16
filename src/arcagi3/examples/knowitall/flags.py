from __future__ import annotations

import argparse

from arcagi3.examples.knowitall import KnowItAllAgent


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--game-rules",
        default="",
        help="Rules text for KnowItAllAgent (required when --agent knowitall)",
    )


def get_kwargs(args):
    if not args.game_rules.strip():
        raise ValueError("KnowItAllAgent requires --game-rules")
    return {"game_rules": args.game_rules}


flags = {
    "name": "knowitall",
    "description": "Action-only agent that uses full game rules at init",
    "agent_class": KnowItAllAgent,
    "get_kwargs": get_kwargs,
    "add_args": add_args,
}

