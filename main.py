# ruff: noqa: E402
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)

import argparse
import json
import logging
import os
import signal
import sys
import threading
from functools import partial
from types import FrameType
from typing import Optional
from urllib.parse import urlparse

import requests

from benchmarking import BenchmarkingAgent, Swarm
from benchmarking.cli_list import print_requested_resource_lists
from benchmarking.model_config import get_model_config, list_model_config_ids

logger = logging.getLogger()

DEFAULT_ARC_BASE_URL = "https://arcprize.org"
SCHEME = os.environ.get("SCHEME", "http")
HOST = os.environ.get("HOST", "localhost")
PORT = os.environ.get("PORT", 8001)
ARC_BASE_URL = os.environ.get("ARC_BASE_URL")


def build_root_url() -> str:
    """Prefer ARC_BASE_URL, otherwise use explicit local host settings, else hosted ARC."""
    if ARC_BASE_URL:
        parsed = urlparse(ARC_BASE_URL)
        if parsed.scheme and parsed.netloc:
            return ARC_BASE_URL.rstrip("/")

    # Only use localhost-style settings if the user explicitly configured them.
    if any(key in os.environ for key in ("SCHEME", "HOST", "PORT")):
        # Hide standard ports in URL
        if (SCHEME == "http" and str(PORT) == "80") or (
            SCHEME == "https" and str(PORT) == "443"
        ):
            return f"{SCHEME}://{HOST}"
        return f"{SCHEME}://{HOST}:{PORT}"

    return DEFAULT_ARC_BASE_URL


ROOT_URL = build_root_url()


def build_headers() -> dict[str, str]:
    return {
        "X-API-Key": os.getenv("ARC_API_KEY", ""),
        "Accept": "application/json",
    }


def fetch_available_games(root_url: str) -> list[str]:
    games: list[str] = []
    try:
        with requests.Session() as session:
            session.headers.update(build_headers())
            response = session.get(f"{root_url}/api/games", timeout=10)

        if response.status_code != 200:
            logger.error(
                "API request failed with status %s: %s",
                response.status_code,
                response.text[:200],
            )
            return []

        try:
            payload = response.json()
        except ValueError as e:
            logger.error("Failed to parse games response: %s", e)
            logger.error("Response content: %s", response.text[:200])
            return []

        for game in payload:
            if isinstance(game, dict) and "game_id" in game:
                games.append(str(game["game_id"]))
        return games
    except requests.exceptions.RequestException as e:
        logger.error("Failed to connect to API server: %s", e)
        return []


def validate_required_model_api_key(config_id: Optional[str]) -> None:
    selected_config_id = config_id or BenchmarkingAgent.MODEL_CONFIG_ID
    entry = get_model_config(selected_config_id)

    client_cfg = entry.get("client", {})
    if not isinstance(client_cfg, dict):
        raise ValueError(
            f"Model config '{selected_config_id}' is missing client configuration."
        )

    api_key_env = str(client_cfg.get("api_key_env", "")).strip()
    if not api_key_env:
        raise ValueError(
            f"Model config '{selected_config_id}' is missing client.api_key_env."
        )

    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise ValueError(
            f"No {api_key_env} set. "
            f"The selected model config '{selected_config_id}' requires "
            f"the {api_key_env} environment variable to be set in your .env file."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ARC-AGI-3 Benchmarking")
    parser.add_argument(
        "-g",
        "--game",
        help="Choose a specific game_id for the agent to play. If none specified, an agent swarm will play all available games.",
    )
    parser.add_argument(
        "-t",
        "--tags",
        type=str,
        help="Comma-separated list of tags for the scorecard (e.g., 'experiment,v1.0')",
        default=None,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Model config ID to use (from model_configs.yaml)",
        default=None,
    )
    parser.add_argument(
        "--list-games",
        action="store_true",
        help="List available game IDs and exit.",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available model config IDs and exit.",
    )
    return parser


def run_agent(swarm: Swarm) -> None:
    swarm.main()
    os.kill(os.getpid(), signal.SIGINT)


def cleanup(
    swarm: Swarm,
    signum: Optional[int],
    frame: Optional[FrameType],
) -> None:
    logger.info("Received SIGINT, exiting...")
    card_id = swarm.card_id
    if card_id:
        scorecard = swarm.close_scorecard(card_id)
        if scorecard:
            logger.info("--- EXISTING SCORECARD REPORT ---")
            logger.info(json.dumps(scorecard.model_dump(), indent=2))
            swarm.cleanup(scorecard)

        # Provide web link to scorecard
        if card_id:
            scorecard_url = f"{ROOT_URL}/scorecards/{card_id}"
            logger.info(f"View your scorecard online: {scorecard_url}")

    sys.exit(0)


def main() -> None:
    log_level = logging.INFO
    if os.environ.get("DEBUG", "False") == "True":
        log_level = logging.DEBUG

    logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler("logs.log", mode="w")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    # logging.getLogger("requests").setLevel(logging.CRITICAL)
    # logging.getLogger("werkzeug").setLevel(logging.CRITICAL)

    parser = build_parser()
    args = parser.parse_args()

    if print_requested_resource_lists(
        args,
        root_url=ROOT_URL,
        fetch_available_games=fetch_available_games,
    ):
        return

    try:
        validate_required_model_api_key(args.config)
    except ValueError as e:
        logger.error(str(e))
        return

    print(f"{ROOT_URL}/api/games")

    # Get the list of games from the API
    full_games = fetch_available_games(ROOT_URL)

    games = full_games[:]
    if args.game:
        filters = args.game.split(",")
        games = [
            gid
            for gid in full_games
            if any(gid.startswith(prefix) for prefix in filters)
        ]

    logger.info(f"Game list: {games}")

    if not games:
        if full_games:
            logger.error(
                f"The specified game '{args.game}' does not exist or is not available with your API key. Please try a different game."
            )
        else:
            logger.error(
                "No games available to play. Check API connection."
            )
        return

    # Start with Empty tags, "agent" and agent name will be added by the Swarm later
    tags: list[str] = []

    # Append user-provided tags if any
    if args.tags:
        user_tags = [tag.strip() for tag in args.tags.split(",")]
        tags.extend(user_tags)

    swarm = Swarm(
        ROOT_URL,
        games,
        tags=tags,
        config=args.config,
    )
    agent_thread = threading.Thread(target=partial(run_agent, swarm))
    agent_thread.daemon = True  # die when the main thread dies
    agent_thread.start()

    signal.signal(signal.SIGINT, partial(cleanup, swarm))  # handler for Ctrl+C

    try:
        # Wait for the agent thread to complete
        while agent_thread.is_alive():
            agent_thread.join(timeout=5)  # Check every 5 second
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received in main thread")
        cleanup(swarm, signal.SIGINT, None)
    except Exception as e:
        logger.error(f"Unexpected error in main thread: {e}")
        cleanup(swarm, None, None)


if __name__ == "__main__":
    os.environ["TESTING"] = "False"
    main()
