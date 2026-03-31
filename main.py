# ruff: noqa: E402
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.example")
load_dotenv(dotenv_path=".env", override=True)

import argparse
import json
import logging
import os
import re
import signal
import sys
import threading
from functools import partial
from pathlib import Path
from types import FrameType
from typing import Optional
from urllib.parse import urlparse

import requests

from agents import AVAILABLE_AGENTS, Swarm
from agents.tracing import initialize as init_agentops

logger = logging.getLogger()

DEFAULT_ARC_BASE_URL = "https://arcprize.org"
SCHEME = os.environ.get("SCHEME", "http")
HOST = os.environ.get("HOST", "localhost")
PORT = os.environ.get("PORT", 8001)
ARC_BASE_URL = os.environ.get("ARC_BASE_URL")
MODEL_CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "agents"
    / "templates"
    / "conversation_rolling_window"
    / "model_configs.yaml"
)
MODEL_CONFIG_NAME_PATTERN = re.compile(
    r'^\s*-\s+name:\s*["\']?([^"\']+)["\']?\s*$',
    re.MULTILINE,
)
RECORDING_SUFFIX = ".recording.jsonl"

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


def list_model_config_ids() -> list[str]:
    if not MODEL_CONFIG_PATH.exists():
        logger.error("Model config file not found: %s", MODEL_CONFIG_PATH)
        return []

    try:
        return MODEL_CONFIG_NAME_PATTERN.findall(MODEL_CONFIG_PATH.read_text())
    except OSError as e:
        logger.error("Failed to read model config file %s: %s", MODEL_CONFIG_PATH, e)
        return []


def is_recording_agent_name(agent_name: str) -> bool:
    return agent_name.endswith(RECORDING_SUFFIX)


def list_agent_names() -> list[str]:
    return sorted(
        agent_name
        for agent_name in AVAILABLE_AGENTS.keys()
        if not is_recording_agent_name(agent_name)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ARC-AGI-3-Agents")
    parser.add_argument(
        "-a",
        "--agent",
        help="Choose which built-in agent to run. Use --list-agents to see built-ins. Recording filenames are also accepted for playback.",
    )
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
    parser.add_argument(
        "--list-agents",
        action="store_true",
        help="List available agent names and exit.",
    )
    return parser


def maybe_handle_list_requests(args: argparse.Namespace) -> bool:
    requested_lists: list[tuple[str, list[str]]] = []

    if args.list_agents:
        requested_lists.append(("Agents", list_agent_names()))
    if args.list_configs:
        requested_lists.append(("Configs", list_model_config_ids()))
    if args.list_games:
        requested_lists.append(("Games", fetch_available_games(ROOT_URL)))

    if not requested_lists:
        return False

    for index, (title, values) in enumerate(requested_lists):
        if index > 0:
            print()
        print(f"{title}:")
        for value in sorted(values):
            if title != "Games":
                print("-", value)
            else:
                print("-", value.split("-")[0])

    return True


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

    if maybe_handle_list_requests(args):
        return

    if not args.agent:
        logger.error("An Agent must be specified")
        return

    if args.agent not in AVAILABLE_AGENTS:
        logger.error(
            "Unknown agent '%s'. Use --list-agents to see built-in agents, or pass a valid .recording.jsonl filename for playback.",
            args.agent,
        )
        return

    print(f"{ROOT_URL}/api/games")

    # Get the list of games from the API
    full_games = fetch_available_games(ROOT_URL)

    # For playback agents, we can derive the game from the recording filename
    if not full_games and args.agent and args.agent.endswith(".recording.jsonl"):
        from agents.recorder import Recorder

        game_prefix = Recorder.get_prefix_one(args.agent)
        full_games = [game_prefix]
        logger.info(
            f"Using game '{game_prefix}' derived from playback recording filename"
        )
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
                "No games available to play. Check API connection or recording file."
            )
        return

    # Start with Empty tags, "agent" and agent name will be added by the Swarm later
    tags: list[str] = []

    # Append user-provided tags if any
    if args.tags:
        user_tags = [tag.strip() for tag in args.tags.split(",")]
        tags.extend(user_tags)

    # Initialize AgentOps client
    init_agentops(api_key=os.getenv("AGENTOPS_API_KEY"), log_level=log_level)

    swarm = Swarm(
        args.agent,
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
