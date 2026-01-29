"""
Unified runner for ARC-AGI-3 agents with a simple agent registry.

Usage:
    python -m arcagi3.runner --agent adcr --game_id ls20-016295f7601e --config gpt-4o-2024-11-20
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from typing import Any, Dict, Iterable, Optional, Sequence

from dotenv import load_dotenv

from arcagi3.adcr_agent.definition import agents as adcr_definition
from arcagi3.arc3tester import ARC3Tester
from arcagi3.game_client import GameClient
from arcagi3.utils import errors
from arcagi3.utils.cli import (
    apply_env_vars_to_args,
    configure_args,
    configure_logging,
    configure_main_args,
    handle_check,
    handle_close_scorecard,
    handle_list_checkpoints,
    handle_list_games,
    handle_list_models,
    validate_args,
)
from arcagi3.utils.scorecard_output import print_result
from arcagi3.utils.task_utils import sanitize_filename

logger = logging.getLogger(__name__)


class AgentRunner:
    """
    Runner that wires CLI args into an agent registry.
    """

    def __init__(self, agents: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self._agents: Dict[str, Dict[str, Any]] = agents or {}

    def register(self, prepared: Dict[str, Any] | Sequence[Dict[str, Any]]) -> None:
        """
        Register one agent definition, or a sequence of them.
        """
        prepared_list = [prepared] if isinstance(prepared, dict) else list(prepared)
        for entry in prepared_list:
            name = entry["name"]
            if name in self._agents:
                raise ValueError(f"Agent '{name}' is already registered")
            self._agents[name] = entry

    def list_agents(self) -> Iterable[Dict[str, Any]]:
        return [self._agents[name] for name in sorted(self._agents)]

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Run ARC-AGI-3 benchmark on a single game with a registered agent"
        )
        configure_args(parser)

        # Add list commands before configure_main_args to avoid conflicts
        parser.add_argument(
            "--list-agents",
            action="store_true",
            help="List available agents and exit",
        )
        parser.add_argument(
            "--list-games",
            action="store_true",
            help="List available games from the ARC-AGI-3 API and exit",
        )
        parser.add_argument(
            "--list-models",
            action="store_true",
            help="List available models for enabled providers and exit",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Output results in JSON format (use with --list-games)",
        )
        parser.add_argument(
            "--check",
            action="store_true",
            help="Check environment variables and test API keys",
        )

        configure_main_args(parser)

        parser.add_argument(
            "--agent",
            default=os.getenv("AGENT", "adcr"),
            help="Agent name (use --list-agents to see options). Can be set via AGENT env var. Defaults to 'adcr' if not specified.",
        )
        for agent in self.list_agents():
            add_args = agent.get("add_args")
            if add_args:
                add_args(parser)
        return parser

    def _resolve_agent(self, args: argparse.Namespace) -> Dict[str, Any]:
        agent_name = args.agent or "adcr"
        if agent_name not in self._agents:
            available = ", ".join(sorted(self._agents))
            raise ValueError(f"Unknown agent '{agent_name}'. Available: {available}")
        return self._agents[agent_name]

    def _print_agents(self) -> None:
        print("Available agents:")
        for agent in self.list_agents():
            print(f"  - {agent['name']}: {agent.get('description', '')}")

    def run(self, cli_args: Optional[list] = None) -> None:
        parser = self.build_parser()
        args = parser.parse_args(cli_args)
        args = apply_env_vars_to_args(args)

        configure_logging(args)

        if args.list_agents:
            self._print_agents()
            return

        if args.check:
            handle_check()
            return

        if args.list_games:
            try:
                game_client = GameClient()
                handle_list_games(game_client, json_output=getattr(args, "json", False))
            except Exception as e:
                if getattr(args, "json", False):
                    import json

                    print(json.dumps({"error": str(e)}, indent=2))
                else:
                    logger.error(f"Failed to list games: {e}")
            return

        if args.list_models:
            handle_list_models()
            return

        if args.list_checkpoints:
            handle_list_checkpoints()
            return

        if args.close_scorecard:
            handle_close_scorecard(args)
            return

        validate_args(args, parser)

        if not args.save_results_dir:
            sanitized_config = sanitize_filename(args.config)
            args.save_results_dir = f"results/{sanitized_config}"

        prepared = self._resolve_agent(args)
        agent_class = prepared["agent_class"]
        agent_kwargs = {}
        get_kwargs = prepared.get("get_kwargs")
        if get_kwargs:
            agent_kwargs = get_kwargs(args) or {}

        agent_kwargs.update(
            {
                "breakpoints_enabled": bool(getattr(args, "breakpoints", False)),
                "breakpoint_ws_url": getattr(args, "breakpoint_ws_url", "ws://localhost:8765/ws"),
                "breakpoint_schema_path": getattr(args, "breakpoint_schema", None),
            }
        )

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
            submit_scorecard=not getattr(args, "offline", False),
            agent_class=agent_class,
            agent_kwargs=agent_kwargs,
        )

        card_id = args.checkpoint if args.checkpoint else None
        resume_from_checkpoint = bool(args.checkpoint)
        result = tester.play_game(
            args.game_id,
            card_id=card_id,
            resume_from_checkpoint=resume_from_checkpoint,
        )

        if result:
            print_result(result)


def _build_default_registry() -> AgentRunner:
    runner = AgentRunner()
    runner.register(adcr_definition)
    return runner


def main_cli(cli_args: Optional[list] = None) -> None:
    load_dotenv()
    runner = _build_default_registry()
    try:
        runner.run(cli_args)
    except Exception as e:
        trace = traceback.format_exc()
        payload = errors.build_error_payload(e, context={"phase": "cli"}, trace=trace)
        print(errors.format_user_message(payload), file=sys.stderr)
        if not getattr(e, "_friendly_logged", False):
            traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main_cli()
