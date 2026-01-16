"""
Unified runner for ARC-AGI-3 agents with a simple prepared-agent registry.

Usage:
    python -m arcagi3.runner --agent adcr --game_id ls20-016295f7601e --config gpt-4o-2024-11-20
"""
from __future__ import annotations

import argparse
import logging
from typing import Any, Dict, Iterable, Optional, Sequence

from dotenv import load_dotenv

from arcagi3.arc3tester import ARC3Tester
from arcagi3.examples.adcr.flags import flags as adcr_flags
from arcagi3.examples.curriculum_agent.flags import flags as curriculum_flags
from arcagi3.examples.hypothesis_agent.flags import flags as hypothesis_flags
from arcagi3.examples.knowitall.flags import flags as knowitall_flags
from arcagi3.examples.rules_agent.flags import flags as rules_flags
from arcagi3.examples.state_transform_adcr.flags import flags as state_transform_flags
from arcagi3.examples.swarm_agent.flags import flags as swarm_flags
from arcagi3.utils.cli import (
    apply_env_vars_to_args,
    configure_args,
    configure_logging,
    configure_main_args,
    handle_close_scorecard,
    handle_list_checkpoints,
    print_result,
    validate_args,
)

logger = logging.getLogger(__name__)


class AgentRunner:
    """
    Runner that wires CLI args into a prepared-agent registry.
    """

    def __init__(self, agents: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self._agents: Dict[str, Dict[str, Any]] = agents or {}

    def add_flags(self, flags: Dict[str, Any] | Sequence[Dict[str, Any]]) -> None:
        if isinstance(flags, dict):
            flags_list = [flags]
        else:
            flags_list = list(flags)
        for entry in flags_list:
            self.register(entry)

    def register(self, prepared: Dict[str, Any]) -> None:
        name = prepared["name"]
        if name in self._agents:
            raise ValueError(f"Agent '{name}' is already registered")
        self._agents[name] = prepared

    def list_agents(self) -> Iterable[Dict[str, Any]]:
        return [self._agents[name] for name in sorted(self._agents)]

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Run ARC-AGI-3 benchmark on a single game with a prepared agent"
        )
        configure_args(parser)
        configure_main_args(parser)

        parser.add_argument(
            "--agent",
            default="adcr",
            help="Prepared agent name (use --list-agents to see options)",
        )
        parser.add_argument(
            "--list-agents",
            action="store_true",
            help="List available prepared agents and exit",
        )
        for agent in self.list_agents():
            add_args = agent.get("add_args")
            if add_args:
                add_args(parser)
        return parser

    def _resolve_agent(self, args: argparse.Namespace) -> Dict[str, Any]:
        if args.agent not in self._agents:
            available = ", ".join(sorted(self._agents))
            raise ValueError(f"Unknown agent '{args.agent}'. Available: {available}")
        return self._agents[args.agent]

    def _print_agents(self) -> None:
        print("Available prepared agents:")
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

        if args.list_checkpoints:
            handle_list_checkpoints()
            return

        if args.close_scorecard:
            handle_close_scorecard(args)
            return

        validate_args(args, parser)

        if not args.save_results_dir:
            args.save_results_dir = f"results/{args.config}"

        prepared = self._resolve_agent(args)
        agent_class = prepared["agent_class"]
        agent_kwargs = {}
        get_kwargs = prepared.get("get_kwargs")
        if get_kwargs:
            agent_kwargs = get_kwargs(args) or {}

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
            submit_scorecard=not getattr(args, "no_scorecard_submission", False),
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
    runner.add_flags(
        [
            adcr_flags,
            curriculum_flags,
            hypothesis_flags,
            knowitall_flags,
            rules_flags,
            state_transform_flags,
            swarm_flags,
        ]
    )
    return runner


def main_cli(cli_args: Optional[list] = None) -> None:
    load_dotenv()
    runner = _build_default_registry()
    runner.run(cli_args)


if __name__ == "__main__":
    main_cli()

