"""
Example CLI demonstrating how to use the ARC-AGI-3 runner with agents.

This file shows the minimal setup needed to run the project:
1. Create an AgentRunner instance
2. Register one or more agents
3. Run the CLI with standard ARC-AGI-3 arguments

Usage:
    python main.py --game_id ls20-016295f7601e --config gpt-4o-2024-11-20

To use a different agent, create your own agent class and register it:
    from my_agent import MyAgent
    runner.register({"name": "my-agent", "agent_class": MyAgent})

Or use the full runner with the ADCR agent pre-registered:
    python -m arcagi3.runner --agent adcr --game_id ... --config ...
"""
from __future__ import annotations

from dotenv import load_dotenv

from arcagi3.adcr_agent.definition import agents as adcr_definition
from arcagi3.runner import AgentRunner


def main_cli(cli_args: list | None = None) -> None:
    """
    Main entry point demonstrating how to set up and run the ARC-AGI-3 runner.

    This is a minimal example that:
    - Loads environment variables (for API keys)
    - Creates a runner instance
    - Registers the ADCR agent
    - Runs the CLI with all standard ARC-AGI-3 arguments
    """
    # Load environment variables (API keys, etc.)
    load_dotenv()

    # Create a runner instance - this manages the agent registry
    runner = AgentRunner()

    # Register the ADCR agent
    runner.register(adcr_definition)

    # Run the CLI - this handles argument parsing, agent instantiation,
    # game execution, checkpointing, and result printing
    runner.run(cli_args)


if __name__ == "__main__":
    main_cli()
