from __future__ import annotations

from arcagi3.my_agent import MyAgent

definition = {
    "name": "my_agent",
    "description": "Minimal starter agent that cycles through available actions",
    "agent_class": MyAgent,
}

agents = [definition]

__all__ = ["definition", "agents"]
