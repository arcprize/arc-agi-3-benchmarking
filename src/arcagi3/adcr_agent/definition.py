from __future__ import annotations

from arcagi3.adcr_agent import ADCRAgent

definition = {
    "name": "adcr",
    "description": "Analyze -> Decide -> Convert -> Review reference agent",
    "agent_class": ADCRAgent,
}

agents = [definition]

__all__ = ["definition", "agents"]
