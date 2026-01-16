from __future__ import annotations

from arcagi3.examples.rules_agent import RulesAgent


flags = {
    "name": "rules",
    "description": "Periodically extracts rules and uses them in decisions",
    "agent_class": RulesAgent,
}

