from __future__ import annotations

from arcagi3.conversation_rolling_window import ConversationRollingWindow

definition = {
    "name": "conversation_rolling_window",
    "description": "Rolling text conversation agent with action parsing and prompt trimming",
    "agent_class": ConversationRollingWindow,
}

agents = [definition]

__all__ = ["definition", "agents"]
