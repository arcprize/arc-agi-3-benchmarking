from __future__ import annotations

from typing import Dict, List

from arcagi3.agent import HUMAN_ACTIONS, HUMAN_ACTIONS_LIST, MultimodalAgent
from arcagi3.schemas import GameStep
from arcagi3.utils.context import SessionContext


class MyAgent(MultimodalAgent):
    """
    Minimal starter agent for learning the ARC harness.

    This intentionally does not call the model provider. It shows the smallest
    useful pattern:
    - read state from `context`
    - store durable JSON state in `context.datastore`
    - return a valid `GameStep`
    """

    def _normalize_available_actions(self, context: SessionContext) -> List[str]:
        raw_actions = list(context.game.available_actions) or list(HUMAN_ACTIONS_LIST)
        normalized = []
        for action in raw_actions:
            action_name = str(action)
            if not action_name.startswith("ACTION"):
                action_name = f"ACTION{action_name}"
            normalized.append(action_name)
        return normalized

    def _choose_action(self, available_actions: List[str], step_index: int) -> str:
        non_click_actions = [action for action in available_actions if action != "ACTION6"]
        candidates = non_click_actions or available_actions or list(HUMAN_ACTIONS_LIST)
        return candidates[step_index % len(candidates)]

    def step(self, context: SessionContext) -> GameStep:
        available_actions = self._normalize_available_actions(context)
        step_index = int(context.datastore.get("action_index", 0))
        action_name = self._choose_action(available_actions, step_index)

        context.datastore["action_index"] = step_index + 1
        history = list(context.datastore.get("action_history", []))
        history.append(action_name)
        context.datastore["action_history"] = history[-10:]
        context.datastore["last_decision"] = {
            "step_index": step_index,
            "available_actions": available_actions,
            "chosen_action": action_name,
        }

        action: Dict[str, int | str] = {"action": action_name}
        if action_name == "ACTION6":
            # The harness downscales 0..127 click coordinates to the API grid.
            action["x"] = 63
            action["y"] = 63

        reasoning = {
            "agent": "my_agent",
            "strategy": "cycle_available_actions",
            "step_index": step_index,
            "chosen_action": action_name,
            "chosen_action_meaning": HUMAN_ACTIONS.get(action_name, action_name),
            "available_actions": available_actions,
            "current_score": context.game.current_score,
            "score_increased": context.score_increased,
        }
        return GameStep(action=action, reasoning=reasoning)


__all__ = ["MyAgent"]
