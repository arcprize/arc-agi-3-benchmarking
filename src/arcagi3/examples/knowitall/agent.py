from __future__ import annotations

from typing import Any, Dict, List, Optional

from arcagi3.agent import HUMAN_ACTIONS, HUMAN_ACTIONS_LIST, MultimodalAgent
from arcagi3.prompts import PromptManager
from arcagi3.schemas import GameStep
from arcagi3.utils.context import SessionContext
from arcagi3.utils.formatting import grid_to_text_matrix
from arcagi3.utils.image import image_to_base64, make_image_block
from arcagi3.utils.parsing import extract_json_from_response


class KnowItAllAgent(MultimodalAgent):
    """
    A single-step agent that receives full game rules up front and returns only
    a concrete action each turn.

    Memory contract:
    - Reads `context.datastore["memory_prompt"]` if present.
    - Does not update memory (action-only output).
    """

    def __init__(
        self,
        *args,
        game_rules: str,
        use_vision: bool = True,
        memory_word_limit: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prompt_manager = PromptManager()
        self.game_rules = game_rules.strip()
        self.use_vision = use_vision

        if memory_word_limit is not None:
            self.memory_word_limit = memory_word_limit
        else:
            try:
                self.memory_word_limit = int(
                    getattr(self.provider.model_config, "kwargs", {}).get(
                        "memory_word_limit", 500
                    )
                )
            except Exception:
                self.memory_word_limit = 500

    def _truncate_memory(self, memory: str) -> str:
        if not memory:
            return ""
        words = memory.split()
        if len(words) <= self.memory_word_limit:
            return memory
        return " ".join(words[: self.memory_word_limit])

    def _available_action_descriptions(self, context: SessionContext) -> List[str]:
        if context.game.available_actions:
            indices = [int(str(a)) for a in context.game.available_actions]
            return [
                HUMAN_ACTIONS[HUMAN_ACTIONS_LIST[i - 1]]
                for i in indices
                if 1 <= i <= len(HUMAN_ACTIONS_LIST)
            ]
        return list(HUMAN_ACTIONS.values())

    def _validate_action(self, context: SessionContext, action_name: str) -> bool:
        if not action_name or not action_name.startswith("ACTION"):
            return False
        if not context.game.available_actions:
            return True
        action_num = action_name.replace("ACTION", "")
        normalized_available = {str(a) for a in context.game.available_actions}
        return action_num in normalized_available

    def step(self, context: SessionContext) -> GameStep:
        action_descriptions = self._available_action_descriptions(context)
        available_actions_list = "\n".join(f"  • {desc}" for desc in action_descriptions)
        json_example_action = (
            f'"{action_descriptions[0]}"' if action_descriptions else '"Move Up"'
        )

        action_instruct = self.prompt_manager.render(
            "action_instruct",
            {
                "available_actions_list": available_actions_list,
                "json_example_action": json_example_action,
            },
        )

        memory = self._truncate_memory(context.datastore.get("memory_prompt", ""))
        prompt_text = f"{self.game_rules}\n\n{action_instruct}"
        if memory:
            prompt_text = f"{self.game_rules}\n\nMemory:\n{memory}\n\n{action_instruct}"

        content: List[Dict[str, Any]] = []
        want_vision = self.use_vision and bool(
            getattr(self.provider.model_config, "is_multimodal", False)
        )
        if want_vision:
            content.extend(
                [make_image_block(image_to_base64(img)) for img in context.frame_images]
            )
        else:
            for i, grid in enumerate(context.frames.frame_grids):
                content.append({"type": "text", "text": f"Frame {i}:\n{grid_to_text_matrix(grid)}"})
        content.append({"type": "text", "text": prompt_text})

        messages = [
            {"role": "system", "content": self.prompt_manager.render("system")},
            {"role": "user", "content": content},
        ]

        response = self.provider.call_with_tracking(context, messages)
        action_message = self.provider.extract_content(response)
        action_dict = extract_json_from_response(action_message)

        action_name = action_dict.get("action")
        if not action_name:
            raise ValueError("No action in response")

        if not self._validate_action(context, str(action_name)):
            raise ValueError(
                f"Invalid action '{action_name}' for available_actions={context.game.available_actions}"
            )

        action_payload: Dict[str, Any] = {"action": action_name}
        action_data = action_dict.get("data")
        if isinstance(action_data, dict) and action_data:
            action_payload["data"] = dict(action_data)
        elif "x" in action_dict or "y" in action_dict:
            action_payload["x"] = action_dict.get("x", 0)
            action_payload["y"] = action_dict.get("y", 0)

        return GameStep(action=action_payload, reasoning={})


__all__ = ["KnowItAllAgent"]

