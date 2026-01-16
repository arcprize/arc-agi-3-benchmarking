from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from PIL import Image

from arcagi3.agent import HUMAN_ACTIONS, HUMAN_ACTIONS_LIST, MultimodalAgent
from arcagi3.prompts import PromptManager
from arcagi3.schemas import GameStep
from arcagi3.utils.context import SessionContext
from arcagi3.utils.formatting import grid_to_text_matrix
from arcagi3.utils.image import grid_to_image, image_diff, image_to_base64, make_image_block
from arcagi3.utils.parsing import extract_json_from_response


@dataclass(frozen=True)
class StateTransformPayload:
    """
    State transform output injected into the ADCR prompts.

    - text: text representation of a transformed state (arbitrary format).
    - images: images representing the transformed state (optional). I the model
        is not multimodal, the images will be ignored.
    - label: heading used in prompt blocks.
    """

    text: Optional[str] = None
    images: Optional[List[Image.Image]] = None
    label: str = "Transformed State"


StateTransform = Callable[[SessionContext], StateTransformPayload]


class StateTransformADCRAgent(MultimodalAgent):
    """
    ADCR variant with a user-supplied state transform step.

    The state transform runs once per turn and its output is injected into:
    - Analyze step
    - Decide step
    - Convert step

    ...replacing the usual input to the model.
    """

    def __init__(
        self,
        *args,
        state_transform: StateTransform,
        use_vision: bool = True,
        memory_word_limit: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prompt_manager = PromptManager()
        self.use_vision = use_vision
        self.state_transform = state_transform

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

    def _transform_blocks(
        self,
        transform_payload: Optional[StateTransformPayload],
        allow_images: bool,
    ) -> List[Dict[str, Any]]:
        if not transform_payload:
            return []
        blocks: List[Dict[str, Any]] = []
        label = transform_payload.label or "Transformed State"

        if transform_payload.text:
            blocks.append({"type": "text", "text": f"{label} (text):\n{transform_payload.text}"})

        if transform_payload.images:
            if allow_images:
                blocks.extend(
                    [make_image_block(image_to_base64(img)) for img in transform_payload.images]
                )
            else:
                blocks.append(
                    {
                        "type": "text",
                        "text": f"{label} (images omitted: model not multimodal).",
                    }
                )

        return blocks

    def _apply_state_transform(self, context: SessionContext) -> StateTransformPayload:
        return self.state_transform(context)

    def analyze_outcome_step(
        self,
        context: SessionContext,
        transform_payload: StateTransformPayload,
    ) -> str:
        previous_action = context.datastore.get("previous_action")
        if not isinstance(previous_action, dict) or not previous_action:
            return "no previous action"

        level_complete = ""
        if context.game.current_score > context.game.previous_score:
            level_complete = "NEW LEVEL!!!! - Whatever you did must have been good!"

        analyze_instruct = self.prompt_manager.render("analyze_instruct", {"memory_limit": self.memory_word_limit})
        memory_prompt = context.datastore.get("memory_prompt", "")
        analyze_prompt = f"{level_complete}\n\n{analyze_instruct}\n\n{memory_prompt}"

        want_vision = self.use_vision and self.provider.model_config.is_multimodal
        transform_blocks = self._transform_blocks(transform_payload, allow_images=want_vision)
        msg_parts = []
        msg_parts.extend(transform_blocks)
        msg_parts.append({"type": "text", "text": analyze_prompt})

        previous_prompt = context.datastore.get("previous_prompt", "")
        messages = [
            {"role": "system", "content": self.prompt_manager.render("system")},
            {"role": "user", "content": [{"type": "text", "text": previous_prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": str(previous_action)}]},
            {"role": "user", "content": msg_parts},
        ]

        response = self.provider.call_with_tracking(context, messages)
        analysis_message = self.provider.extract_content(response)

        before, _, after = analysis_message.partition("---")
        analysis = before.strip()
        if after.strip():
            context.datastore["memory_prompt"] = after.strip()
        return analysis

    def decide_human_action_step(
        self,
        context: SessionContext,
        analysis: str,
        transform_payload: StateTransformPayload,
    ) -> Dict[str, Any]:
        if context.game.available_actions:
            indices = [int(str(a)) for a in context.game.available_actions]
            action_descriptions = [
                HUMAN_ACTIONS[HUMAN_ACTIONS_LIST[i - 1]]
                for i in indices
                if 1 <= i <= len(HUMAN_ACTIONS_LIST)
            ]
        else:
            action_descriptions = list(HUMAN_ACTIONS.values())

        available_actions_list = "\n".join(f"  • {desc}" for desc in action_descriptions)
        example_actions = (
            ", ".join(f'"{desc}"' for desc in action_descriptions[:3])
            if action_descriptions
            else '"Move Up"'
        )
        json_example_action = (
            f'"{action_descriptions[0]}"' if action_descriptions else '"Move Up"'
        )

        action_instruct = self.prompt_manager.render(
            "action_instruct",
            {
                "available_actions_list": available_actions_list,
                "example_actions": example_actions,
                "json_example_action": json_example_action,
            },
        )

        memory = context.datastore.get("memory_prompt", "")
        if len(analysis) > 20:
            prompt_text = f"{analysis}\n\n{memory}\n\n{action_instruct}"
        else:
            prompt_text = f"{memory}\n\n{action_instruct}"
        context.datastore["previous_prompt"] = prompt_text

        content: List[Dict[str, Any]] = []
        want_vision = self.use_vision and bool(
            getattr(self.provider.model_config, "is_multimodal", False)
        )
        transform_blocks = self._transform_blocks(transform_payload, allow_images=want_vision)
        content.extend(transform_blocks)
        content.append({"type": "text", "text": prompt_text})

        messages = [
            {"role": "system", "content": self.prompt_manager.render("system")},
            {"role": "user", "content": content},
        ]

        response = self.provider.call_with_tracking(context, messages)
        action_message = self.provider.extract_content(response)
        return extract_json_from_response(action_message)

    def convert_human_to_game_action_step(
        self,
        context: SessionContext,
        human_action: str,
        transform_payload: StateTransformPayload,
    ) -> Dict[str, Any]:
        if context.game.available_actions:
            indices = [int(str(a)) for a in context.game.available_actions]
            available_actions_text = "\n".join(
                f"{HUMAN_ACTIONS_LIST[i - 1]} = {HUMAN_ACTIONS[HUMAN_ACTIONS_LIST[i - 1]]}"
                for i in indices
                if 1 <= i <= len(HUMAN_ACTIONS_LIST)
            )
            valid_actions = ", ".join(
                HUMAN_ACTIONS_LIST[i - 1]
                for i in indices
                if 1 <= i <= len(HUMAN_ACTIONS_LIST)
            )
        else:
            available_actions_text = "\n".join(
                f"{name} = {desc}" for name, desc in HUMAN_ACTIONS.items()
            )
            valid_actions = ", ".join(HUMAN_ACTIONS_LIST)

        find_action_instruct = self.prompt_manager.render(
            "find_action_instruct",
            {"action_list": available_actions_text, "valid_actions": valid_actions},
        )

        content: List[Dict[str, Any]] = []
        want_vision = self.use_vision and bool(
            getattr(self.provider.model_config, "is_multimodal", False)
        )
        transform_blocks = self._transform_blocks(transform_payload, allow_images=want_vision)
        content.extend(transform_blocks)
        content.append(
            {"type": "text", "text": human_action + "\n\n" + find_action_instruct}
        )

        messages = [
            {"role": "system", "content": self.prompt_manager.render("system")},
            {"role": "user", "content": content},
        ]

        response = self.provider.call_with_tracking(context, messages)
        action_message = self.provider.extract_content(response)
        return extract_json_from_response(action_message)

    def _validate_action(self, context: SessionContext, action_name: str) -> bool:
        if not action_name or not action_name.startswith("ACTION"):
            return False
        if not context.game.available_actions:
            return True
        try:
            action_num = action_name.replace("ACTION", "")
            normalized_available = {str(a) for a in context.game.available_actions}
            return action_num in normalized_available
        except Exception:
            return False

    def step(self, context: SessionContext) -> GameStep:
        transform_payload = self._apply_state_transform(context)

        analysis = self.analyze_outcome_step(context, transform_payload)

        human_action_dict = self.decide_human_action_step(context, analysis, transform_payload)
        human_action = human_action_dict.get("human_action")
        if not human_action:
            raise ValueError("No human_action in response")

        game_action_dict = self.convert_human_to_game_action_step(
            context, str(human_action), transform_payload
        )
        action_name = game_action_dict.get("action")
        if not action_name:
            raise ValueError("No action in game action response")

        if not self._validate_action(context, str(action_name)):
            raise ValueError(
                f"Invalid action '{action_name}' for available_actions={context.game.available_actions}"
            )

        context.datastore["previous_action"] = human_action_dict

        reasoning = {
            "analysis": analysis[:1000],
            "human_action": human_action_dict,
            "state_transform": {
                "text": transform_payload.text if transform_payload else None,
                "image_count": len(transform_payload.images) if transform_payload and transform_payload.images else 0,
            },
        }

        return GameStep(action=game_action_dict, reasoning=reasoning)


__all__ = ["StateTransformADCRAgent", "StateTransformPayload", "StateTransform"]


