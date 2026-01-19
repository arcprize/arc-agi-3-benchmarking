from __future__ import annotations

from typing import Any, Dict, List, Optional

from arcagi3.agent import HUMAN_ACTIONS, HUMAN_ACTIONS_LIST, MultimodalAgent
from arcagi3.prompts import PromptManager
from arcagi3.schemas import GameStep
from arcagi3.utils.context import SessionContext
from arcagi3.utils.formatting import grid_to_text_matrix
from arcagi3.utils.image import grid_to_image, image_diff, image_to_base64, make_image_block
from arcagi3.utils.parsing import extract_json_from_response


class CurriculumAgent(MultimodalAgent):
    """
    ADCR-style agent that follows an explicit curriculum:
    - Exploration phase: gather evidence and test mechanics.
    - Refinement phase: consolidate evidence and reduce uncertainty.
    - Execution phase: optimize for progress/win.
    """

    def __init__(
        self,
        *args,
        use_vision: bool = True,
        memory_word_limit: Optional[int] = None,
        exploration_steps: int = 5,
        refinement_steps: int = 10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prompt_manager = PromptManager()
        self.use_vision = use_vision
        self.exploration_steps = max(0, int(exploration_steps))
        self.refinement_steps = max(0, int(refinement_steps))

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

    def _curriculum_phase(self, context: SessionContext) -> str:
        step = context.game.play_action_counter
        if step < self.exploration_steps:
            return "exploration"
        if step < self.exploration_steps + self.refinement_steps:
            return "refinement"
        return "execution"

    def analyze_outcome_step(self, context: SessionContext) -> str:
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
        if want_vision:
            previous_grids = context.frames.previous_grids
            previous_imgs = [grid_to_image(g) for g in previous_grids] if previous_grids else []
            current_imgs = context.frame_images
            if previous_imgs and current_imgs:
                imgs = [
                    previous_imgs[-1],
                    *current_imgs,
                    image_diff(previous_imgs[-1], current_imgs[-1]),
                ]
            else:
                imgs = current_imgs

            msg_parts = [make_image_block(image_to_base64(img)) for img in imgs] + [
                {"type": "text", "text": analyze_prompt}
            ]
        else:
            msg_parts: List[Dict[str, Any]] = []
            if context.frames.previous_grids:
                msg_parts.append(
                    {
                        "type": "text",
                        "text": f"Frame 0 (before action):\n{grid_to_text_matrix(context.frames.previous_grids[-1])}",
                    }
                )
            for i, grid in enumerate(context.frames.frame_grids):
                msg_parts.append(
                    {"type": "text", "text": f"Frame {i+1} (after action):\n{grid_to_text_matrix(grid)}"}
                )
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

    def decide_human_action_step(self, context: SessionContext, analysis: str) -> Dict[str, Any]:
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

        phase = self._curriculum_phase(context)
        curriculum_prompt = self.prompt_manager.render(
            "curriculum_instruct",
            {
                "phase": phase,
                "current_step": context.game.play_action_counter,
                "exploration_steps": self.exploration_steps,
                "refinement_steps": self.refinement_steps,
            },
        )

        memory = context.datastore.get("memory_prompt", "")
        if len(analysis) > 20:
            prompt_text = f"{analysis}\n\n{memory}\n\n{curriculum_prompt}\n\n{action_instruct}"
        else:
            prompt_text = f"{memory}\n\n{curriculum_prompt}\n\n{action_instruct}"
        context.datastore["previous_prompt"] = prompt_text

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
        return extract_json_from_response(action_message)

    def convert_human_to_game_action_step(self, context: SessionContext, human_action: str) -> Dict[str, Any]:
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
        if want_vision:
            img = context.last_frame_image()
            if img is not None:
                content.append(make_image_block(image_to_base64(img)))
        else:
            if context.last_frame_grid is not None:
                content.append(
                    {"type": "text", "text": f"Current frame:\n{grid_to_text_matrix(context.last_frame_grid)}"}
                )
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
        analysis = self.analyze_outcome_step(context)

        human_action_dict = self.decide_human_action_step(context, analysis)
        human_action = human_action_dict.get("human_action")
        if not human_action:
            raise ValueError("No human_action in response")

        game_action_dict = self.convert_human_to_game_action_step(
            context, str(human_action)
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
            "curriculum_phase": self._curriculum_phase(context),
        }

        return GameStep(action=game_action_dict, reasoning=reasoning)


__all__ = ["CurriculumAgent"]




