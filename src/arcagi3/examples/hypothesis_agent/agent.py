from __future__ import annotations

from typing import Any, Dict, List, Optional

from arcagi3.agent import HUMAN_ACTIONS, HUMAN_ACTIONS_LIST, MultimodalAgent
from arcagi3.prompts import PromptManager
from arcagi3.schemas import GameStep
from arcagi3.utils.context import SessionContext
from arcagi3.utils.formatting import grid_to_text_matrix
from arcagi3.utils.image import grid_to_image, image_diff, image_to_base64, make_image_block
from arcagi3.utils.parsing import extract_json_from_response


class HypothesisAgent(MultimodalAgent):
    """
    ADCR-style agent that maintains explicit hypotheses and runs experiments.

    Datastore keys:
    - "memory_prompt": str
    - "previous_prompt": str
    - "previous_action": dict | None
    - "hypotheses": list[dict]
    - "active_hypothesis": str
    - "active_experiment": str
    - "hypothesis_last_extracted_action": int
    """

    def __init__(
        self,
        *args,
        use_vision: bool = True,
        memory_word_limit: Optional[int] = None,
        hypothesis_interval: int = 3,
        hypothesis_window: int = 5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prompt_manager = PromptManager()
        self.use_vision = use_vision
        self.hypothesis_interval = max(1, int(hypothesis_interval))
        self.hypothesis_window = max(1, int(hypothesis_window))

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

    def _format_hypotheses(self, hypotheses: List[Dict[str, Any]]) -> str:
        if not hypotheses:
            return "None"
        lines = []
        for item in hypotheses:
            hyp = str(item.get("hypothesis", "")).strip()
            status = str(item.get("status", "candidate")).strip()
            evidence = str(item.get("evidence", "")).strip()
            if hyp:
                line = f"- {hyp} [{status}]"
                if evidence:
                    line += f" (evidence: {evidence})"
                lines.append(line)
        return "\n".join(lines) if lines else "None"

    def _maybe_update_hypotheses(self, context: SessionContext) -> None:
        action_count = context.game.action_counter
        last_extracted = int(context.datastore.get("hypothesis_last_extracted_action", 0) or 0)
        if action_count <= 0:
            return
        if action_count - last_extracted < self.hypothesis_interval:
            return

        hypotheses, active_hypothesis, active_experiment = self.update_hypotheses_step(context)
        if hypotheses:
            context.datastore["hypotheses"] = hypotheses
        if active_hypothesis:
            context.datastore["active_hypothesis"] = active_hypothesis
        if active_experiment:
            context.datastore["active_experiment"] = active_experiment
        context.datastore["hypothesis_last_extracted_action"] = action_count

    def update_hypotheses_step(
        self, context: SessionContext
    ) -> tuple[List[Dict[str, Any]], str, str]:
        history = list(context.history.actions)
        recent = history[-self.hypothesis_window :] if history else []
        recent_lines = []
        for action in recent:
            action_data = ""
            if action.action_data and (action.action_data.x is not None or action.action_data.y is not None):
                action_data = f" (x={action.action_data.x}, y={action.action_data.y})"
            recent_lines.append(
                f"- #{action.action_num}: {action.action}{action_data} → score {action.result_score}, state {action.result_state}"
            )
        recent_text = "\n".join(recent_lines) if recent_lines else "None"

        hypotheses = context.datastore.get("hypotheses", []) or []
        active_hypothesis = str(context.datastore.get("active_hypothesis", "") or "")
        active_experiment = str(context.datastore.get("active_experiment", "") or "")

        prompt_text = self.prompt_manager.render(
            "hypothesis_instruct",
            {
                "hypotheses": self._format_hypotheses(hypotheses),
                "active_hypothesis": active_hypothesis or "None",
                "active_experiment": active_experiment or "None",
                "recent_actions": recent_text,
            },
        )

        content: List[Dict[str, Any]] = []
        want_vision = self.use_vision and bool(
            getattr(self.provider.model_config, "is_multimodal", False)
        )
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
            content.extend([make_image_block(image_to_base64(img)) for img in imgs])
        else:
            if context.frames.previous_grids:
                content.append(
                    {
                        "type": "text",
                        "text": f"Frame 0 (before action):\n{grid_to_text_matrix(context.frames.previous_grids[-1])}",
                    }
                )
            for i, grid in enumerate(context.frames.frame_grids):
                content.append(
                    {"type": "text", "text": f"Frame {i+1} (after action):\n{grid_to_text_matrix(grid)}"}
                )

        content.append({"type": "text", "text": prompt_text})

        messages = [
            {"role": "system", "content": self.prompt_manager.render("system")},
            {"role": "user", "content": content},
        ]

        response = self.provider.call_with_tracking(context, messages)
        message = self.provider.extract_content(response)
        data = extract_json_from_response(message)

        hypotheses_out = data.get("hypotheses", []) or []
        active_hypothesis_out = str(data.get("active_hypothesis", "") or "")
        active_experiment_out = str(data.get("active_experiment", "") or "")

        normalized = []
        for item in hypotheses_out:
            if isinstance(item, dict):
                normalized.append(
                    {
                        "hypothesis": str(item.get("hypothesis", "")).strip(),
                        "status": str(item.get("status", "candidate")).strip(),
                        "evidence": str(item.get("evidence", "")).strip(),
                    }
                )
            else:
                normalized.append(
                    {"hypothesis": str(item).strip(), "status": "candidate", "evidence": ""}
                )

        return normalized, active_hypothesis_out, active_experiment_out

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

        memory = context.datastore.get("memory_prompt", "")
        hypotheses = context.datastore.get("hypotheses", []) or []
        active_hypothesis = context.datastore.get("active_hypothesis", "")
        active_experiment = context.datastore.get("active_experiment", "")
        hypotheses_block = self._format_hypotheses(hypotheses)

        focus_block = ""
        if active_hypothesis or active_experiment:
            focus_block = (
                f"Active hypothesis: {active_hypothesis or 'None'}\n"
                f"Active experiment: {active_experiment or 'None'}\n"
            )

        if len(analysis) > 20:
            prompt_text = (
                f"{analysis}\n\n{memory}\n\nHypotheses:\n{hypotheses_block}\n\n"
                f"{focus_block}\n{action_instruct}"
            )
        else:
            prompt_text = (
                f"{memory}\n\nHypotheses:\n{hypotheses_block}\n\n"
                f"{focus_block}\n{action_instruct}"
            )
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

    def step(self, context: SessionContext) -> GameStep:
        analysis = self.analyze_outcome_step(context)

        self._maybe_update_hypotheses(context)

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
            "active_hypothesis": context.datastore.get("active_hypothesis", ""),
            "active_experiment": context.datastore.get("active_experiment", ""),
        }

        return GameStep(action=game_action_dict, reasoning=reasoning)


__all__ = ["HypothesisAgent"]


