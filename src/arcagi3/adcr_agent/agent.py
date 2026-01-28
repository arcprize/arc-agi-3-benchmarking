from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from arcagi3.agent import HUMAN_ACTIONS, HUMAN_ACTIONS_LIST, MultimodalAgent
from arcagi3.prompts import PromptManager
from arcagi3.schemas import GameStep
from arcagi3.utils.context import SessionContext
from arcagi3.utils.formatting import grid_to_text_matrix
from arcagi3.utils.image import grid_to_image, image_diff, image_to_base64, make_image_block
from arcagi3.utils.parsing import extract_json_from_response

from .breakpoints import get_adcr_breakpoint_hooks, get_adcr_breakpoint_spec

logger = logging.getLogger(__name__)


class MalformedJsonReset(RuntimeError):
    pass


class ADCRAgent(MultimodalAgent):
    """
    ADCR Agent (Analyze -> Decide -> Convert -> Review).

    This is a reference implementation of a common pattern:
    - Analyze: interpret the outcome of the previous action and optionally update memory
    - Decide: pick a human-level action
    - Convert: map the human-level action into a concrete game action
    - Review: Review what occurred, updating the memory as to what happened and what we learned

    Memory contract:
    - Stored in `context.datastore` under these JSON-serializable keys:
      - "memory_prompt": str
      - "previous_prompt": str
      - "previous_action": dict | None
    """

    def __init__(
        self,
        *args,
        use_vision: bool = True,
        show_images: bool = False,
        memory_word_limit: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Prompts are loaded relative to this module (./prompts/*.prompt).
        self.prompt_manager = PromptManager()
        self.use_vision = use_vision
        self.show_images = show_images

        # Agent-level configuration (NOT session state). If provided, overrides model config.
        if memory_word_limit is not None:
            self.memory_word_limit = memory_word_limit
        else:
            try:
                self.memory_word_limit = int(
                    getattr(self.provider.model_config, "kwargs", {}).get("memory_word_limit", 500)
                )
            except Exception:
                self.memory_word_limit = 500

        self.register_breakpoints(
            runtime_spec=get_adcr_breakpoint_spec(),
            hooks=get_adcr_breakpoint_hooks(self),
        )

    def _get_want_vision(self, context: SessionContext) -> bool:
        """
        Get want_vision from context datastore, or calculate and store it if not
        present.

        This value is cached in the datastore since it depends on agent instance
        properties that don't change during a session.

        Uses .get() with a sentinel to avoid race conditions in the
        check-then-set pattern.
        """
        # Use a sentinel to check if key exists, since False is a valid value
        _SENTINEL = object()
        want_vision = context.datastore.get("want_vision", _SENTINEL)
        if want_vision is _SENTINEL:
            want_vision = self.use_vision and bool(
                getattr(self.provider.model_config, "is_multimodal", False)
            )
            context.datastore["want_vision"] = want_vision
        return want_vision

    def _append_memory_note(self, context: SessionContext, note: str) -> None:
        memory_prompt = context.datastore.get("memory_prompt", "")
        if memory_prompt:
            memory_prompt = f"{memory_prompt}\n\n{note}"
        else:
            memory_prompt = note
        context.datastore["memory_prompt"] = memory_prompt

    def _parse_json_with_retry(
        self,
        context: SessionContext,
        messages: List[Dict[str, Any]],
        step_name: str,
    ) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        last_message = ""
        for attempt in range(2):
            response = self.provider.call_with_tracking(
                context,
                messages,
                step_name=f"{step_name}.attempt_{attempt + 1}",
            )
            action_message = self.provider.extract_content(response)
            last_message = action_message or ""
            try:
                return extract_json_from_response(action_message)
            except Exception as e:
                last_error = e
                logger.warning(
                    "Malformed JSON in %s step (attempt %s/2): %s",
                    step_name,
                    attempt + 1,
                    str(e),
                )
                if attempt == 0:
                    logger.info("Retrying %s step once due to malformed JSON.", step_name)
                    continue

        note = f"Note: produced malformed JSON twice in a row during {step_name}; resetting game."
        self._append_memory_note(context, note)
        logger.error(
            "Two consecutive malformed JSON outputs during %s. Resetting game. "
            "Last error: %s. Last response (truncated): %s",
            step_name,
            last_error,
            last_message[:200],
        )
        raise MalformedJsonReset(note)

    def analyze_outcome_step(self, context: SessionContext) -> str:
        previous_action = context.datastore.get("previous_action")
        if not isinstance(previous_action, dict) or not previous_action:
            return "no previous action"

        level_complete = ""
        if context.game.current_score > context.game.previous_score:
            level_complete = "NEW LEVEL!!!! - Whatever you did must have been good!"

        want_vision = self._get_want_vision(context)
        analyze_instruct = self.prompt_manager.render(
            "analyze_instruct", {"memory_limit": self.memory_word_limit, "use_vision": want_vision}
        )
        memory_prompt = context.datastore.get("memory_prompt", "")
        analyze_prompt = f"{level_complete}\n\n{analyze_instruct}\n\n{memory_prompt}"
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
                    {
                        "type": "text",
                        "text": f"Frame {i+1} (after action):\n{grid_to_text_matrix(grid)}",
                    }
                )
            msg_parts.append({"type": "text", "text": analyze_prompt})

        previous_prompt = context.datastore.get("previous_prompt", "")
        messages = [
            {
                "role": "system",
                "content": self.prompt_manager.render("system", {"use_vision": want_vision}),
            },
            {"role": "user", "content": [{"type": "text", "text": previous_prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": str(previous_action)}]},
            {"role": "user", "content": msg_parts},
        ]

        response = self.provider.call_with_tracking(context, messages, step_name="analyze")
        analysis_message = self.provider.extract_content(response)

        before, _, after = analysis_message.partition("---")
        analysis = before.strip()
        if after.strip():
            context.datastore["memory_prompt"] = after.strip()

        if self.breakpoint_manager:
            latest_frame = context.last_frame_image()
            previous_frame = context.previous_images[-1] if context.previous_images else None
            payload = {
                "analysis": analysis,
                "memory_prompt": context.datastore.get("memory_prompt", ""),
                "memory_word_limit": self.memory_word_limit,
                "latest_frame_image": (
                    {
                        "kind": "image",
                        "data": image_to_base64(latest_frame),
                    }
                    if latest_frame
                    else None
                ),
                "previous_frame_image": (
                    {
                        "kind": "image",
                        "data": image_to_base64(previous_frame),
                    }
                    if previous_frame
                    else None
                ),
            }
            updated = self.breakpoint_manager.pause(
                "analyze.post",
                payload,
                context=context,
                step_name="analyze.post",
                score=context.game.current_score,
            )
            if isinstance(updated.get("analysis"), str):
                analysis = updated["analysis"]
            if "memory_prompt" in updated and isinstance(updated.get("memory_prompt"), str):
                context.datastore["memory_prompt"] = updated["memory_prompt"]
            if "memory_word_limit" in updated:
                try:
                    self.memory_word_limit = int(updated["memory_word_limit"])
                except Exception:
                    pass

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
        json_example_action = f'"{action_descriptions[0]}"' if action_descriptions else '"Move Up"'

        want_vision = self._get_want_vision(context)
        action_instruct = self.prompt_manager.render(
            "action_instruct",
            {
                "available_actions_list": available_actions_list,
                "example_actions": example_actions,
                "json_example_action": json_example_action,
                "use_vision": want_vision,
            },
        )

        memory = context.datastore.get("memory_prompt", "")
        if len(analysis) > 20:
            prompt_text = f"{analysis}\n\n{memory}\n\n{action_instruct}"
        else:
            prompt_text = f"{memory}\n\n{action_instruct}"
        context.datastore["previous_prompt"] = prompt_text

        content: List[Dict[str, Any]] = []
        if want_vision:
            content.extend([make_image_block(image_to_base64(img)) for img in context.frame_images])
        else:
            for i, grid in enumerate(context.frames.frame_grids):
                content.append({"type": "text", "text": f"Frame {i}:\n{grid_to_text_matrix(grid)}"})
        content.append({"type": "text", "text": prompt_text})

        messages = [
            {
                "role": "system",
                "content": self.prompt_manager.render("system", {"use_vision": want_vision}),
            },
            {"role": "user", "content": content},
        ]

        result = self._parse_json_with_retry(context, messages, "decide")

        if self.breakpoint_manager:
            payload = {
                "result": result,
                "memory_prompt": context.datastore.get("memory_prompt", ""),
                "memory_word_limit": self.memory_word_limit,
            }
            updated = self.breakpoint_manager.pause(
                "decide.post",
                payload,
                context=context,
                step_name="decide.post",
                score=context.game.current_score,
            )
            if isinstance(updated.get("result"), dict):
                result = updated["result"]
            if "memory_prompt" in updated and isinstance(updated.get("memory_prompt"), str):
                context.datastore["memory_prompt"] = updated["memory_prompt"]
            if "memory_word_limit" in updated:
                try:
                    self.memory_word_limit = int(updated["memory_word_limit"])
                except Exception:
                    pass

        return result

    def convert_human_to_game_action_step(
        self, context: SessionContext, human_action: str
    ) -> Dict[str, Any]:
        if context.game.available_actions:
            indices = [int(str(a)) for a in context.game.available_actions]
            available_actions_text = "\n".join(
                f"{HUMAN_ACTIONS_LIST[i - 1]} = {HUMAN_ACTIONS[HUMAN_ACTIONS_LIST[i - 1]]}"
                for i in indices
                if 1 <= i <= len(HUMAN_ACTIONS_LIST)
            )
            valid_actions = ", ".join(
                HUMAN_ACTIONS_LIST[i - 1] for i in indices if 1 <= i <= len(HUMAN_ACTIONS_LIST)
            )
        else:
            available_actions_text = "\n".join(
                f"{name} = {desc}" for name, desc in HUMAN_ACTIONS.items()
            )
            valid_actions = ", ".join(HUMAN_ACTIONS_LIST)

        want_vision = self._get_want_vision(context)
        find_action_instruct = self.prompt_manager.render(
            "find_action_instruct",
            {
                "action_list": available_actions_text,
                "valid_actions": valid_actions,
                "use_vision": want_vision,
            },
        )

        content: List[Dict[str, Any]] = []
        if want_vision:
            img = context.last_frame_image()
            if img is not None:
                content.append(make_image_block(image_to_base64(img)))
        else:
            if context.last_frame_grid is not None:
                content.append(
                    {
                        "type": "text",
                        "text": f"Current frame:\n{grid_to_text_matrix(context.last_frame_grid)}",
                    }
                )
        content.append({"type": "text", "text": human_action + "\n\n" + find_action_instruct})

        messages = [
            {
                "role": "system",
                "content": self.prompt_manager.render("system", {"use_vision": want_vision}),
            },
            {"role": "user", "content": content},
        ]

        result = self._parse_json_with_retry(context, messages, "convert")

        if self.breakpoint_manager:
            payload = {
                "result": result,
                "memory_prompt": context.datastore.get("memory_prompt", ""),
                "memory_word_limit": self.memory_word_limit,
            }
            updated = self.breakpoint_manager.pause(
                "convert.post",
                payload,
                context=context,
                step_name="convert.post",
                score=context.game.current_score,
            )
            if isinstance(updated.get("result"), dict):
                result = updated["result"]
            if "memory_prompt" in updated and isinstance(updated.get("memory_prompt"), str):
                context.datastore["memory_prompt"] = updated["memory_prompt"]
            if "memory_word_limit" in updated:
                try:
                    self.memory_word_limit = int(updated["memory_word_limit"])
                except Exception:
                    pass

        return result

    def validate_action(self, context: SessionContext, action_name: str) -> bool:
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
        try:
            analysis = self.analyze_outcome_step(context)

            human_action_dict = self.decide_human_action_step(context, analysis)
            human_action = human_action_dict.get("human_action")
            if not human_action:
                raise ValueError("No human_action in response")

            game_action_dict = self.convert_human_to_game_action_step(context, str(human_action))
            action_name = game_action_dict.get("action")
            if not action_name:
                raise ValueError("No action in game action response")

            if not self.validate_action(context, str(action_name)):
                raise ValueError(
                    f"Invalid action '{action_name}' for available_actions={context.game.available_actions}"
                )

            context.datastore["previous_action"] = human_action_dict

            reasoning = {
                # Keep concise: ARC API has a ~16kb limit for reasoning/metadata.
                "analysis": analysis[:1000],
                "human_action": human_action_dict,
            }

            return GameStep(action=game_action_dict, reasoning=reasoning)
        except MalformedJsonReset as e:
            logger.error("Forcing RESET action due to malformed JSON: %s", e)
            return GameStep(
                action={"action": "RESET"},
                reasoning={"system": "malformed_json_reset"},
            )


__all__ = ["ADCRAgent"]
