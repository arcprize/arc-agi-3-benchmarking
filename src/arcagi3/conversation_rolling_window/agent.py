import json
import logging
import math
import os
import re
import textwrap
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from arcagi3.agent import HUMAN_ACTIONS, HUMAN_ACTIONS_LIST, MultimodalAgent
from arcagi3.schemas import GameStep
from arcagi3.utils.context import SessionContext
from arcagi3.utils.formatting import grid_to_text_matrix

from .exceptions import EmptyResponseError
from .models import (
    ActionMetadata,
    CostDetails,
    InputTokensDetails,
    OutputTokensDetails,
    ResponseUsage,
    calculate_cost,
)
from .recording import RunRecord, StepRecord, StepUsage

logger = logging.getLogger(__name__)


class ConversationRollingWindow(MultimodalAgent):
    """Agent that keeps a rolling text conversation with the model.

    This is the current-harness version of the original idea:
    every step appends a text rendering of the latest frames, calls the
    configured provider, parses the model's final action mention, and trims
    old turns from the front when the prompt gets too large.
    """

    MODEL_CONFIG_ID: str = "gpt-5-2-openrouter"
    MAX_RETRIES: int = 3
    MAX_CONTEXT_LENGTH: int = 100000
    MAX_ANIMATION_FRAMES: int = 7
    ESTIMATED_CHARS_PER_TOKEN: float = 1.0

    CONVERSATION_KEY = "conversation_rolling_window.conversation"
    GUID_KEY = "conversation_rolling_window.guid"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.config:
            self.MODEL_CONFIG_ID = self.config

        agent_cfg = self._load_agent_config()

        self.MAX_CONTEXT_LENGTH = int(
            agent_cfg.get("MAX_CONTEXT_LENGTH", self.MAX_CONTEXT_LENGTH)
        )
        self.MAX_ANIMATION_FRAMES = int(
            agent_cfg.get("MAX_ANIMATION_FRAMES", self.MAX_ANIMATION_FRAMES)
        )
        self.MAX_RETRIES = int(agent_cfg.get("MAX_RETRIES", self.MAX_RETRIES))
        self.MODEL = getattr(self.provider.model_config, "model_name", self.MODEL_CONFIG_ID)
        self.step_counter: int = 0
        self.token_counter: int = 0

        run_id = uuid.uuid4()
        self.run_dir = os.path.join("recordings", f"{self.name}.{run_id}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.run_record = RunRecord(
            run_id=str(run_id),
            game_id="",
            agent_name=self.name,
            model=self.MODEL,
            started_at=datetime.now(timezone.utc),
            run_dir=self.run_dir,
        )
        self._write_run_meta()

    def _load_agent_config(self) -> dict[str, Any]:
        """Load optional agent-only settings from model_configs.yaml."""
        cfg_path = Path(__file__).parent / "model_configs.yaml"
        if not cfg_path.exists():
            return {}
        configs = yaml.safe_load(cfg_path.read_text()) or []
        for entry in configs:
            if entry.get("name") == self.MODEL_CONFIG_ID:
                return dict(entry.get("agent", {}))
        logger.info(
            "No conversation_rolling_window-specific settings found for %s; using defaults.",
            self.MODEL_CONFIG_ID,
        )
        return {}

    @property
    def name(self) -> str:
        sanitized = self.MODEL_CONFIG_ID.replace("/", "-").replace(":", "-")
        return f"conversation_rolling_window.{sanitized}.anim{self.MAX_ANIMATION_FRAMES}"

    def _conversation(self, context: SessionContext) -> list[dict[str, Any]]:
        return list(context.datastore.get(self.CONVERSATION_KEY, []))

    def _set_conversation(
        self, context: SessionContext, conversation: list[dict[str, Any]]
    ) -> None:
        context.datastore[self.CONVERSATION_KEY] = conversation

    def _build_system_prompt(self) -> str:
        return textwrap.dedent("""\
            You are playing an ARC-AGI-3 game.
            Your goal is to win.
            Reply with your reasoning if you want, but end with one valid action from the allowed list.

            Valid action formats:
            - RESET
            - ACTION1
            - ACTION2
            - ACTION3
            - ACTION4
            - ACTION5
            - ACTION7
            - ACTION6 x y   where x and y are integers from 0 to 127

            The last valid action in your reply will be executed.
        """)

    def _get_actions(self, context: SessionContext) -> list[str]:
        raw_actions = list(context.game.available_actions) or list(HUMAN_ACTIONS_LIST)
        actions = []
        for raw in raw_actions:
            action_name = str(raw)
            if not action_name.startswith("ACTION"):
                action_name = f"ACTION{action_name}"
            actions.append(action_name)
        if "RESET" not in actions:
            actions.insert(0, "RESET")
        return actions

    def _build_available_actions_text(self, actions: list[str]) -> str:
        lines = []
        for action in actions:
            if action == "RESET":
                lines.append("- RESET  (restart the current level)")
            elif action == "ACTION6":
                lines.append("- ACTION6 x y  (where x and y are integers 0-127)")
            else:
                lines.append(f"- {action}  ({HUMAN_ACTIONS.get(action, action)})")
        return "\n".join(lines)

    def interpolate_frames(
        self, frame_grids: list[list[list[int]]]
    ) -> list[list[list[int]]]:
        n = len(frame_grids)
        target = self.MAX_ANIMATION_FRAMES
        if n <= target:
            return frame_grids
        if target == 1:
            return [frame_grids[-1]]
        indices = [round(i * (n - 1) / (target - 1)) for i in range(target)]
        return [frame_grids[i] for i in indices]

    def build_frame_content(self, context: SessionContext, actions: list[str]) -> str:
        frames = self.interpolate_frames(list(context.frames.frame_grids))

        parts = [
            f"State: {context.game.current_state}\n"
            f"Levels completed: {context.game.current_score}",
        ]

        for i, frame in enumerate(frames):
            frame_lines = []

            if context.score_increased and i == len(frames) - 1:
                frame_lines.append("")
                frame_lines.append("New Level:")
                frame_lines.append("")

            frame_lines.append(f"Frame {i}:")
            frame_lines.append(grid_to_text_matrix(frame))

            parts.append("\n".join(frame_lines))

        actions_text = self._build_available_actions_text(actions)
        parts.append(f"Available actions:\n{actions_text}")

        return "\n\n".join(parts)

    def _parse_action(
        self, text: str, available_actions: list[str]
    ) -> Optional[dict[str, Any]]:
        """Parse the last mentioned action from the assistant's response."""
        text_upper = text.upper()
        candidates: list[tuple[int, dict[str, Any]]] = []

        for action in available_actions:
            if action == "ACTION6":
                pattern = r"ACTION6\s*[:(]?\s*(\d+)\s*[,\s]\s*(\d+)\s*\)?"
                for match in re.finditer(pattern, text_upper):
                    x = int(match.group(1))
                    y = int(match.group(2))
                    if not (0 <= x <= 127 and 0 <= y <= 127):
                        logger.warning(
                            "Ignoring out-of-bounds coordinates for ACTION6: (%s, %s)",
                            x,
                            y,
                        )
                        continue
                    candidates.append((match.start(), {"action": "ACTION6", "x": x, "y": y}))
            else:
                pattern = rf"\b{re.escape(action)}\b"
                for match in re.finditer(pattern, text_upper):
                    candidates.append((match.start(), {"action": action}))

        if not candidates:
            return None

        candidates.sort(key=lambda c: c[0])
        return candidates[-1][1]

    @staticmethod
    def _format_parsed_action(action: dict[str, Any]) -> str | dict[str, Any]:
        if action.get("action") == "ACTION6":
            return dict(action)
        return str(action.get("action"))

    def _write_run_meta(self) -> None:
        path = os.path.join(self.run_dir, "run_meta.json")
        with open(path, "w") as f:
            f.write(self.run_record.model_dump_json(indent=2))

    def _save_diagnostic(self, response: Any) -> None:
        """Dump a raw API response to a diagnostic file for post-mortem debugging."""
        filename = os.path.join(
            self.run_dir,
            f"diagnostic_step_{self.step_counter + 1}_{int(time.time())}.json",
        )
        try:
            raw = (
                response.model_dump()
                if hasattr(response, "model_dump")
                else repr(response)
            )
            with open(filename, "w") as f:
                json.dump(raw, f, indent=2, default=str)
        except Exception as exc:
            with open(filename, "w") as f:
                f.write(f"Failed to serialize response: {exc}\nrepr: {repr(response)}")
        logger.warning(f"Saved diagnostic response to {filename}")

    def _ensure_run_context(self, context: SessionContext) -> None:
        if not self.run_record.game_id and context.game.game_id:
            self.run_record.game_id = context.game.game_id
            self._write_run_meta()

    def _save_step(self, step: StepRecord) -> None:
        self.step_counter += 1
        self.run_record.total_usage = self.run_record.total_usage + step.usage
        self.run_record.total_steps = self.step_counter
        filename = os.path.join(self.run_dir, f"step_{self.step_counter:03d}.json")
        with open(filename, "w") as f:
            f.write(step.model_dump_json(indent=2))
        self._write_run_meta()
        logger.info(f"Saved step {self.step_counter} to {filename}")

    def _estimate_conversation_tokens(self, conversation: list[dict[str, Any]]) -> int:
        total_chars = sum(len(str(m.get("content", ""))) for m in conversation)
        return math.ceil(total_chars / self.ESTIMATED_CHARS_PER_TOKEN)

    def _trim_to_fit_context(self, conversation: list[dict[str, Any]]) -> None:
        estimated = self._estimate_conversation_tokens(conversation)
        while estimated > self.MAX_CONTEXT_LENGTH:
            if not self._trim_oldest_turn(conversation):
                logger.warning(
                    f"Cannot trim further but estimated tokens ({estimated}) "
                    f"still exceed MAX_CONTEXT_LENGTH ({self.MAX_CONTEXT_LENGTH})."
                )
                break
            estimated = self._estimate_conversation_tokens(conversation)
            logger.info(
                f"Proactive context trim: ~{estimated} tokens "
                f"(limit {self.MAX_CONTEXT_LENGTH}), "
                f"{len(conversation)} messages remaining."
            )

    def _extract_reasoning(self, response: Any) -> str | None:
        try:
            choices = response.get("choices") if isinstance(response, dict) else response.choices
            if not choices:
                return None
            msg = choices[0].get("message") if isinstance(choices[0], dict) else choices[0].message
            if isinstance(msg, dict):
                value = msg.get("reasoning") or msg.get("reasoning_content")
            else:
                value = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
            return str(value) if value else None
        except Exception:
            return None

    def _build_step_usage(self, response: Any) -> StepUsage:
        prompt_tokens, completion_tokens, reasoning_tokens = self.provider.extract_usage(response)
        pricing = getattr(self.provider.model_config, "pricing", None)
        input_cost = 0.0
        output_cost = 0.0
        if pricing is not None:
            input_cost = calculate_cost(prompt_tokens, float(getattr(pricing, "input", 0.0)))
            output_cost = calculate_cost(
                completion_tokens + reasoning_tokens,
                float(getattr(pricing, "output", 0.0)),
            )
        total_cost = input_cost + output_cost
        return StepUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            reasoning_tokens=reasoning_tokens,
            cost=total_cost,
            cost_details={
                "input_cost": input_cost,
                "output_cost": output_cost,
            },
        )

    def _request_with_retries(
        self,
        context: SessionContext,
        messages: list[dict[str, Any]],
        actions: list[str],
    ) -> tuple[str, str | None, dict[str, Any], StepUsage, int]:
        step_usage = StepUsage()
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                response = self.provider.call_with_tracking(
                    context,
                    messages,
                    step_name=f"conversation_rolling_window.attempt_{attempt + 1}",
                )
                assistant_text = self.provider.extract_content(response) or ""
                if not assistant_text.strip():
                    raise EmptyResponseError("Provider returned an empty assistant message.")
            except EmptyResponseError:
                logger.warning(
                    f"Empty API response "
                    f"(attempt {attempt + 1}/{self.MAX_RETRIES + 1})."
                )
                continue
            except Exception as e:
                logger.warning(
                    f"API error: {type(e).__name__}: {e} "
                    f"(attempt {attempt + 1}/{self.MAX_RETRIES + 1})."
                )
                continue

            step_usage = step_usage + self._build_step_usage(response)
            reasoning = self._extract_reasoning(response)
            logger.info(f"Assistant response: {assistant_text[:200]}")

            action = self._parse_action(assistant_text, actions)
            if action is not None:
                return assistant_text, reasoning, action, step_usage, attempt

            logger.warning(
                f"Could not parse action from response "
                f"(attempt {attempt + 1}/{self.MAX_RETRIES + 1})."
            )

        raise RuntimeError(
            f"Failed to get a valid action after {self.MAX_RETRIES + 1} attempts."
        )

    def _trim_oldest_turn(self, conversation: list[dict[str, Any]]) -> bool:
        for i in range(1, len(conversation)):
            if conversation[i]["role"] == "user":
                end = i + 1
                if (
                    end < len(conversation)
                    and conversation[end]["role"] == "assistant"
                ):
                    end += 1
                if len(conversation) - (end - i) < 2:
                    return False
                removed = conversation[i:end]
                del conversation[i:end]
                logger.info(
                    f"Trimmed oldest turn: {[m.get('role', '?') for m in removed]}"
                )
                return True
        return False

    def _build_reasoning_metadata(
        self,
        assistant_text: str,
        reasoning: str | None,
        step_usage: StepUsage,
    ) -> dict[str, Any]:
        usage_obj = ResponseUsage(
            input_tokens=step_usage.prompt_tokens,
            input_tokens_details=InputTokensDetails(
                cached_tokens=step_usage.cached_tokens,
            ),
            output_tokens=step_usage.completion_tokens,
            output_tokens_details=OutputTokensDetails(
                reasoning_tokens=step_usage.reasoning_tokens,
            ),
            total_tokens=step_usage.total_tokens,
        )
        metadata = ActionMetadata(
            output=assistant_text,
            reasoning=reasoning,
            usage=usage_obj,
            cost=CostDetails(
                input_cost=step_usage.cost_details.get("input_cost", 0.0),
                output_cost=step_usage.cost_details.get("output_cost", 0.0),
                total_cost=step_usage.cost,
            ),
        )
        return metadata.model_dump()

    def step(self, context: SessionContext) -> GameStep:
        self._ensure_run_context(context)

        previous_guid = context.datastore.get(self.GUID_KEY)
        if previous_guid and previous_guid != context.game.guid:
            self._set_conversation(context, [])
        if context.game.guid:
            context.datastore[self.GUID_KEY] = context.game.guid

        conversation = self._conversation(context)
        if not conversation:
            conversation.append({"role": "system", "content": self._build_system_prompt()})

        actions = self._get_actions(context)
        user_message = {
            "role": "user",
            "content": self.build_frame_content(context, actions),
        }
        working_messages = list(conversation) + [user_message]
        self._trim_to_fit_context(working_messages)

        start = time.monotonic()
        try:
            assistant_text, reasoning, action, step_usage, retries = self._request_with_retries(
                context, working_messages, actions
            )
            duration = round(time.monotonic() - start, 3)
            working_messages.append({"role": "assistant", "content": assistant_text})
            self._trim_to_fit_context(working_messages)
            self._set_conversation(context, working_messages)
            self.token_counter += step_usage.total_tokens

            self._save_step(
                StepRecord(
                    step=self.step_counter + 1,
                    timestamp=datetime.now(timezone.utc),
                    duration_seconds=duration,
                    model=self.MODEL,
                    messages_sent=list(working_messages),
                    assistant_response=assistant_text,
                    reasoning=reasoning,
                    parsed_action=self._format_parsed_action(action),
                    usage=step_usage,
                    retries=retries,
                )
            )

            return GameStep(
                action=action,
                reasoning=self._build_reasoning_metadata(assistant_text, reasoning, step_usage),
            )
        except Exception as exc:
            duration = round(time.monotonic() - start, 3)
            logger.warning(
                "conversation_rolling_window failed to choose an action: %s: %s. Returning RESET.",
                type(exc).__name__,
                exc,
            )
            self._save_step(
                StepRecord(
                    step=self.step_counter + 1,
                    timestamp=datetime.now(timezone.utc),
                    duration_seconds=duration,
                    model=self.MODEL,
                    messages_sent=list(working_messages),
                    assistant_response=None,
                    reasoning=str(exc),
                    parsed_action="RESET",
                    retries=self.MAX_RETRIES,
                )
            )
            return GameStep(
                action={"action": "RESET"},
                reasoning={
                    "system": "conversation_rolling_window_failure",
                    "error": type(exc).__name__,
                },
            )
