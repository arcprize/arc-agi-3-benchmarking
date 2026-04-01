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
from arcengine import FrameData, GameAction, GameState
from openai import OpenAI as OpenAIClient

from ...agent import Agent
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

logger = logging.getLogger()


class ConversationRollingWindow(Agent):
    """An agent that maintains a growing conversation with an OpenAI model.

    Each turn appends a user message (frame data as text) and an assistant
    message (reasoning + chosen action). On context overflow, the oldest
    turns are trimmed from the front of the conversation.
    """

    MODEL_CONFIG_ID: str = "gpt-5.4-openrouter"
    MAX_ACTIONS: int = 10  # Fallback only when baseline_actions are unavailable.
    MAX_ACTIONS_BASELINE_MULTIPLIER: float = 2.0
    MAX_RETRIES: int = 3
    MAX_CONTEXT_LENGTH: int = 100000
    MAX_ANIMATION_FRAMES: int = 7
    # Empirically, rendered ARC grid payloads are close to 1 char per token.
    # Using 1.0 is intentionally conservative relative to observed runs.
    ESTIMATED_CHARS_PER_TOKEN: float = 1.0

    # Defaults used when the YAML entry is missing client fields
    _DEFAULT_BASE_URL: str = "https://openrouter.ai/api/v1"
    _DEFAULT_API_KEY_ENV: str = "OPENROUTER_API_KEY"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.config:
            self.MODEL_CONFIG_ID = self.config
        self.conversation: list[dict[str, Any]] = []
        self.token_counter: int = 0

        agent_cfg, client_cfg, call_cfg, pricing_cfg = self._load_model_config()
        self._pricing: dict[str, float] = pricing_cfg

        # Agent-level overrides
        self.MAX_ACTIONS_BASELINE_MULTIPLIER = agent_cfg.get(
            "MAX_ACTIONS_BASELINE_MULTIPLIER", self.MAX_ACTIONS_BASELINE_MULTIPLIER
        )
        self.MAX_CONTEXT_LENGTH = agent_cfg.get(
            "MAX_CONTEXT_LENGTH", self.MAX_CONTEXT_LENGTH
        )
        self.MAX_ANIMATION_FRAMES = agent_cfg.get(
            "MAX_ANIMATION_FRAMES", self.MAX_ANIMATION_FRAMES
        )
        self.MAX_RETRIES = agent_cfg.get("MAX_RETRIES", self.MAX_RETRIES)

        # Per-level action budgets from baseline_actions * multiplier.
        # MAX_ACTIONS becomes the derived total budget across all levels.
        baseline_actions = self.arc_env.info.baseline_actions or []
        if baseline_actions:
            self._level_action_budgets = [
                math.ceil(b * self.MAX_ACTIONS_BASELINE_MULTIPLIER)
                for b in baseline_actions
            ]
            self.MAX_ACTIONS = sum(self._level_action_budgets)
            logger.info(
                f"{self.game_id} - Per-level action budgets "
                f"(multiplier={self.MAX_ACTIONS_BASELINE_MULTIPLIER}): "
                f"baselines={baseline_actions}, "
                f"budgets={self._level_action_budgets}, "
                f"total={self.MAX_ACTIONS}"
            )
        else:
            self._level_action_budgets = []
            logger.info(
                f"{self.game_id} - No baseline_actions available, "
                f"using MAX_ACTIONS={self.MAX_ACTIONS}"
            )
        self._level_action_counter: int = 0
        self._last_levels_completed: int = 0
        self._level_just_advanced: bool = False

        # Call kwargs passed directly to chat.completions.create()
        self.MODEL: str = call_cfg["model"]
        self._call_kwargs: dict[str, Any] = call_cfg

        # Client
        self._client = OpenAIClient(
            base_url=client_cfg["base_url"],
            api_key=os.environ.get(client_cfg["api_key_env"], ""),
        )
        # Per-step recording
        self.step_counter: int = 0
        run_id = uuid.uuid4()
        self.run_dir = os.path.join("recordings", f"{self.name}.{run_id}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.run_record = RunRecord(
            run_id=str(run_id),
            game_id=self.game_id,
            agent_name=self.name,
            model=self.MODEL,
            started_at=datetime.now(timezone.utc),
            run_dir=self.run_dir,
        )
        self._write_run_meta()

    def _load_model_config(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, float]]:
        """Load config from model_configs.yaml matching MODEL_CONFIG_ID.

        Returns four dicts: (agent_cfg, client_cfg, call_cfg, pricing_cfg).
        - agent_cfg:    agent-level settings
                        (MAX_ACTIONS_BASELINE_MULTIPLIER, MAX_CONTEXT_LENGTH, …)
        - client_cfg:   OpenAI client constructor args (base_url, api_key_env)
        - call_cfg:     kwargs passed directly to chat.completions.create()
                        (model, max_completion_tokens, reasoning_effort, …)
        - pricing_cfg:  per-million-token pricing (input, output)

        Raises ``ValueError`` if the YAML file is missing or no matching entry.
        """
        cfg_path = Path(__file__).parent / "model_configs.yaml"
        if not cfg_path.exists():
            raise ValueError(
                f"model_configs.yaml not found at {cfg_path}. "
                f"Cannot resolve MODEL_CONFIG_ID={self.MODEL_CONFIG_ID!r}."
            )
        configs = yaml.safe_load(cfg_path.read_text()) or []
        raw: dict[str, Any] | None = None
        for entry in configs:
            if entry.get("name") == self.MODEL_CONFIG_ID:
                raw = entry
                break
        if raw is None:
            available = [e.get("name") for e in configs]
            raise ValueError(
                f"Model config {self.MODEL_CONFIG_ID!r} not found in {cfg_path}. "
                f"Available configs: {available}"
            )

        agent_cfg: dict[str, Any] = dict(raw.get("agent", {}))
        client_cfg: dict[str, Any] = dict(raw.get("client", {}))
        call_cfg: dict[str, Any] = dict(raw.get("call", {}))
        pricing_cfg: dict[str, float] = dict(raw.get("pricing", {}))

        client_cfg.setdefault("base_url", self._DEFAULT_BASE_URL)
        client_cfg.setdefault("api_key_env", self._DEFAULT_API_KEY_ENV)

        return agent_cfg, client_cfg, call_cfg, pricing_cfg

    @property
    def name(self) -> str:
        sanitized = self.MODEL_CONFIG_ID.replace("/", "-").replace(":", "-")
        return f"{super().name}.{sanitized}.anim{self.MAX_ANIMATION_FRAMES}"

    # ── Prompts ──────────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        return textwrap.dedent("""\
            You are playing a game. Your goal is to win. Reply with the exact action you want to take. The final action in your reply will be executed next turn. Your entire reply will be carried to the next turn.
        """)

    def _get_actions(self, latest_frame: FrameData) -> list[GameAction]:
        """Convert frame's available_actions (list[int]) to GameAction objects.

        Always includes RESET so the model can choose to restart the current
        level even when the game engine does not advertise it.
        * Exception: Do not include RESET on the first action
        and when the previous action was RESET.
        """
        actions = [GameAction.from_id(a) for a in latest_frame.available_actions]
        if not any(a.name == "RESET" for a in actions):
            if self.is_reset_a_valid_action():
                actions.insert(0, GameAction.RESET)
        return actions

    def _build_available_actions_text(self, actions: list[GameAction]) -> str:
        lines = []
        for action in actions:
            if action.is_complex():
                lines.append(f"- {action.name} x y  (where x and y are integers 0-63)")
            else:
                lines.append(f"- {action.name}")
        return "\n".join(lines)

    # ── Frame rendering ──────────────────────────────────────────────────

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

    def build_frame_content(self, latest_frame: FrameData) -> str:
        frames = self.interpolate_frames(latest_frame.frame)

        parts = [
            f"State: {latest_frame.state.name}\n"
            f"Levels completed: {latest_frame.levels_completed}",
        ]

        for i, frame in enumerate(frames):
            frame_lines = []

            if self._level_just_advanced and i == len(frames) - 1:
                frame_lines.append("")
                frame_lines.append("New Level:")
                frame_lines.append("")
                self._level_just_advanced = False

            frame_lines.append(f"Frame {i}:")
            frame_lines.extend(f"  {row}" for row in frame)

            parts.append("\n".join(frame_lines))

        actions_text = self._build_available_actions_text(
            self._get_actions(latest_frame)
        )
        parts.append(f"Available actions:\n{actions_text}")

        return "\n\n".join(parts)

    # ── Action parsing ───────────────────────────────────────────────────

    def _parse_action(
        self, text: str, available_actions: list[GameAction]
    ) -> Optional[GameAction]:
        """Parse the last mentioned action from the assistant's response."""
        text_upper = text.upper()
        candidates: list[tuple[int, GameAction]] = []

        for action in available_actions:
            if action.is_complex():
                pattern = rf"{action.name}\s*[:(]?\s*(\d+)\s*[,\s]\s*(\d+)\s*\)?"
                for match in re.finditer(pattern, text_upper):
                    a = GameAction.from_name(action.name)
                    x = int(match.group(1))
                    y = int(match.group(2))
                    if not (0 <= x <= 63 and 0 <= y <= 63):
                        logger.warning(
                            "Ignoring out-of-bounds coordinates for %s: (%s, %s)",
                            action.name,
                            x,
                            y,
                        )
                        continue
                    a.set_data({"x": x, "y": y})
                    candidates.append((match.start(), a))
            else:
                start = 0
                while True:
                    pos = text_upper.find(action.name, start)
                    if pos == -1:
                        break
                    candidates.append((pos, GameAction.from_name(action.name)))
                    start = pos + len(action.name)

        if not candidates:
            return None

        candidates.sort(key=lambda c: c[0])
        return candidates[-1][1]

    # ── Per-step recording ──────────────────────────────────────────────

    @staticmethod
    def _format_parsed_action(action: GameAction) -> str | dict[str, Any]:
        """Format a parsed action for recording. Complex actions include coordinates."""
        if action.is_complex():
            return {"action": action.name, **action.action_data.model_dump()}
        return str(action.name)

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

    def _save_step(self, step: StepRecord) -> None:
        self.step_counter += 1
        self.run_record.total_usage = self.run_record.total_usage + step.usage
        self.run_record.total_steps = self.step_counter
        filename = os.path.join(self.run_dir, f"step_{self.step_counter:03d}.json")
        with open(filename, "w") as f:
            f.write(step.model_dump_json(indent=2))
        self._write_run_meta()
        logger.info(f"Saved step {self.step_counter} to {filename}")

    # ── Action submission ──────────────────────────────────────────────

    def do_action_request(self, action: GameAction) -> FrameData:
        data = action.action_data.model_dump()
        reasoning = getattr(action, "reasoning", {}) or {}
        raw = self.arc_env.step(action, data=data, reasoning=reasoning)
        return self._convert_raw_frame_data(raw)

    # ── Core loop ────────────────────────────────────────────────────────

    def _sync_level_progress(self, latest_frame: FrameData) -> None:
        current_level = latest_frame.levels_completed
        if current_level > self._last_levels_completed:
            logger.info(
                f"{self.game_id} - Level advanced: {self._last_levels_completed} -> {current_level}. "
                f"Resetting level action counter (was {self._level_action_counter})."
            )
            self._level_action_counter = 0
            self._last_levels_completed = current_level
            self._level_just_advanced = True

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        if latest_frame.state is GameState.WIN:
            return True
        # Check per-level action budget
        if self._level_action_budgets:
            self._sync_level_progress(latest_frame)
            level = latest_frame.levels_completed
            if level < len(self._level_action_budgets):
                budget = self._level_action_budgets[level]
                if self._level_action_counter >= budget:
                    logger.info(
                        f"{self.game_id} - Exceeded action budget for level {level}: "
                        f"{self._level_action_counter}/{budget}. Stopping."
                    )
                    return True
        return False

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        self._sync_level_progress(latest_frame)
        self._level_action_counter += 1

        # Reset whenever the environment indicates the game is not currently playable.
        # Show the game-over frame to the model so it sees why it died.
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            if latest_frame.state == GameState.GAME_OVER:
                self.conversation.append(
                    {
                        "role": "user",
                        "content": self.build_frame_content(latest_frame),
                    }
                )
            self._save_step(
                StepRecord(
                    step=self.step_counter + 1,
                    timestamp=datetime.now(timezone.utc),
                    duration_seconds=0.0,
                    model=self.MODEL,
                    messages_sent=list(self.conversation),
                    parsed_action="RESET",
                )
            )
            return GameAction.RESET

        # Ensure the system prompt is present before the first real turn
        if not self.conversation:
            self.conversation.append(
                {"role": "system", "content": self._build_system_prompt()}
            )

        # Normal turn: append frame, call the model, parse action
        self.conversation.append(
            {"role": "user", "content": self.build_frame_content(latest_frame)}
        )

        actions = self._get_actions(latest_frame)
        start = time.monotonic()
        assistant_text, reasoning, action, step_usage, retries = (
            self._request_with_retries(actions)
        )
        duration = round(time.monotonic() - start, 3)

        self.conversation.append({"role": "assistant", "content": assistant_text})

        logger.info(f"Parsed action: {self._format_parsed_action(action)}")
        self._save_step(
            StepRecord(
                step=self.step_counter + 1,
                timestamp=datetime.now(timezone.utc),
                duration_seconds=duration,
                model=self.MODEL,
                messages_sent=list(self.conversation),
                assistant_response=assistant_text,
                reasoning=reasoning,
                parsed_action=self._format_parsed_action(action),
                usage=step_usage,
                retries=retries,
            )
        )

        # Build ActionMetadata and pass as dict through the reasoning field
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
        input_cost = calculate_cost(
            step_usage.prompt_tokens, self._pricing.get("input", 0.0)
        )
        output_cost = calculate_cost(
            step_usage.completion_tokens, self._pricing.get("output", 0.0)
        )
        total_cost = input_cost + output_cost
        metadata = ActionMetadata(
            output=assistant_text,
            reasoning=reasoning,
            usage=usage_obj,
            cost=CostDetails(
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=total_cost,
            ),
        )
        action.reasoning = metadata.model_dump()
        logger.info(
            f"Step cost: ${total_cost:.6f} "
            f"(input: ${input_cost:.6f}, output: ${output_cost:.6f})"
        )

        return action

    # ── Token estimation & proactive trimming ─────────────────────────

    def _estimate_conversation_tokens(self) -> int:
        """Estimate token count using an empirically calibrated chars-per-token ratio."""
        total_chars = sum(len(m.get("content", "")) for m in self.conversation)
        return math.ceil(total_chars / self.ESTIMATED_CHARS_PER_TOKEN)

    def _trim_to_fit_context(self) -> None:
        """Proactively trim oldest turns if estimated tokens exceed MAX_CONTEXT_LENGTH."""
        estimated = self._estimate_conversation_tokens()
        while estimated > self.MAX_CONTEXT_LENGTH:
            if not self._trim_oldest_turn():
                logger.warning(
                    f"Cannot trim further but estimated tokens ({estimated}) "
                    f"still exceed MAX_CONTEXT_LENGTH ({self.MAX_CONTEXT_LENGTH})."
                )
                break
            estimated = self._estimate_conversation_tokens()
            logger.info(
                f"Proactive context trim: ~{estimated} tokens "
                f"(limit {self.MAX_CONTEXT_LENGTH}), "
                f"{len(self.conversation)} messages remaining."
            )

    # ── API calls & retries ────────────────────────────────────────────

    def _request_with_retries(
        self, actions: list[GameAction]
    ) -> tuple[str, str | None, GameAction, StepUsage, int]:
        """Call the API with retries. Returns (assistant_text, reasoning, action, usage, retries)."""
        step_usage = StepUsage()
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                response = self._call_api()
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

            step_usage = step_usage + StepUsage.from_response(response)
            msg = response.choices[0].message
            assistant_text = msg.content or ""
            reasoning = getattr(msg, "reasoning", None) or getattr(
                msg, "reasoning_content", None
            )
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

    def _call_api(self) -> Any:
        self._trim_to_fit_context()

        response = self._client.chat.completions.create(
            messages=self.conversation,
            **self._call_kwargs,
        )

        if not response.choices:
            self._save_diagnostic(response)
            raise EmptyResponseError(
                f"API returned 200 with empty choices. "
                f"Diagnostics saved to {self.run_dir}"
            )

        if response.usage:
            self.track_tokens(response.usage.total_tokens)
        return response

    def _trim_oldest_turn(self) -> bool:
        """Remove the oldest user/assistant pair, preserving the system message."""
        # Find the first user message (skips system prompt and bootstrap assistant)
        for i in range(1, len(self.conversation)):
            if self.conversation[i]["role"] == "user":
                # Remove this user message and its assistant reply if present
                end = i + 1
                if (
                    end < len(self.conversation)
                    and self.conversation[end]["role"] == "assistant"
                ):
                    end += 1
                # Keep at least 2 messages (system + current user turn)
                if len(self.conversation) - (end - i) < 2:
                    return False
                removed = self.conversation[i:end]
                self.conversation = self.conversation[:i] + self.conversation[end:]
                logger.info(
                    f"Trimmed oldest turn: {[m.get('role', '?') for m in removed]}"
                )
                return True
        return False

    # ── Token tracking & cleanup ─────────────────────────────────────────

    def track_tokens(self, tokens: int, message: str = "") -> None:
        self.token_counter += tokens
        if hasattr(self, "recorder") and not self.is_playback:
            self.recorder.record(
                {
                    "tokens": tokens,
                    "total_tokens": self.token_counter,
                    "conversation_length": len(self.conversation),
                    "assistant": message,
                }
            )
        logger.info(
            f"Tokens: {tokens}, total: {self.token_counter}, "
            f"messages: {len(self.conversation)}"
        )

    def cleanup(self, *args: Any, **kwargs: Any) -> None:
        if self._cleanup:
            now = datetime.now(timezone.utc)
            self.run_record.ended_at = now
            self.run_record.duration_seconds = round(
                (now - self.run_record.started_at).total_seconds(), 3
            )
            if self.state is GameState.WIN:
                self.run_record.outcome = "WIN"
            elif self.state is GameState.GAME_OVER:
                self.run_record.outcome = "GAME_OVER"
            elif self.action_counter >= self.MAX_ACTIONS:
                self.run_record.outcome = "MAX_ACTIONS"
            self._write_run_meta()

            if hasattr(self, "recorder") and not self.is_playback:
                self.recorder.record(
                    {
                        "system_prompt": self._build_system_prompt(),
                        "final_conversation_length": len(self.conversation),
                        "total_tokens": self.token_counter,
                    }
                )
        super().cleanup(*args, **kwargs)
