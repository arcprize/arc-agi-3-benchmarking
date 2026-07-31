import json
import logging
import math
import os
import re
import textwrap
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from arcengine import FrameData, GameAction, GameState

from .action_metadata import fit_action_metadata_payload
from .base import Agent, ExitReason
from .exceptions import EmptyResponseError
from .model_config import (
    ENCRYPTED_REPLAY_RUNTIME_STATE,
    SERVER_RUNTIME_STATE,
    get_model_config,
)
from .models import ActionStateMetadata
from .recording import RunRecord, StepRecord, StepUsage
from .runtime_adapters import build_model_runtime_adapter
from .runtime_clients import build_model_runtime_client
from .runtime_models import (
    Message,
    ModelRequest,
    ModelResponse,
    NormalizedUsage,
    action_metadata_from_model_response,
)

logger = logging.getLogger()


class BenchmarkingAgent(Agent):
    """An agent that maintains a growing conversation with a model runtime.

    Each turn appends a user message (frame data as text) and an assistant
    message (reasoning + chosen action). On context overflow, the oldest
    turns are trimmed from the front of the conversation.
    """

    MODEL_CONFIG_ID: str = ""
    MAX_ACTIONS: int = 10  # Fallback only when baseline_actions are unavailable.
    MAX_ACTIONS_BASELINE_MULTIPLIER: float = 2.0
    MAX_RETRIES: int = 3
    MAX_CONTEXT_LENGTH: int = 100000
    MAX_ANIMATION_FRAMES: int = 7
    analysis_mode: bool = False
    include_carry_forward_instruction: bool = True
    _encrypted_replay: bool = False
    # Empirically, rendered ARC grid payloads are close to 1 char per token.
    # Using 1.0 is intentionally conservative relative to observed runs.
    ESTIMATED_CHARS_PER_TOKEN: float = 1.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not self.config:
            raise ValueError(
                "No model config specified. Pass --config=<config_id> when running main.py. "
                "Use --list-configs to see available options."
            )
        self.MODEL_CONFIG_ID = self.config
        self.conversation: list[dict[str, Any]] = []
        self.token_counter: int = 0

        agent_cfg, runtime_cfg, client_cfg, request_cfg, pricing_cfg = (
            self._load_model_config()
        )
        self._pricing: dict[str, float] = pricing_cfg

        # Server-managed conversation state (OpenAI Responses previous_response_id
        # + compaction). When enabled, we send only the new message(s) each turn
        # and let OpenAI hold the transcript; `self.conversation` becomes a
        # recording-only mirror and client-side trimming is disabled.
        self._server_state: bool = (
            runtime_cfg.get("state") == SERVER_RUNTIME_STATE
        )
        self._encrypted_replay: bool = (
            runtime_cfg.get("state") == ENCRYPTED_REPLAY_RUNTIME_STATE
        )
        self._previous_response_id: str | None = None
        self._pending_user_messages: list[dict[str, Any]] = []
        self._encrypted_input_items: list[dict[str, Any]] = []

        # Agent-level overrides
        self.MAX_ACTIONS_BASELINE_MULTIPLIER = agent_cfg.get(
            "MAX_ACTIONS_BASELINE_MULTIPLIER", self.MAX_ACTIONS_BASELINE_MULTIPLIER
        )
        self.MAX_CONTEXT_LENGTH = agent_cfg.get(
            "MAX_CONTEXT_LENGTH", self.MAX_CONTEXT_LENGTH
        )
        self.MAX_RUNTIME_SECONDS = agent_cfg.get(
            "MAX_RUNTIME_SECONDS", self.MAX_RUNTIME_SECONDS
        )
        self.MAX_ANIMATION_FRAMES = agent_cfg.get(
            "MAX_ANIMATION_FRAMES", self.MAX_ANIMATION_FRAMES
        )
        self.MAX_RETRIES = agent_cfg.get("MAX_RETRIES", self.MAX_RETRIES)
        self.analysis_mode = agent_cfg.get("analysis_mode", self.analysis_mode)
        self.include_carry_forward_instruction = agent_cfg.get(
            "include_carry_forward_instruction",
            self.include_carry_forward_instruction,
        )

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
        self._pending_action_reasoning: dict[str, Any] = {}

        # Adapter-specific request kwargs from the selected model config.
        self.MODEL: str = request_cfg["model"]
        self._request_kwargs: dict[str, Any] = request_cfg

        self._client = build_model_runtime_client(
            runtime_config=runtime_cfg,
            client_config=client_cfg,
            config_id=self.MODEL_CONFIG_ID,
        )
        self._adapter = build_model_runtime_adapter(
            client=self._client,
            runtime_config=runtime_cfg,
            config_id=self.MODEL_CONFIG_ID,
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
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, float],
    ]:
        """Load config from model_configs.yaml matching MODEL_CONFIG_ID.

        Returns five dicts:
        (agent_cfg, runtime_cfg, client_cfg, request_cfg, pricing_cfg).
        - agent_cfg:    agent-level settings
                        (MAX_ACTIONS_BASELINE_MULTIPLIER, MAX_CONTEXT_LENGTH, …)
        - runtime_cfg:  execution profile (sdk, api)
        - client_cfg:   SDK client constructor args (base_url, api_key_env)
        - request_cfg:  kwargs passed to the selected runtime adapter
                        (model, max_completion_tokens, reasoning_effort, ...)
        - pricing_cfg:  per-million-token pricing (input, output)

        Raises ``ValueError`` if the YAML file is missing or no matching entry.
        """
        raw = get_model_config(self.MODEL_CONFIG_ID)

        agent_cfg: dict[str, Any] = dict(raw.get("agent", {}))
        runtime_cfg: dict[str, Any] = dict(raw.get("runtime", {}))
        client_cfg: dict[str, Any] = dict(raw.get("client", {}))
        request_cfg: dict[str, Any] = dict(raw.get("request", {}))
        pricing_cfg: dict[str, float] = dict(raw.get("pricing", {}))

        return agent_cfg, runtime_cfg, client_cfg, request_cfg, pricing_cfg

    @property
    def name(self) -> str:
        sanitized = self.MODEL_CONFIG_ID.replace("/", "-").replace(":", "-")
        return f"{super().name}.{sanitized}.anim{self.MAX_ANIMATION_FRAMES}"

    # ── Prompts ──────────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        if self.analysis_mode:
            return textwrap.dedent("""\
                You are playing a game. Your goal is to win. Include any context you want to carry forward in your reply, along with the action you want to take. The final action mentioned in your reply will be executed next turn.

                Prior assistant turns may include a <reasoning_summary> block before the prior action text. Treat those summaries as compact helper context about the earlier decision process, then continue by choosing the next action normally.
            """)
        if not self.include_carry_forward_instruction:
            return textwrap.dedent("""\
                You are playing a game. Your goal is to win. The final action mentioned in your reply will be executed next turn.
            """)
        return textwrap.dedent("""\
            You are playing a game. Your goal is to win. Include any context you want to carry forward in your reply, along with the action you want to take. The final action mentioned in your reply will be executed next turn.
        """)

    def _build_assistant_turn_content(
        self,
        output_text: str,
        reasoning_text: str | None,
    ) -> str:
        if not self.analysis_mode or not reasoning_text:
            return output_text
        return textwrap.dedent(f"""\
            <reasoning_summary>
            {reasoning_text}
            </reasoning_summary>

            {output_text}
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

        if not self.is_reset_a_valid_action():
            actions = [a for a in actions if a != GameAction.RESET]

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

    @staticmethod
    def _parse_structured_action(
        payload: dict[str, Any],
        available_actions: list[GameAction],
    ) -> Optional[GameAction]:
        """Validate one available action and its required arguments."""
        # 1. Envelope: only `actions`, containing exactly one object.
        actions = payload.get("actions")
        valid_envelope = (
            set(payload) == {"actions"}
            and isinstance(actions, list)
            and len(actions) == 1
        )
        if not valid_envelope:
            return None
        entry = actions[0]
        if not isinstance(entry, dict):
            return None

        # 2. Availability: the exact action name must be offered this turn.
        action = next(
            (a for a in available_actions if a.name == entry.get("action_type")),
            None,
        )
        if action is None:
            return None

        # 3. Shape: simple actions have no arguments; complex actions require x/y.
        expected_fields = {"action_type", "x", "y"}
        if not action.is_complex():
            expected_fields = {"action_type"}
        if set(entry) != expected_fields:
            return None
        parsed = GameAction.from_name(action.name)
        if not action.is_complex():
            return parsed

        # 4. Coordinates: genuine integers within the game grid.
        x, y = entry["x"], entry["y"]
        invalid_coordinate = any(
            type(value) is not int or not 0 <= value <= 63 for value in (x, y)
        )
        if invalid_coordinate:
            return None
        parsed.set_data({"x": x, "y": y})
        return parsed

    def _parse_action(
        self, text: str, available_actions: list[GameAction]
    ) -> Optional[GameAction]:
        """Parse a structured action or fall back to legacy text parsing."""
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            payload = None

        # A response with `actions` claims the structured schema. Validate it
        # strictly; invalid structured responses must not fall through to regex.
        if isinstance(payload, dict) and "actions" in payload:
            return self._parse_structured_action(payload, available_actions)

        # Legacy free-text responses execute the last valid action mentioned.
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

    def _build_model_request(self) -> ModelRequest:
        if self._server_state:
            return self._build_server_state_request()
        return ModelRequest(
            messages=[Message.model_validate(message) for message in self.conversation],
            request_config=dict(self._request_kwargs),
            input_items=(
                list(self._encrypted_input_items) if self._encrypted_replay else None
            ),
        )

    @staticmethod
    def _serialize_encrypted_replay_output(
        model_response: ModelResponse,
    ) -> list[dict[str, Any]]:
        """Serialize every Responses output item for stateless replay."""
        raw_response = model_response.raw_response
        if isinstance(raw_response, dict):
            raw_output = raw_response.get("output", ())
        else:
            raw_output = getattr(raw_response, "output", ())

        serialized: list[dict[str, Any]] = []
        for item in raw_output or ():
            if hasattr(item, "model_dump"):
                value = item.model_dump(mode="json")
            elif isinstance(item, dict):
                value = dict(item)
            elif hasattr(item, "__dict__"):
                value = dict(vars(item))
            else:
                raise TypeError(
                    "Encrypted replay response output items must be mappings "
                    "or support model_dump()."
                )
            if not isinstance(value, dict):
                raise TypeError(
                    "Encrypted replay response output items must serialize "
                    "to mappings."
                )
            # openai-python 2.41.1 includes response-only fields when dumping
            # these output items, but the Responses input schema rejects them
            # when the encrypted items are replayed.
            item_type = value.get("type")
            if item_type == "reasoning":
                value.pop("status", None)
            elif item_type == "compaction":
                value.pop("created_by", None)
            serialized.append(value)

        if not serialized:
            raise RuntimeError(
                "Encrypted replay response did not contain replayable output items."
            )
        return serialized

    @staticmethod
    def _prune_encrypted_replay_history(
        input_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Drop items superseded by the latest encrypted compaction item."""
        for index in range(len(input_items) - 1, -1, -1):
            if input_items[index].get("type") == "compaction":
                return input_items[index:]
        return input_items

    @staticmethod
    def _messages_sent_for_request(
        model_request: ModelRequest,
    ) -> list[dict[str, Any]]:
        """Return the effective request items for per-step recording."""
        messages = [message.model_dump() for message in model_request.messages]
        if model_request.input_items is None:
            return messages

        recorded_items = list(model_request.input_items)
        if model_request.messages and model_request.messages[0].role == "system":
            return [messages[0], *recorded_items]
        return recorded_items

    def _build_server_state_request(self) -> ModelRequest:
        """Build a request that sends only the new turn(s) to OpenAI.

        On the first turn (no prior response id) the system prompt is sent as a
        leading system message; thereafter only the buffered user message(s) are
        sent, with ``previous_response_id`` carrying the server-side state.
        """
        request_config = dict(self._request_kwargs)
        messages: list[dict[str, Any]] = []
        if self._previous_response_id is None:
            messages.append(
                {"role": "system", "content": self._build_system_prompt()}
            )
        else:
            request_config["previous_response_id"] = self._previous_response_id
        messages.extend(self._pending_user_messages)
        return ModelRequest(
            messages=[Message.model_validate(message) for message in messages],
            request_config=request_config,
        )

    def _record_forced_action_observation(
        self,
        frames: list[FrameData],
        latest_frame: FrameData,
        forced_action: GameAction,
    ) -> None:
        self._sync_level_progress(latest_frame)
        self._level_action_counter += 1

        if forced_action == GameAction.RESET and latest_frame.state is GameState.GAME_OVER:
            frame_message = {
                "role": "user",
                "content": self.build_frame_content(latest_frame),
            }
            self.conversation.append(frame_message)
            # This observation reaches the model only on the next real API call,
            # so buffer it for server-managed state too.
            if self._server_state:
                self._pending_user_messages.append(frame_message)
            elif self._encrypted_replay:
                self._encrypted_input_items.append(frame_message)

        self._save_step(
            StepRecord(
                step=self.step_counter + 1,
                timestamp=datetime.now(timezone.utc),
                duration_seconds=0.0,
                model=self.MODEL,
                messages_sent=list(self.conversation),
                parsed_action=self._format_parsed_action(forced_action),
            )
        )

    # ── Action submission ──────────────────────────────────────────────

    def do_action_request(self, action: GameAction) -> FrameData:
        data = action.action_data.model_dump()
        reasoning = self._pending_action_reasoning or getattr(action, "reasoning", {}) or {}
        self._pending_action_reasoning = {}
        raw = self.arc_env.step(action, data=data, reasoning=reasoning)
        self._previous_action = action
        frame = self._convert_raw_frame_data(raw)
        if reasoning:
            frame.action_input.reasoning = reasoning
        return frame

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
            self.exit_reason = ExitReason.GAME_WIN
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
                    self.exit_reason = ExitReason.ACTION_BUDGET
                    return True
        return False

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        forced_action = self._forced_action_for_frame(latest_frame)
        if forced_action is not None:
            self._record_forced_action_observation(
                frames,
                latest_frame,
                forced_action,
            )
            return forced_action

        self._sync_level_progress(latest_frame)
        self._level_action_counter += 1

        # A forced reset can buffer a user observation before the first model
        # call, so ensure the system prompt is the leading conversation item
        # rather than relying on an otherwise-empty transcript.
        if not self.conversation or self.conversation[0].get("role") != "system":
            self.conversation.insert(
                0,
                {"role": "system", "content": self._build_system_prompt()}
            )

        # Normal turn: append frame, call the model, parse action
        frame_message = {
            "role": "user",
            "content": self.build_frame_content(latest_frame),
        }
        self.conversation.append(frame_message)
        if self._server_state:
            self._pending_user_messages.append(frame_message)
        elif self._encrypted_replay:
            self._encrypted_input_items.append(frame_message)

        actions = self._get_actions(latest_frame)
        start = time.monotonic()
        model_response, action, retries, messages_sent = self._request_with_retries(
            actions
        )
        duration = round(time.monotonic() - start, 3)
        step_usage = StepUsage.from_normalized_usage(model_response.usage)
        action_state: ActionStateMetadata | None = None

        # Commit state only after a response yields a valid action. Response-ID
        # mode advances the pointer; encrypted replay retains the raw output.
        if self._server_state:
            self._previous_response_id = model_response.response_id
            self._pending_user_messages = []
        elif self._encrypted_replay:
            input_items_sent = len(self._encrypted_input_items)
            replay_output = self._serialize_encrypted_replay_output(model_response)
            compaction_items_returned = sum(
                item.get("type") == "compaction" for item in replay_output
            )
            self._encrypted_input_items.extend(replay_output)
            history_items_before_prune = len(self._encrypted_input_items)
            self._encrypted_input_items = self._prune_encrypted_replay_history(
                self._encrypted_input_items
            )
            compaction_occurred = compaction_items_returned > 0
            action_state = ActionStateMetadata(
                input_items_sent=input_items_sent,
                compaction_occurred=compaction_occurred,
                compaction_items_returned=compaction_items_returned,
                history_items_before_prune=(
                    history_items_before_prune if compaction_occurred else None
                ),
                history_items_after_prune=(
                    len(self._encrypted_input_items) if compaction_occurred else None
                ),
            )

        self.conversation.append(
            {
                "role": "assistant",
                "content": self._build_assistant_turn_content(
                    model_response.output_text,
                    model_response.reasoning_text,
                ),
            }
        )

        logger.info(f"Parsed action: {self._format_parsed_action(action)}")
        self._save_step(
            StepRecord(
                step=self.step_counter + 1,
                timestamp=datetime.now(timezone.utc),
                duration_seconds=duration,
                model=self.MODEL,
                messages_sent=messages_sent,
                assistant_response=model_response.output_text,
                reasoning=model_response.reasoning_text,
                parsed_action=self._format_parsed_action(action),
                usage=step_usage,
                retries=retries,
            )
        )

        # Build ActionMetadata and pass as dict through the reasoning field
        metadata = action_metadata_from_model_response(
            model_response=model_response,
            pricing=self._pricing,
        )
        metadata.state = action_state
        arc_reasoning = metadata.to_reasoning_dict()
        if self._encrypted_replay:
            # OpenAI Responses exposes a human-readable summary rather than raw
            # chain-of-thought. Label it precisely in ARC's reasoning payload.
            arc_reasoning["reasoning_summary"] = arc_reasoning.pop(
                "reasoning",
                None,
            )
        self._pending_action_reasoning = fit_action_metadata_payload(
            arc_reasoning
        )
        total_cost = metadata.cost.total_cost
        input_cost = metadata.cost.input_cost
        output_cost = metadata.cost.output_cost
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
    ) -> tuple[ModelResponse, GameAction, int, list[dict[str, Any]]]:
        """Call the API with retries.

        Returns (model_response, action, retries, messages_sent) where
        messages_sent is the exact request transcript used by the successful
        attempt before the current assistant reply is appended locally.
        """
        accumulated_usage = NormalizedUsage()
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                # OpenAI-managed state and encrypted replay each have their own
                # compaction path, so neither uses the manual transcript trimmer.
                if not (self._server_state or self._encrypted_replay):
                    self._trim_to_fit_context()
                model_request = self._build_model_request()
                model_response = self._call_api(model_request)
            except EmptyResponseError as e:
                if e.response is not None:
                    self._save_diagnostic(e.response)
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

            self.track_tokens(model_response.usage.total_tokens)
            accumulated_usage = accumulated_usage + model_response.usage
            model_response = model_response.model_copy(
                update={"usage": accumulated_usage}
            )
            logger.info(f"Assistant response: {model_response.output_text[:200]}")

            action = self._parse_action(model_response.output_text, actions)
            if action is not None:
                return (
                    model_response,
                    action,
                    attempt,
                    self._messages_sent_for_request(model_request),
                )

            logger.warning(
                f"Could not parse action from response "
                f"(attempt {attempt + 1}/{self.MAX_RETRIES + 1})."
            )

        raise RuntimeError(
            f"Failed to get a valid action after {self.MAX_RETRIES + 1} attempts."
        )

    def _call_api(self, model_request: ModelRequest) -> ModelResponse:
        return self._adapter.invoke(model_request)

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
        if hasattr(self, "recorder"):
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
            elif self._timed_out:
                self.run_record.outcome = "TIMEOUT"
            elif self.action_counter >= self.MAX_ACTIONS:
                self.run_record.outcome = "MAX_ACTIONS"
            self._write_run_meta()

            if hasattr(self, "recorder"):
                self.recorder.record(
                    {
                        "system_prompt": self._build_system_prompt(),
                        "final_conversation_length": len(self.conversation),
                        "total_tokens": self.token_counter,
                    }
                )
        super().cleanup(*args, **kwargs)
