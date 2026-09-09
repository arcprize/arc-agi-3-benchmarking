"""Provider-neutral, model-generated conversation compaction.

The summary-and-bridge shape is inspired by Stirrup's MIT-licensed context
summarization design, adapted to the runtime state contract in this repository.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, Field

from .exceptions import ContextOverflowError, EmptyResponseError
from .runtime_models import Message, NormalizedUsage
from .runtime_state import (
    AcceptedTurn,
    ModelTurnRequest,
    RuntimeState,
    StatefulRuntimeAdapter,
)

HARNESS_SUMMARY_COMPACTION = "harness_summary"

SUMMARY_SYSTEM_PROMPT = (
    "You create concise conversation summaries for reliable continuation. "
    "Return only the requested summary."
)

SUMMARY_REQUEST_PROMPT = """The conversation is approaching its context limit. Create a concise continuation summary that preserves:

1. The objective.
2. Established facts and decisions.
3. Progress made and results obtained.
4. The current state.
5. Constraints and unsuccessful approaches.
6. Important identifiers and references.
7. Clear next steps.

Return only the summary. Do not continue the task."""

SUMMARY_BRIDGE_TEMPLATE = """Earlier conversation history was compacted. Use the following summary as prior context. If it conflicts with newer input, prefer the newer input.

<conversation_summary>
{summary}
</conversation_summary>"""

RECENT_CONTEXT_TEMPLATE = """

Some recent turns were excluded from summarization so the summary request would fit. They are newer than the summary and are reproduced below in readable form.

<recent_context>
{recent_context}
</recent_context>"""


class SummaryCompactionPolicy(BaseModel):
    strategy: Literal["harness_summary"]
    trigger_tokens: int = Field(strict=True, gt=0)
    summary_max_output_tokens: int = Field(default=8_192, strict=True, gt=0)
    summary_input_headroom_tokens: int = Field(default=8_192, strict=True, ge=0)


class SummaryCompactionResult(BaseModel):
    state: RuntimeState
    summary: str
    usage: NormalizedUsage
    attempts: int
    trigger_tokens: int
    history_items_before: int
    history_items_after: int
    overflow_recoveries: int = 0
    excluded_turns: int = 0
    excluded_history_items: int = 0
    opaque_continuity_preserved: bool = False


def runtime_state_item_count(state: RuntimeState) -> int:
    return sum(
        len(value) for value in state.payload.values() if isinstance(value, list)
    )


def request_config_with_output_limit(
    request_config: dict[str, Any], limit: int
) -> dict[str, Any]:
    updated = deepcopy(request_config)
    generation_config = updated.get("generation_config")
    if isinstance(generation_config, dict):
        generation_config["max_output_tokens"] = limit
        return updated
    for field in ("max_output_tokens", "max_completion_tokens", "max_tokens"):
        if field in updated:
            updated[field] = limit
            return updated
    updated["max_output_tokens"] = limit
    return updated


def render_recent_context(turns: list[AcceptedTurn]) -> str:
    """Render excluded accepted turns without provider-native opaque state."""

    readable_turns: list[dict[str, Any]] = []
    for turn in turns:
        readable: dict[str, Any] = {
            "messages": [message.model_dump() for message in turn.messages]
        }
        if turn.reasoning_summary:
            readable["reasoning_summary"] = turn.reasoning_summary
        readable_turns.append(readable)
    return json.dumps(readable_turns, ensure_ascii=False, indent=2)


def build_summary_bridge(summary: str, excluded_turns: list[AcceptedTurn]) -> str:
    bridge = SUMMARY_BRIDGE_TEMPLATE.format(summary=summary)
    if not excluded_turns:
        return bridge
    return bridge + RECENT_CONTEXT_TEMPLATE.format(
        recent_context=render_recent_context(excluded_turns)
    )


class SummaryCompactor:
    def __init__(self, policy: SummaryCompactionPolicy) -> None:
        self.policy = policy

    def should_compact(self, usage: NormalizedUsage) -> bool:
        return usage.total_tokens >= self.policy.trigger_tokens

    def compact(
        self,
        *,
        adapter: StatefulRuntimeAdapter,
        state: RuntimeState,
        request_config: dict[str, Any],
        trigger_tokens: int,
        max_context_length: int,
        max_retries: int,
        estimated_chars_per_token: float = 1.0,
    ) -> SummaryCompactionResult:
        if estimated_chars_per_token <= 0:
            raise ValueError("estimated_chars_per_token must be greater than zero.")
        history_items_before = runtime_state_item_count(state)
        summary_request_config = request_config_with_output_limit(
            request_config,
            self.policy.summary_max_output_tokens,
        )
        accumulated_usage = NormalizedUsage()
        candidate_state = state
        excluded_turns: list[AcceptedTurn] = []
        excluded_history_items = 0
        overflow_recoveries = 0
        empty_attempts = 0
        attempts = 0
        while empty_attempts <= max_retries:
            attempts += 1
            try:
                result = adapter.invoke_turn(
                    ModelTurnRequest(
                        system_prompt=SUMMARY_SYSTEM_PROMPT,
                        new_messages=[
                            Message(role="user", content=SUMMARY_REQUEST_PROMPT)
                        ],
                        request_config=summary_request_config,
                        previous_state=candidate_state,
                        max_context_length=max_context_length,
                        estimated_chars_per_token=estimated_chars_per_token,
                    )
                )
            except ContextOverflowError as exc:
                unwind = adapter.unwind_latest_accepted_turn(candidate_state)
                if unwind is None:
                    raise ContextOverflowError(
                        "Harness summary compaction reached the protected context "
                        "boundary and still exceeds provider capacity."
                    ) from exc
                candidate_state = unwind.state
                excluded_turns.insert(0, unwind.turn)
                excluded_history_items += unwind.removed_items
                overflow_recoveries += 1
                continue
            except EmptyResponseError:
                empty_attempts += 1
                continue

            accumulated_usage = accumulated_usage + result.response.usage
            summary = result.response.output_text.strip()
            if not summary:
                empty_attempts += 1
                continue

            bridge = build_summary_bridge(summary, excluded_turns)
            estimated_bridge_tokens = len(bridge) / estimated_chars_per_token
            if estimated_bridge_tokens >= max_context_length:
                raise ContextOverflowError(
                    "Harness summary compaction continuation bridge is estimated "
                    "to exceed provider context capacity."
                )
            next_state = adapter.buffer_inputs(
                adapter.initial_state(),
                [
                    Message(
                        role="user",
                        content=bridge,
                    )
                ],
            )
            return SummaryCompactionResult(
                state=next_state,
                summary=summary,
                usage=accumulated_usage,
                attempts=attempts,
                trigger_tokens=trigger_tokens,
                history_items_before=history_items_before,
                history_items_after=runtime_state_item_count(next_state),
                overflow_recoveries=overflow_recoveries,
                excluded_turns=len(excluded_turns),
                excluded_history_items=excluded_history_items,
            )

        raise RuntimeError(
            "Harness summary compaction failed to produce non-empty text after "
            f"{max_retries + 1} empty attempts."
        )
