"""Provider-neutral, model-generated conversation compaction.

The summary-and-bridge shape is inspired by Stirrup's MIT-licensed context
summarization design, adapted to the runtime state contract in this repository.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, Field

from .exceptions import EmptyResponseError
from .runtime_models import Message, NormalizedUsage
from .runtime_state import ModelTurnRequest, RuntimeState, StatefulRuntimeAdapter

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


class SummaryCompactionPolicy(BaseModel):
    strategy: Literal["harness_summary"]
    summary_max_output_tokens: int = Field(default=8_192, strict=True, gt=0)


class SummaryCompactionResult(BaseModel):
    state: RuntimeState
    summary: str
    usage: NormalizedUsage
    attempts: int
    trigger_tokens: int
    history_items_before: int
    history_items_after: int
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


class SummaryCompactor:
    def __init__(self, policy: SummaryCompactionPolicy) -> None:
        self.policy = policy

    @staticmethod
    def should_compact(usage: NormalizedUsage, max_context_length: int) -> bool:
        return usage.total_tokens >= max_context_length

    def compact(
        self,
        *,
        adapter: StatefulRuntimeAdapter,
        state: RuntimeState,
        request_config: dict[str, Any],
        trigger_tokens: int,
        max_context_length: int,
        max_retries: int,
    ) -> SummaryCompactionResult:
        history_items_before = runtime_state_item_count(state)
        summary_request_config = request_config_with_output_limit(
            request_config,
            self.policy.summary_max_output_tokens,
        )
        accumulated_usage = NormalizedUsage()
        for attempt in range(max_retries + 1):
            try:
                result = adapter.invoke_turn(
                    ModelTurnRequest(
                        system_prompt=SUMMARY_SYSTEM_PROMPT,
                        new_messages=[
                            Message(role="user", content=SUMMARY_REQUEST_PROMPT)
                        ],
                        request_config=summary_request_config,
                        previous_state=state,
                        max_context_length=max_context_length,
                    )
                )
            except EmptyResponseError:
                continue

            accumulated_usage = accumulated_usage + result.response.usage
            summary = result.response.output_text.strip()
            if not summary:
                continue

            next_state = adapter.buffer_inputs(
                adapter.initial_state(),
                [
                    Message(
                        role="user",
                        content=SUMMARY_BRIDGE_TEMPLATE.format(summary=summary),
                    )
                ],
            )
            return SummaryCompactionResult(
                state=next_state,
                summary=summary,
                usage=accumulated_usage,
                attempts=attempt + 1,
                trigger_tokens=trigger_tokens,
                history_items_before=history_items_before,
                history_items_after=runtime_state_item_count(next_state),
            )

        raise RuntimeError(
            "Harness summary compaction failed to produce non-empty text after "
            f"{max_retries + 1} attempts."
        )
