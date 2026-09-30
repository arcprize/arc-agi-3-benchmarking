"""Provider-neutral, model-generated conversation compaction.

The summary-and-bridge shape is inspired by Stirrup's MIT-licensed context
summarization design, adapted to the runtime state contract in this repository.
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, Field

from .exceptions import (
    CompactionContextOverflowError,
    CompactionFailureError,
    ContextOverflowError,
    EmptyResponseError,
    TransientProviderError,
)
from .runtime_models import Message, NormalizedUsage
from .runtime_state import (
    CompactionUnwindResult,
    ModelTurnRequest,
    PendingCompactionInputs,
    RuntimeState,
    StatefulRuntimeAdapter,
    SummaryCompactionRuntimeAdapter,
    sanitize_settings,
)

HARNESS_SUMMARY_COMPACTION = "harness_summary"

SUMMARY_SYSTEM_PROMPT = (
    "You create concise conversation summaries for reliable continuation. "
    "Return only the requested summary."
)

SUMMARY_REQUEST_PROMPT = """Summarize the conversation so you can continue making progress on the task in a future context. The conversation history will be replaced by this summary and will no longer be available.
Use your judgment about what matters for this task."""

SUMMARY_BRIDGE_TEMPLATE = """Earlier conversation history was compacted. Use the following summary as prior context. If it conflicts with newer input, prefer the newer input.

{summary}"""


class SummaryCompactionPolicy(BaseModel):
    strategy: Literal["harness_summary"]
    trigger_tokens: int = Field(strict=True, gt=0)
    summary_max_output_tokens: int = Field(default=8_192, strict=True, gt=0)
    summary_input_headroom_tokens: int = Field(default=8_192, strict=True, ge=0)


class SummaryCompactionResult(BaseModel):
    state: RuntimeState
    prompt: dict[str, Any]
    summary: str
    usage: NormalizedUsage
    attempts: int
    trigger_tokens: int
    history_items_to_compact: int
    overflow_recoveries: int = 0
    excluded_turns: int = 0
    excluded_history_items: int = 0


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


def build_summary_bridge(summary: str) -> str:
    return SUMMARY_BRIDGE_TEMPLATE.format(summary=summary)


def estimate_runtime_state_tokens(
    state: RuntimeState, *, estimated_chars_per_token: float
) -> float:
    serialized_payload = json.dumps(
        state.payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return len(serialized_payload) / estimated_chars_per_token


def build_summary_prompt_record(state: RuntimeState) -> dict[str, Any]:
    """Return the complete readable compaction prompt sent to the adapter.

    Provider-native conversation state is structured rather than one text string,
    so preserve its request shape while removing opaque reasoning continuity data.
    """

    return {
        "system_prompt": SUMMARY_SYSTEM_PROMPT,
        "context": sanitize_settings(state.payload),
        "user_prompt": SUMMARY_REQUEST_PROMPT,
    }


class SummaryCompactor:
    TRANSIENT_RETRY_BASE_SECONDS = 0.25

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
        if not isinstance(adapter, SummaryCompactionRuntimeAdapter):
            raise ValueError(
                "Selected adapter does not support harness summary compaction."
            )
        pending_inputs: list[Message] = []
        if isinstance(adapter, PendingCompactionInputs):
            state, pending_inputs = adapter.split_pending_inputs(state)
        history_items_to_compact = runtime_state_item_count(state)
        summary_request_config = request_config_with_output_limit(
            request_config,
            self.policy.summary_max_output_tokens,
        )
        accumulated_usage = NormalizedUsage()
        candidate_state = state
        excluded_turns: list[CompactionUnwindResult] = []
        excluded_history_items = 0
        overflow_recoveries = 0
        response_failures = 0
        attempts = 0
        while response_failures <= max_retries:
            attempts += 1
            prompt = build_summary_prompt_record(candidate_state)
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
                    raise CompactionContextOverflowError(
                        "Harness summary compaction reached the protected context "
                        "boundary and still exceeds provider capacity.",
                        usage=accumulated_usage,
                    ) from exc
                candidate_state = unwind.state
                excluded_turns.insert(0, unwind)
                excluded_history_items += unwind.removed_items
                overflow_recoveries += 1
                continue
            except EmptyResponseError as exc:
                if isinstance(exc.usage, NormalizedUsage):
                    accumulated_usage = accumulated_usage + exc.usage
                response_failures += 1
                continue
            except TransientProviderError:
                response_failures += 1
                if response_failures <= max_retries:
                    delay = self.TRANSIENT_RETRY_BASE_SECONDS * (
                        2 ** (response_failures - 1)
                    )
                    time.sleep(delay)
                continue
            except Exception as exc:
                if accumulated_usage != NormalizedUsage():
                    raise CompactionFailureError(
                        "Harness summary compaction failed after billable attempts.",
                        usage=accumulated_usage,
                    ) from exc
                raise

            accumulated_usage = accumulated_usage + result.response.usage
            response_status = result.response.response_status
            if response_status is not None and response_status != "completed":
                response_failures += 1
                continue
            summary = result.response.output_text.strip()
            if not summary:
                response_failures += 1
                continue

            try:
                bridge = build_summary_bridge(summary)
                next_state = adapter.rebuild_after_compaction(
                    Message(role="user", content=bridge),
                    excluded_turns,
                )
                if pending_inputs:
                    next_state = adapter.buffer_inputs(next_state, pending_inputs)
                estimated_state_tokens = estimate_runtime_state_tokens(
                    next_state,
                    estimated_chars_per_token=estimated_chars_per_token,
                )
            except Exception as exc:
                raise CompactionFailureError(
                    "Harness summary compaction could not rebuild continuation "
                    "state.",
                    usage=accumulated_usage,
                ) from exc
            if estimated_state_tokens >= max_context_length:
                raise CompactionContextOverflowError(
                    "Harness summary compaction continuation state is estimated to "
                    "exceed provider context capacity.",
                    usage=accumulated_usage,
                )
            return SummaryCompactionResult(
                state=next_state,
                prompt=prompt,
                summary=summary,
                usage=accumulated_usage,
                attempts=attempts,
                trigger_tokens=trigger_tokens,
                history_items_to_compact=history_items_to_compact,
                overflow_recoveries=overflow_recoveries,
                excluded_turns=len(excluded_turns),
                excluded_history_items=excluded_history_items,
            )

        raise CompactionFailureError(
            "Harness summary compaction failed to produce a completed, non-empty "
            f"summary after {max_retries + 1} response failures.",
            usage=accumulated_usage,
        )
