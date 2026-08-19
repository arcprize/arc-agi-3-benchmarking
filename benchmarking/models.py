"""Pydantic models for action metadata passed through the reasoning field."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class InputTokensDetails(BaseModel):
    """A detailed breakdown of the input tokens."""

    cached_tokens: int = 0


class OutputTokensDetails(BaseModel):
    """A detailed breakdown of the output tokens."""

    reasoning_tokens: int = 0


class ResponseUsage(BaseModel):
    """Token usage details mirroring OpenAI's ResponseUsage schema.

    Represents token usage including input tokens, output tokens,
    a breakdown of each, and the total tokens used.
    """

    input_tokens: int = 0
    input_tokens_details: InputTokensDetails = Field(
        default_factory=InputTokensDetails,
    )
    output_tokens: int = 0
    output_tokens_details: OutputTokensDetails = Field(
        default_factory=OutputTokensDetails,
    )
    total_tokens: int = 0


class CostDetails(BaseModel):
    """Computed dollar costs for a single action."""

    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0


class ActionStateMetadata(BaseModel):
    """Action-level state telemetry for runtimes that expose it."""

    input_items_sent: int
    compaction_occurred: bool
    compaction_items_returned: int
    history_items_before_prune: int | None = None
    history_items_after_prune: int | None = None


class ActionMetadata(BaseModel):
    """Metadata attached to every action via the reasoning field.

    Attributes:
        output: The reply text from the AI model.
        reasoning: Extra reasoning thoughts produced outside the main output
                   (e.g. chain-of-thought, extended thinking).
        usage: Token usage for this action, following OpenAI's ResponseUsage
               schema.
        cost: Computed dollar costs broken down by input and output.
        state: Optional action-level state telemetry for encrypted replay.
    """

    output: str | None = None
    reasoning: str | None = None
    usage: ResponseUsage = Field(default_factory=ResponseUsage)
    cost: CostDetails = Field(default_factory=CostDetails)
    state: ActionStateMetadata | None = None

    def to_reasoning_dict(self) -> dict[str, Any]:
        """Serialize for ARC without adding null state to other runtimes."""
        payload = self.model_dump()
        if self.state is None:
            payload.pop("state", None)
        else:
            payload["state"] = self.state.model_dump(exclude_none=True)
        return payload


def calculate_cost(
    token_count: int,
    price_per_million: float,
) -> float:
    """Calculate the dollar cost for a given number of tokens.

    Args:
        token_count: Number of tokens consumed.
        price_per_million: Price in dollars per 1,000,000 tokens
                           (as listed by the API provider).

    Returns:
        Cost in dollars.
    """
    return (token_count / 1_000_000) * price_per_million
