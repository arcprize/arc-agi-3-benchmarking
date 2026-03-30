"""Pydantic models for action metadata passed through the reasoning field."""

from __future__ import annotations

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


class ActionMetadata(BaseModel):
    """Metadata attached to every action via the reasoning field.

    Attributes:
        output: The reply text from the AI model.
        reasoning: Extra reasoning thoughts produced outside the main output
                   (e.g. chain-of-thought, extended thinking).
        usage: Token usage for this action, following OpenAI's ResponseUsage
               schema.
        cost: Computed dollar costs broken down by input and output.
    """

    output: str | None = None
    reasoning: str | None = None
    usage: ResponseUsage = Field(default_factory=ResponseUsage)
    cost: CostDetails = Field(default_factory=CostDetails)


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
