"""Pydantic models for per-step and per-run recording."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from .runtime_models import NormalizedUsage


class StepUsage(BaseModel):
    """Token usage and cost for a single step (or accumulated across a run)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    cost: float = 0.0
    cost_details: dict[str, float] = Field(default_factory=dict)

    def __add__(self, other: StepUsage) -> StepUsage:
        merged_cost_details: dict[str, float] = {}
        for key in set(
            list(self.cost_details.keys()) + list(other.cost_details.keys())
        ):
            merged_cost_details[key] = self.cost_details.get(
                key, 0.0
            ) + other.cost_details.get(key, 0.0)
        return StepUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cost=self.cost + other.cost,
            cost_details=merged_cost_details,
        )

    @classmethod
    def from_response(cls, response: Any) -> StepUsage:
        """Extract usage from an OpenRouter chat-completion response."""
        if not response.usage:
            return cls()
        usage = response.usage
        kwargs: dict[str, Any] = {
            "prompt_tokens": usage.prompt_tokens or 0,
            "completion_tokens": usage.completion_tokens or 0,
            "total_tokens": usage.total_tokens or 0,
        }
        if usage.completion_tokens_details:
            kwargs["reasoning_tokens"] = (
                usage.completion_tokens_details.reasoning_tokens or 0
            )
        if usage.prompt_tokens_details:
            kwargs["cached_tokens"] = usage.prompt_tokens_details.cached_tokens or 0
            kwargs["cache_write_tokens"] = (
                getattr(usage.prompt_tokens_details, "cache_write_tokens", 0) or 0
            )
        extras = getattr(usage, "model_extra", {}) or {}
        if "cost" in extras:
            kwargs["cost"] = extras["cost"]
        if "cost_details" in extras:
            kwargs["cost_details"] = extras["cost_details"]
        return cls(**kwargs)

    @classmethod
    def from_normalized_usage(cls, usage: NormalizedUsage) -> StepUsage:
        return cls(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            cached_tokens=usage.cached_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            cost=usage.cost,
            cost_details=dict(usage.cost_details),
        )


class StepRecord(BaseModel):
    """One recorded step (one choose_action call)."""

    step: int
    timestamp: datetime
    duration_seconds: float = 0.0
    model: str
    messages_sent: list[dict[str, Any]]
    assistant_response: str | None = None
    reasoning: str | None = None
    parsed_action: str | dict[str, Any]
    usage: StepUsage = Field(default_factory=StepUsage)
    retries: int = 0


class RunRecord(BaseModel):
    """Metadata for an entire agent run, written to run_meta.json."""

    run_id: str
    game_id: str
    agent_name: str
    model: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    total_steps: int = 0
    total_usage: StepUsage = Field(default_factory=StepUsage)
    outcome: Optional[str] = None
    run_dir: str
