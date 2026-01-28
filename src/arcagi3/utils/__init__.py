"""Utility functions for ARC-AGI-3 benchmarking."""

from arcagi3.utils import errors
from arcagi3.utils.context import SessionContext
from arcagi3.utils.rate_limiter import AsyncRequestRateLimiter
from arcagi3.utils.retry import RetryConfig, retry_on_rate_limit, retry_with_exponential_backoff
from arcagi3.utils.task_utils import (
    find_hints_file,
    generate_execution_map,
    generate_scorecard_tags,
    generate_summary,
    load_hints,
    read_models_config,
    read_provider_rate_limits,
    result_exists,
    save_result,
    save_result_in_timestamped_structure,
)

__all__ = [
    "read_models_config",
    "result_exists",
    "save_result",
    "save_result_in_timestamped_structure",
    "read_provider_rate_limits",
    "generate_execution_map",
    "generate_summary",
    "generate_scorecard_tags",
    "load_hints",
    "find_hints_file",
    "retry_with_exponential_backoff",
    "retry_on_rate_limit",
    "RetryConfig",
    "AsyncRequestRateLimiter",
    "SessionContext",
    "errors",
    "truncate_memory",
]


# Lazy import to avoid circular dependency
def __getattr__(name: str):
    if name == "truncate_memory":
        from arcagi3.utils.truncate import truncate_memory

        return truncate_memory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
