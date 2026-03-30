"""AgentOps integration module for tracing agent execution."""

import functools
import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from .agent import Agent

# Module-level variables
logger = logging.getLogger()
is_initialized = False
agentops_client: Optional[Any] = None


class NoOpAgentOps:
    """No-op implementation when AgentOps is not available."""

    def init(self, *args: Any, **kwargs: Any) -> None:
        """No-op initialization."""
        pass

    class NoOpTrace:
        """No-op trace context manager."""

        def __enter__(self) -> "NoOpAgentOps.NoOpTrace":
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            pass

        def set_status(self, *args: Any, **kwargs: Any) -> None:
            """No-op status setting."""
            pass

    def start_trace(self, *args: Any, **kwargs: Any) -> NoOpTrace:
        """Create a no-op trace context."""
        return self.NoOpTrace()


# Try to import AgentOps library
try:
    import agentops as agentops_module

    agentops_client = agentops_module
except ImportError:
    agentops_client = NoOpAgentOps()


def initialize(
    api_key: Optional[str] = None, log_level: Optional[int] = logging.INFO
) -> None:
    """Initialize the AgentOps SDK with an optional API key.

    Args:
        api_key: Optional AgentOps API key
        log_level: Optional log level for AgentOps
    """
    global is_initialized

    # Check if library is available
    if isinstance(agentops_client, NoOpAgentOps):
        return

    # Validate API key
    if not api_key or not api_key.strip() or api_key == "your_agentops_api_key_here":
        return

    if agentops_client is None:
        return

    try:
        agentops_client.init(
            api_key=api_key,
            auto_start_session=False,
            log_level=log_level,
        )
        is_initialized = True
    except Exception:
        # AgentOps may handle status automatically
        pass


def is_available() -> bool:
    """Check if AgentOps is available and initialized."""
    return not isinstance(agentops_client, NoOpAgentOps) and is_initialized


def _set_trace_status(trace: Any, agent_instance: "Agent") -> None:
    """Set trace status based on agent execution outcome."""
    if not hasattr(trace, "set_status"):
        return

    try:
        if agent_instance.action_counter >= agent_instance.MAX_ACTIONS:
            trace.set_status("Indeterminate")
        else:
            trace.set_status("Success")
    except AttributeError:
        # AgentOps may handle status automatically
        pass


def _handle_trace_error(trace: Any, agent_instance: "Agent", error: Exception) -> None:
    """Handle trace error by setting error status."""
    if hasattr(trace, "set_status"):
        try:
            trace.set_status(f"Error: {error}")
        except AttributeError:
            pass


def trace_agent_session(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that wraps an agent's main execution loop to trace it."""

    @functools.wraps(func)
    def wrapper(agent_instance: "Agent", *args: Any, **kwargs: Any) -> Any:
        if not is_available():
            return func(agent_instance, *args, **kwargs)

        tags = agent_instance.tags or []

        if agentops_client is None:
            return func(agent_instance, *args, **kwargs)

        trace = None
        try:
            with agentops_client.start_trace(
                trace_name=agent_instance.name, tags=tags
            ) as trace:
                agent_instance.trace = trace
                result = func(agent_instance, *args, **kwargs)
                _set_trace_status(trace, agent_instance)
                return result
        except Exception as e:
            if trace is not None:
                _handle_trace_error(trace, agent_instance, e)
            logger.error(
                f"Agent {agent_instance.name} failed with exception: {e}", exc_info=True
            )
            raise

    return wrapper
