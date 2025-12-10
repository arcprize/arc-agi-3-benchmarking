"""
Retry utilities with exponential backoff for API calls.
"""
import logging
import time
from functools import wraps
from typing import Any, Callable, Tuple, Type

logger = logging.getLogger(__name__)


def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Decorator to retry a function with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplier for delay after each retry
        max_delay: Maximum delay in seconds between retries
        exceptions: Tuple of exception types to catch and retry
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries. "
                            f"Last error: {type(e).__name__}: {str(e)}"
                        )
                        raise
                    
                    logger.warning(
                        f"Function {func.__name__} failed on attempt {attempt + 1}/{max_retries + 1}. "
                        f"Error: {type(e).__name__}: {str(e)}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    
                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
            
        return wrapper
    return decorator


def retry_on_rate_limit(
    max_retries: int = 5,
    initial_delay: float = 2.0,
):
    """
    Decorator specifically for handling rate limit errors.
    Catches common rate limit exceptions from various providers.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        
    Returns:
        Decorated function
    """
    rate_limit_exceptions = []
    
    # Try to import provider-specific rate limit exceptions
    try:
        from anthropic import RateLimitError as AnthropicRateLimitError
        rate_limit_exceptions.append(AnthropicRateLimitError)
    except ImportError:
        pass
    
    try:
        from openai import RateLimitError as OpenAIRateLimitError
        rate_limit_exceptions.append(OpenAIRateLimitError)
    except ImportError:
        pass
    
    try:
        from google.api_core.exceptions import \
            ResourceExhausted as GoogleRateLimitError
        rate_limit_exceptions.append(GoogleRateLimitError)
    except ImportError:
        pass
    
    # Fallback to catching generic exceptions if no specific ones found
    if not rate_limit_exceptions:
        rate_limit_exceptions = [Exception]
    
    return retry_with_exponential_backoff(
        max_retries=max_retries,
        initial_delay=initial_delay,
        backoff_factor=2.0,
        max_delay=120.0,
        exceptions=tuple(rate_limit_exceptions)
    )


class RetryConfig:
    """Configuration for retry behavior"""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay

