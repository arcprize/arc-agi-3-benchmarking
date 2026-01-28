"""Custom exceptions for ARC-AGI-3 benchmarking"""


class TokenMismatchError(Exception):
    """Raised when token counts do not add up correctly."""

    pass


class GameClientError(Exception):
    """Raised when there's an error communicating with the ARC-AGI-3 API."""

    pass


class ProviderError(Exception):
    """Raised when there's an error with an LLM provider."""

    pass
