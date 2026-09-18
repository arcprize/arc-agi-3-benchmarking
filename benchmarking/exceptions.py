class EmptyResponseError(Exception):
    """Raised when the API returns HTTP 200 but with null/empty choices."""

    def __init__(
        self,
        message: str,
        response: object | None = None,
        usage: object | None = None,
    ) -> None:
        super().__init__(message)
        self.response = response
        self.usage = usage


class InvalidProviderResponseError(EmptyResponseError):
    """A failed or incomplete provider turn with any observed billable usage."""


class ContextOverflowError(Exception):
    """Raised when a provider rejects a request for exceeding context capacity."""


class TransientProviderError(Exception):
    """Raised when a provider failure is safe to retry without changing state."""


class CompactionFailureError(RuntimeError):
    """Raised when compaction fails after one or more billable attempts."""

    def __init__(self, message: str, usage: object | None = None) -> None:
        super().__init__(message)
        self.usage = usage


class CompactionContextOverflowError(ContextOverflowError):
    """Context overflow raised by compaction with accumulated billable usage."""

    def __init__(self, message: str, usage: object | None = None) -> None:
        super().__init__(message)
        self.usage = usage
