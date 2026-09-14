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
