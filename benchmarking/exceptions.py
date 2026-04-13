class EmptyResponseError(Exception):
    """Raised when the API returns HTTP 200 but with null/empty choices."""

    def __init__(self, message: str, response: object | None = None) -> None:
        super().__init__(message)
        self.response = response
