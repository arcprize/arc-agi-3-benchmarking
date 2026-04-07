class EmptyResponseError(Exception):
    """Raised when the API returns HTTP 200 but with null/empty choices."""
