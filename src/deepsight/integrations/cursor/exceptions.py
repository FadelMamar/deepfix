"""
Custom exceptions for Cursor API integration.
"""


class CursorAPIError(Exception):
    """Base exception for Cursor API related errors."""
    
    def __init__(self, message: str, status_code: int = None, response_data: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class CursorAuthError(CursorAPIError):
    """Raised when authentication fails with the Cursor API."""
    pass


class CursorValidationError(CursorAPIError):
    """Raised when input validation fails."""
    pass


class CursorRateLimitError(CursorAPIError):
    """Raised when rate limit is exceeded."""
    pass


class CursorAgentNotFoundError(CursorAPIError):
    """Raised when an agent is not found."""
    pass
