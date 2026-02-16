"""
Pine AI error types — maps spec section 4.2 error codes.
"""

from typing import Any, Optional


class PineAIError(Exception):
    def __init__(self, code: str, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.details = details


class AuthError(PineAIError):
    def __init__(self, message: str, code: str = "auth_error"):
        super().__init__(code, message)


class SessionError(PineAIError):
    def __init__(self, message: str, code: str = "session_error", details: Optional[dict[str, Any]] = None):
        super().__init__(code, message, details)


class ConnectionError(PineAIError):
    def __init__(self, message: str):
        super().__init__("connection_error", message)
