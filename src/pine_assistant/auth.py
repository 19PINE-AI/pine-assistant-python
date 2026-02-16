"""
Auth module — spec section 4.1.

Two-step email verification. No reCAPTCHA required.
"""

from typing import Any

from pine_assistant.transport.http import HttpClient
from pine_assistant.errors import AuthError


class Auth:
    def __init__(self, http: HttpClient):
        self._http = http

    async def request_code(self, email: str) -> dict[str, Any]:
        """Step 1: Request verification code — spec 4.1.1"""
        try:
            return await self._http.post("/v2/auth/email/request", {"email": email}, authenticated=False)
        except Exception as e:
            raise AuthError(f"Failed to request auth code: {e}")

    async def verify_code(self, email: str, code: str, request_token: str) -> dict[str, Any]:
        """Step 2: Verify code and get access token — spec 4.1.2"""
        try:
            result = await self._http.post(
                "/v2/auth/email/verify",
                {"email": email, "code": code, "request_token": request_token},
                authenticated=False,
            )
            self._http.set_token(result["access_token"])
            return result
        except Exception as e:
            raise AuthError(f"Failed to verify auth code: {e}")
