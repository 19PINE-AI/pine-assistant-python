"""
Auth module — spec section 4.1.

Two-step email verification. No reCAPTCHA required.
"""

from typing import Any

from pydantic import ValidationError

from pine_assistant.errors import AuthError, PineAIError
from pine_assistant.models.auth import AuthIdentity, AuthTicket, RedeemedTicket
from pine_assistant.transport.http import HttpClient, SyncHttpClient


def _auth_failure(error: PineAIError) -> AuthError:
    return AuthError("Pine authentication request failed", error.code, status_code=error.status_code)


def _non_empty(value: str, field: str, *, min_length: int = 1, max_length: int | None = None) -> str:
    if not isinstance(value, str) or not (min_length <= len(value) <= (max_length or len(value))):
        raise ValueError(f"{field} has an invalid length")
    return value


class Auth:
    def __init__(self, http: HttpClient):
        self._http = http

    async def request_code(self, email: str) -> dict[str, Any]:
        """Step 1: Request verification code — spec 4.1.1"""
        try:
            return await self._http.post("/v2/auth/email/request", {"email": email}, authenticated=False)
        except PineAIError as exc:
            raise _auth_failure(exc) from exc

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
        except PineAIError as exc:
            raise _auth_failure(exc) from exc

    async def me(self) -> AuthIdentity:
        """Return the authenticated account's minimal stable identity."""
        try:
            return AuthIdentity.model_validate(await self._http.get("/v2/auth/me"))
        except PineAIError as exc:
            raise _auth_failure(exc) from exc
        except ValidationError:
            raise AuthError("Pine API returned an invalid identity response", "invalid_response") from None

    async def mint_ticket(
        self,
        *,
        purpose: str,
        client_id: str,
        auth_attempt_id: str,
        code_challenge: str,
    ) -> AuthTicket:
        """Mint a short-lived, purpose-bound authorization ticket."""
        _non_empty(purpose, "purpose")
        _non_empty(client_id, "client_id")
        _non_empty(auth_attempt_id, "auth_attempt_id", min_length=22, max_length=64)
        _non_empty(code_challenge, "code_challenge", min_length=43, max_length=43)
        try:
            return AuthTicket.model_validate(await self._http.post("/v2/auth/tickets", {
                "purpose": purpose,
                "client_id": client_id,
                "auth_attempt_id": auth_attempt_id,
                "code_challenge": code_challenge,
            }))
        except PineAIError as exc:
            raise _auth_failure(exc) from exc
        except ValidationError:
            raise AuthError("Pine API returned an invalid ticket response", "invalid_response") from None

    async def redeem_ticket(
        self,
        *,
        ticket: str,
        auth_attempt_id: str,
        code_verifier: str,
        service_token: str,
    ) -> RedeemedTicket:
        """Redeem a ticket using a service bearer without changing this client's user bearer."""
        _non_empty(ticket, "ticket", min_length=43, max_length=43)
        _non_empty(auth_attempt_id, "auth_attempt_id", min_length=22, max_length=64)
        _non_empty(code_verifier, "code_verifier", min_length=43, max_length=128)
        _non_empty(service_token, "service_token")
        try:
            return RedeemedTicket.model_validate(await self._http.post(
                "/v2/auth/tickets/redeem",
                {"ticket": ticket, "auth_attempt_id": auth_attempt_id, "code_verifier": code_verifier},
                token=service_token,
            ))
        except PineAIError as exc:
            raise _auth_failure(exc) from exc
        except ValidationError:
            raise AuthError("Pine API returned an invalid ticket redemption response", "invalid_response") from None


class SyncAuth:
    """Synchronous REST authentication resource."""

    def __init__(self, http: SyncHttpClient):
        self._http = http

    def request_code(self, email: str) -> dict[str, Any]:
        try:
            return self._http.post("/v2/auth/email/request", {"email": email}, authenticated=False)
        except PineAIError as exc:
            raise _auth_failure(exc) from exc

    def verify_code(self, email: str, code: str, request_token: str) -> dict[str, Any]:
        try:
            result = self._http.post(
                "/v2/auth/email/verify",
                {"email": email, "code": code, "request_token": request_token},
                authenticated=False,
            )
            self._http.set_token(result["access_token"])
            return result
        except PineAIError as exc:
            raise _auth_failure(exc) from exc

    def me(self) -> AuthIdentity:
        try:
            return AuthIdentity.model_validate(self._http.get("/v2/auth/me"))
        except PineAIError as exc:
            raise _auth_failure(exc) from exc
        except ValidationError:
            raise AuthError("Pine API returned an invalid identity response", "invalid_response") from None

    def mint_ticket(
        self, *, purpose: str, client_id: str, auth_attempt_id: str, code_challenge: str,
    ) -> AuthTicket:
        _non_empty(purpose, "purpose")
        _non_empty(client_id, "client_id")
        _non_empty(auth_attempt_id, "auth_attempt_id", min_length=22, max_length=64)
        _non_empty(code_challenge, "code_challenge", min_length=43, max_length=43)
        try:
            return AuthTicket.model_validate(self._http.post("/v2/auth/tickets", {
                "purpose": purpose,
                "client_id": client_id,
                "auth_attempt_id": auth_attempt_id,
                "code_challenge": code_challenge,
            }))
        except PineAIError as exc:
            raise _auth_failure(exc) from exc
        except ValidationError:
            raise AuthError("Pine API returned an invalid ticket response", "invalid_response") from None

    def redeem_ticket(
        self, *, ticket: str, auth_attempt_id: str, code_verifier: str, service_token: str,
    ) -> RedeemedTicket:
        _non_empty(ticket, "ticket", min_length=43, max_length=43)
        _non_empty(auth_attempt_id, "auth_attempt_id", min_length=22, max_length=64)
        _non_empty(code_verifier, "code_verifier", min_length=43, max_length=128)
        _non_empty(service_token, "service_token")
        try:
            return RedeemedTicket.model_validate(self._http.post(
                "/v2/auth/tickets/redeem",
                {"ticket": ticket, "auth_attempt_id": auth_attempt_id, "code_verifier": code_verifier},
                token=service_token,
            ))
        except PineAIError as exc:
            raise _auth_failure(exc) from exc
        except ValidationError:
            raise AuthError("Pine API returned an invalid ticket redemption response", "invalid_response") from None
