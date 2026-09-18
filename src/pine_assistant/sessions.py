"""
Sessions REST API — spec section 4.3.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from pine_assistant.errors import PineAIError, SessionError
from pine_assistant.models.session import SessionInfo, SessionListResponse
from pine_assistant.transport.http import HttpClient, SyncHttpClient


def _session_id(session_id: str | int) -> str:
    if isinstance(session_id, bool) or not isinstance(session_id, (str, int)):
        raise ValueError("session_id must be a string or integer")
    value = str(session_id).strip()
    if not value.isdecimal() or int(value) <= 0:
        raise ValueError("session_id must be a positive decimal identifier")
    return value


def _parse_session(payload: Any) -> SessionInfo:
    try:
        return SessionInfo.model_validate(payload)
    except ValidationError:
        raise SessionError("Pine API returned an invalid session response", "invalid_response") from None


def _parse_session_list(payload: Any) -> SessionListResponse:
    try:
        return SessionListResponse.model_validate(payload)
    except ValidationError:
        raise SessionError("Pine API returned an invalid session list response", "invalid_response") from None


class SessionsAPI:
    def __init__(self, http: HttpClient):
        self._http = http

    async def list(
        self, state: str | None = None, limit: int = 30, offset: int = 0, *, ensure_copilot: bool = False,
    ) -> SessionListResponse:
        """List sessions without creating a Copilot session by default."""
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        params: dict[str, str | int] = {
            "limit": limit,
            "offset": offset,
            "ensure_copilot": str(ensure_copilot).lower(),
        }
        if state:
            params["state"] = state
        try:
            return _parse_session_list(await self._http.get("/v2/sessions", params=params))
        except PineAIError as exc:
            raise SessionError(str(exc), exc.code, status_code=exc.status_code) from exc

    async def get(self, session_id: str | int) -> SessionInfo:
        """Get session — spec 4.3.2"""
        try:
            return _parse_session(await self._http.get(f"/v2/sessions/{_session_id(session_id)}"))
        except PineAIError as exc:
            raise SessionError(str(exc), exc.code, status_code=exc.status_code) from exc

    async def create(self) -> dict[str, Any]:
        """Create session — spec 4.3.3"""
        return await self._http.post("/v2/sessions")

    async def delete(self, session_id: str, force_delete: bool = False) -> Any:
        """Delete session — spec 4.3.4"""
        params = {"force_delete": "true"} if force_delete else None
        return await self._http.delete(f"/v2/sessions/{session_id}", params=params)

    async def start_task(self, session_id: str) -> dict[str, Any]:
        """Start task — spec 4.3.9"""
        return await self._http.post(f"/v2/sessions/{session_id}/start")

    async def stop_task(self, session_id: str) -> dict[str, Any]:
        """Stop task — spec 4.3.10"""
        return await self._http.post(f"/v2/sessions/{session_id}/stop")

    async def update_scheduled_call_reminder(
        self, session_id: str, message_id: str, scheduled_time: str, enabled: bool,
    ) -> dict[str, Any]:
        """Update scheduled call reminder — spec 4.3.5"""
        return await self._http.put(f"/v2/sessions/{session_id}/scheduled-call-reminder", {
            "message_id": message_id,
            "scheduled_time": scheduled_time,
            "scheduled_call_reminder": enabled,
        })

    async def social_share(
        self, session_id: str, platform: str, shared_url: str,
    ) -> dict[str, Any]:
        """Social share — spec 4.3.11. Earn credits for sharing results."""
        return await self._http.post(f"/v2/sessions/{session_id}/social-share", {
            "metadata": {"platform": platform, "shared_url": shared_url},
        })

    async def upload_attachment(self, file_path: str) -> list[dict[str, Any]]:
        """Upload attachment — spec 4.4.1. Multipart form upload."""
        return await self._http.upload("/v2/attachments", file_path)

    async def delete_attachment(self, attachment_id: str) -> None:
        """Delete attachment — spec 4.4.2"""
        await self._http.delete(f"/v2/attachments/{attachment_id}")


class SyncSessionsAPI:
    """Synchronous REST session resource used by :class:`pine_assistant.PineAI`."""

    def __init__(self, http: SyncHttpClient):
        self._http = http

    def list(
        self, state: str | None = None, limit: int = 30, offset: int = 0, *, ensure_copilot: bool = False,
    ) -> SessionListResponse:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        params: dict[str, str | int] = {
            "limit": limit,
            "offset": offset,
            "ensure_copilot": str(ensure_copilot).lower(),
        }
        if state:
            params["state"] = state
        try:
            return _parse_session_list(self._http.get("/v2/sessions", params=params))
        except PineAIError as exc:
            raise SessionError(str(exc), exc.code, status_code=exc.status_code) from exc

    def get(self, session_id: str | int) -> SessionInfo:
        try:
            return _parse_session(self._http.get(f"/v2/sessions/{_session_id(session_id)}"))
        except PineAIError as exc:
            raise SessionError(str(exc), exc.code, status_code=exc.status_code) from exc

    def create(self) -> dict[str, Any]:
        return self._http.post("/v2/sessions")

    def delete(self, session_id: str, force_delete: bool = False) -> Any:
        params = {"force_delete": "true"} if force_delete else None
        return self._http.delete(f"/v2/sessions/{session_id}", params=params)

    def start_task(self, session_id: str) -> dict[str, Any]:
        return self._http.post(f"/v2/sessions/{session_id}/start")

    def stop_task(self, session_id: str) -> dict[str, Any]:
        return self._http.post(f"/v2/sessions/{session_id}/stop")

    def update_scheduled_call_reminder(
        self, session_id: str, message_id: str, scheduled_time: str, enabled: bool,
    ) -> dict[str, Any]:
        return self._http.put(f"/v2/sessions/{session_id}/scheduled-call-reminder", {
            "message_id": message_id,
            "scheduled_time": scheduled_time,
            "scheduled_call_reminder": enabled,
        })

    def social_share(self, session_id: str, platform: str, shared_url: str) -> dict[str, Any]:
        return self._http.post(f"/v2/sessions/{session_id}/social-share", {
            "metadata": {"platform": platform, "shared_url": shared_url},
        })

    def upload_attachment(self, file_path: str) -> list[dict[str, Any]]:
        return self._http.upload("/v2/attachments", file_path)

    def delete_attachment(self, attachment_id: str) -> None:
        self._http.delete(f"/v2/attachments/{attachment_id}")
