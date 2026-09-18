"""
Sessions REST API — spec section 4.3.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from pine_assistant.errors import PineAIError, SessionError
from pine_assistant.models.session import (
    SessionInfo,
    SessionListResponse,
    SessionMessageStatus,
    SessionOutcomesPage,
)
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


def _parse_message_status(payload: Any) -> SessionMessageStatus:
    try:
        return SessionMessageStatus.model_validate(payload)
    except ValidationError:
        raise SessionError("Pine API returned an invalid session message response", "invalid_response") from None


def _parse_outcomes_page(payload: Any) -> SessionOutcomesPage:
    try:
        return SessionOutcomesPage.model_validate(payload)
    except ValidationError:
        raise SessionError("Pine API returned an invalid session outcomes response", "invalid_response") from None


def _request_id(request_id: str | None) -> str | None:
    if request_id is None:
        return None
    if not isinstance(request_id, str):
        raise ValueError("request_id must be a string")
    if len(request_id) > 128 or any(not 0x21 <= ord(char) <= 0x7E for char in request_id):
        raise ValueError("request_id must contain at most 128 visible ASCII characters")
    return request_id or None


def _outcomes_cursor(before: str | int) -> str:
    if isinstance(before, bool) or not isinstance(before, (str, int)):
        raise ValueError("before must be a decimal message identifier")
    value = str(before).strip()
    if not value.isdecimal():
        raise ValueError("before must be a decimal message identifier")
    return value


def _session_error(exc: PineAIError) -> SessionError:
    """Translate transport failures without exposing an upstream response body."""
    code = exc.code
    input_state = (exc.details or {}).get("input_state")
    if isinstance(input_state, dict) and isinstance(input_state.get("code"), str):
        code = input_state["code"]
    return SessionError(str(exc), code, exc.details, status_code=exc.status_code)


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

    async def send_message(
        self,
        session_id: str | int,
        content: str,
        *,
        type: str | None = None,
        action: dict[str, Any] | None = None,
        attachments: list[dict[str, Any]] | None = None,
        library_artifact_ids: list[str] | None = None,
        referenced_sessions: list[dict[str, str]] | None = None,
        quote: dict[str, str] | None = None,
        client_now_date: str | None = None,
        request_id: str | None = None,
    ) -> SessionMessageStatus:
        """Persist and hand off a new user message without waiting for task completion.

        The acknowledgement's ``status`` is ``received``, ``delivered``,
        ``delivery_failed``, or ``failed``. Network failures leave whether the
        message was persisted unknown, so this method never retries a write.
        """
        if not isinstance(content, str):
            raise ValueError("content must be a string")
        final_request_id = _request_id(request_id)
        body: dict[str, Any] = {"content": content}
        for key, value in {
            "type": type,
            "action": action,
            "attachments": attachments,
            "library_artifact_ids": library_artifact_ids,
            "referenced_sessions": referenced_sessions,
            "quote": quote,
            "client_now_date": client_now_date,
        }.items():
            if value is not None:
                body[key] = value
        headers = {"X-Request-ID": final_request_id} if final_request_id else None
        try:
            return _parse_message_status(
                await self._http.post(f"/v2/sessions/{_session_id(session_id)}/messages", body, headers=headers),
            )
        except PineAIError as exc:
            raise _session_error(exc) from exc

    async def outcomes(
        self, session_id: str | int, *, limit: int = 20, before: str | int | None = None,
    ) -> SessionOutcomesPage:
        """List persisted Outcomes newest first; follow ``next_cursor`` for older pages."""
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        params: dict[str, str | int] = {"limit": limit}
        if before is not None:
            params["before"] = _outcomes_cursor(before)
        try:
            return _parse_outcomes_page(
                await self._http.get(f"/v2/sessions/{_session_id(session_id)}/outcomes", params=params),
            )
        except PineAIError as exc:
            raise _session_error(exc) from exc

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

    def send_message(
        self,
        session_id: str | int,
        content: str,
        *,
        type: str | None = None,
        action: dict[str, Any] | None = None,
        attachments: list[dict[str, Any]] | None = None,
        library_artifact_ids: list[str] | None = None,
        referenced_sessions: list[dict[str, str]] | None = None,
        quote: dict[str, str] | None = None,
        client_now_date: str | None = None,
        request_id: str | None = None,
    ) -> SessionMessageStatus:
        """Persist and hand off a new user message without waiting for task completion."""
        if not isinstance(content, str):
            raise ValueError("content must be a string")
        final_request_id = _request_id(request_id)
        body: dict[str, Any] = {"content": content}
        for key, value in {
            "type": type,
            "action": action,
            "attachments": attachments,
            "library_artifact_ids": library_artifact_ids,
            "referenced_sessions": referenced_sessions,
            "quote": quote,
            "client_now_date": client_now_date,
        }.items():
            if value is not None:
                body[key] = value
        headers = {"X-Request-ID": final_request_id} if final_request_id else None
        try:
            return _parse_message_status(
                self._http.post(f"/v2/sessions/{_session_id(session_id)}/messages", body, headers=headers),
            )
        except PineAIError as exc:
            raise _session_error(exc) from exc

    def outcomes(
        self, session_id: str | int, *, limit: int = 20, before: str | int | None = None,
    ) -> SessionOutcomesPage:
        """List persisted Outcomes newest first; follow ``next_cursor`` for older pages."""
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        params: dict[str, str | int] = {"limit": limit}
        if before is not None:
            params["before"] = _outcomes_cursor(before)
        try:
            return _parse_outcomes_page(
                self._http.get(f"/v2/sessions/{_session_id(session_id)}/outcomes", params=params),
            )
        except PineAIError as exc:
            raise _session_error(exc) from exc

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
