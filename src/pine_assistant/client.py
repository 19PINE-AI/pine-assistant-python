"""
PineAI / AsyncPineAI — main SDK clients.
"""

import asyncio
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Generator, Optional

from pine_assistant.transport.http import HttpClient, DEFAULT_BASE_URL
from pine_assistant.transport.socketio import SocketIOManager
from pine_assistant.auth import Auth
from pine_assistant.sessions import SessionsAPI
from pine_assistant.chat import ChatEngine, ChatEvent
from pine_assistant.errors import ConnectionError
from pine_assistant.models.events import C2SEvent

DEVICE_ID_FILE = Path.home() / ".pine" / "device_id"


def _get_or_create_device_id(provided: Optional[str] = None) -> str:
    if provided:
        return provided
    try:
        return DEVICE_ID_FILE.read_text().strip()
    except FileNotFoundError:
        device_id = str(uuid.uuid4())
        try:
            DEVICE_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
            DEVICE_ID_FILE.write_text(device_id)
        except OSError:
            pass
        return device_id


class AsyncPineAI:
    """Async Pine AI client (primary)."""

    def __init__(
        self,
        access_token: Optional[str] = None,
        user_id: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        device_id: Optional[str] = None,
        transports: Optional[list[str]] = None,
        ready_timeout: float = 15.0,
    ):
        self._base_url = base_url
        self._access_token = access_token
        self._user_id = user_id
        self._device_id = _get_or_create_device_id(device_id)
        self._transports = transports
        self._ready_timeout = ready_timeout

        self.http = HttpClient(base_url=base_url, token=access_token)
        self.auth = Auth(self.http)
        self.sessions = SessionsAPI(self.http)

        self._sio: Optional[SocketIOManager] = None
        self._chat: Optional[ChatEngine] = None

    @property
    def connected(self) -> bool:
        return self._sio is not None and self._sio.connected

    async def connect(self, access_token: Optional[str] = None, user_id: Optional[str] = None) -> None:
        token = access_token or self._access_token
        uid = user_id or self._user_id
        if not token or not uid:
            raise ConnectionError("access_token and user_id required. Run auth flow first.")
        self.http.set_token(token)

        self._sio = SocketIOManager(
            base_url=self._base_url,
            token=token,
            user_id=uid,
            device_id=self._device_id,
            transports=self._transports,
            ready_timeout=self._ready_timeout,
        )
        self._chat = ChatEngine(self._sio, check_session_state=self.sessions.get)
        await self._sio.connect()

    async def disconnect(self) -> None:
        if self._sio:
            await self._sio.disconnect()
            self._sio = None
            self._chat = None

    async def join_session(self, session_id: str) -> dict[str, Any]:
        """Join a session room — must be called before chatting."""
        self._ensure_connected()
        return await self._chat.join_session(session_id)  # type: ignore[union-attr]

    def leave_session(self, session_id: str) -> None:
        """Leave a session room when done."""
        self._ensure_connected()
        self._chat.leave_session(session_id)  # type: ignore[union-attr]

    async def get_history(
        self, session_id: str, max_messages: int = 30, order: str = "desc",
        from_message_id: Optional[str] = None, request_work_log: bool = False,
    ) -> dict[str, Any]:
        """Fetch message history — spec 5.1.1 session:history."""
        self._ensure_connected()
        return await self._sio.emit_and_wait(  # type: ignore[union-attr]
            C2SEvent.SESSION_HISTORY,
            {
                "max_messages": max_messages,
                "max_bytes": 5_242_880,
                "order": order,
                "from_message_id": from_message_id,
                "request_work_log": request_work_log,
            },
            session_id=session_id,
        )

    async def chat(
        self,
        session_id: str,
        content: str,
        *,
        attachments: Optional[list[dict[str, Any]]] = None,
        referenced_sessions: Optional[list[dict[str, str]]] = None,
        action: Optional[dict[str, Any]] = None,
    ) -> AsyncGenerator[ChatEvent, None]:
        """Send a message and yield buffered events."""
        self._ensure_connected()
        async for event in self._chat.chat(  # type: ignore[union-attr]
            session_id, content,
            attachments=attachments,
            referenced_sessions=referenced_sessions,
            action=action,
        ):
            yield event

    def send_message(
        self,
        session_id: str,
        content: str,
        *,
        attachments: Optional[list[dict[str, Any]]] = None,
        referenced_sessions: Optional[list[dict[str, str]]] = None,
        action: Optional[dict[str, Any]] = None,
    ) -> None:
        """Send a message without waiting for events (fire-and-forget)."""
        self._ensure_connected()
        self._chat.send_message(  # type: ignore[union-attr]
            session_id, content,
            attachments=attachments,
            referenced_sessions=referenced_sessions,
            action=action,
        )

    async def listen(self, session_id: str) -> AsyncGenerator[ChatEvent, None]:
        """Listen for events on a joined session without sending a message."""
        self._ensure_connected()
        async for event in self._chat._listen(session_id):  # type: ignore[union-attr]
            yield event

    async def create_and_chat(self, content: str) -> AsyncGenerator[ChatEvent, None]:
        """Convenience: create session, join, chat, return events."""
        session = await self.sessions.create()
        sid = session["id"]
        await self.join_session(sid)
        try:
            async for event in self.chat(sid, content):
                yield event
        finally:
            self.leave_session(sid)

    def send_form_response(self, session_id: str, message_id: str, form_data: dict[str, Any]) -> None:
        """Submit a form response."""
        self._ensure_connected()
        self._chat.send_form_response(session_id, message_id, form_data)  # type: ignore[union-attr]

    def send_auth_confirmation(self, session_id: str, message_id: str, data: dict[str, Any]) -> None:
        """Submit an interactive auth confirmation (OTP, etc)."""
        self._ensure_connected()
        self._chat.send_auth_confirmation(session_id, message_id, data)  # type: ignore[union-attr]

    def send_location_response(self, session_id: str, message_id: str, latitude: str, longitude: str) -> None:
        """Submit a location response."""
        self._ensure_connected()
        self._chat.send_location_response(session_id, message_id, latitude, longitude)  # type: ignore[union-attr]

    def send_location_selection(self, session_id: str, message_id: str, places: list[dict[str, Any]]) -> None:
        """Submit a location selection."""
        self._ensure_connected()
        self._chat.send_location_selection(session_id, message_id, places)  # type: ignore[union-attr]

    @staticmethod
    def session_url(session_id: str) -> str:
        """Build the Pine AI web app URL for a session (for payment)."""
        return f"https://www.19pine.ai/app/chat/{session_id}"

    def _ensure_connected(self) -> None:
        if not self._chat or not self._sio or not self._sio.connected:
            raise ConnectionError("Not connected. Call connect() first.")


class PineAI:
    """Sync wrapper around AsyncPineAI. Runs the event loop internally."""

    def __init__(self, **kwargs: Any):
        self._async = AsyncPineAI(**kwargs)
        self._loop = asyncio.new_event_loop()

    def _run(self, coro: Any) -> Any:
        return self._loop.run_until_complete(coro)

    @property
    def auth(self) -> Auth:
        return self._async.auth

    @property
    def sessions(self) -> SessionsAPI:
        return self._async.sessions

    @property
    def connected(self) -> bool:
        return self._async.connected

    def connect(self, **kwargs: Any) -> None:
        self._run(self._async.connect(**kwargs))

    def disconnect(self) -> None:
        self._run(self._async.disconnect())

    def join_session(self, session_id: str) -> dict[str, Any]:
        return self._run(self._async.join_session(session_id))

    def leave_session(self, session_id: str) -> None:
        self._async.leave_session(session_id)

    def get_history(self, session_id: str, **kwargs: Any) -> dict[str, Any]:
        return self._run(self._async.get_history(session_id, **kwargs))

    def chat_sync(
        self,
        session_id: str,
        content: str,
        *,
        attachments: Optional[list[dict[str, Any]]] = None,
        referenced_sessions: Optional[list[dict[str, str]]] = None,
        action: Optional[dict[str, Any]] = None,
    ) -> list[ChatEvent]:
        """Send a message and return all events as a list (blocking)."""
        async def _collect() -> list[ChatEvent]:
            events = []
            async for event in self._async.chat(
                session_id, content,
                attachments=attachments,
                referenced_sessions=referenced_sessions,
                action=action,
            ):
                events.append(event)
            return events
        return self._run(_collect())

    def send_message(
        self,
        session_id: str,
        content: str,
        *,
        attachments: Optional[list[dict[str, Any]]] = None,
        referenced_sessions: Optional[list[dict[str, str]]] = None,
        action: Optional[dict[str, Any]] = None,
    ) -> None:
        """Send a message without waiting for events (fire-and-forget)."""
        self._async.send_message(
            session_id, content,
            attachments=attachments,
            referenced_sessions=referenced_sessions,
            action=action,
        )

    def send_form_response(self, session_id: str, message_id: str, form_data: dict[str, Any]) -> None:
        self._async.send_form_response(session_id, message_id, form_data)

    def send_auth_confirmation(self, session_id: str, message_id: str, data: dict[str, Any]) -> None:
        self._async.send_auth_confirmation(session_id, message_id, data)

    def send_location_response(self, session_id: str, message_id: str, latitude: str, longitude: str) -> None:
        self._async.send_location_response(session_id, message_id, latitude, longitude)

    def send_location_selection(self, session_id: str, message_id: str, places: list[dict[str, Any]]) -> None:
        self._async.send_location_selection(session_id, message_id, places)

    session_url = staticmethod(AsyncPineAI.session_url)
