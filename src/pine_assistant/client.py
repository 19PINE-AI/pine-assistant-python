"""
PineAI / AsyncPineAI — main SDK clients.
"""

import asyncio
import logging
import os
import uuid
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import Any

from pine_assistant.auth import Auth
from pine_assistant.chat import ChatEngine, ChatEvent, Deduplicator, event_from_envelope
from pine_assistant.errors import ConnectionError
from pine_assistant.models.events import C2SEvent
from pine_assistant.sessions import SessionsAPI
from pine_assistant.transport.http import DEFAULT_BASE_URL, HttpClient
from pine_assistant.transport.socketio import SocketIOManager

DEVICE_ID_FILE = Path.home() / ".pine" / "device_id"
DEVICE_ID_ENV = "PINE_DEVICE_ID"

HISTORY_PAGE_SIZE = 30
HISTORY_MAX_BYTES = 5_242_880

_logger = logging.getLogger(__name__)


def _get_or_create_device_id(provided: str | None = None) -> str:
    """Resolve a stable device_id.

    Precedence: explicit argument → PINE_DEVICE_ID env var → ~/.pine/device_id
    → a new random UUID persisted to ~/.pine/device_id.

    If persistence fails, a warning is logged so callers running in sandboxed
    or read-only environments learn why their device_id rotates on each launch
    — set PINE_DEVICE_ID to pin it.
    """
    if provided:
        return provided
    env_value = os.environ.get(DEVICE_ID_ENV)
    if env_value:
        return env_value.strip()
    try:
        return DEVICE_ID_FILE.read_text().strip()
    except (FileNotFoundError, OSError):
        device_id = str(uuid.uuid4())
        try:
            DEVICE_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
            DEVICE_ID_FILE.write_text(device_id)
        except OSError as exc:
            _logger.warning(
                "Could not persist device_id to %s (%s). "
                "Set the %s env var to pin a stable device_id.",
                DEVICE_ID_FILE, exc, DEVICE_ID_ENV,
            )
        return device_id


class AsyncPineAI:
    """Async Pine AI client (primary).

    A client tracks one session. Concurrent sessions need one client each.
    """

    def __init__(
        self,
        access_token: str | None = None,
        user_id: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        device_id: str | None = None,
        transports: list[str] | None = None,
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

        self._sio: SocketIOManager | None = None
        self._chat: ChatEngine | None = None

    @property
    def connected(self) -> bool:
        return self._sio is not None and self._sio.connected

    async def connect(self, access_token: str | None = None, user_id: str | None = None) -> None:
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
        """Enter a session — must be called before chatting.

        The join carries `since_revision` "0" and its incremental-sync fields
        are ignored. Call `rebuild()` after joining to load the session's
        messages; a join alone does not deliver them.
        """
        self._ensure_connected()
        return await self._chat.join_session(session_id)  # type: ignore[union-attr]

    def leave_session(self, session_id: str) -> None:
        """Leave a session room.

        Room management, outside the supported protocol scope.
        """
        self._ensure_connected()
        self._chat.leave_session(session_id)  # type: ignore[union-attr]

    def on_reconnect(self, handler: Callable[[], None]) -> Callable[[], None]:
        """Register a callback fired after a reconnect re-joins its sessions.

        Rebuild from `rebuild()` when it fires: a connection can stay open after
        delivery has stopped, so anything accumulated before it is unreliable.
        """
        self._ensure_connected()
        return self._sio.add_reconnect_handler(handler)  # type: ignore[union-attr]

    async def get_history(
        self, session_id: str, max_messages: int = HISTORY_PAGE_SIZE, order: str = "desc",
        from_message_id: str | None = None,
    ) -> dict[str, Any]:
        """Fetch one page of persisted messages.

        A short or empty page does not mean the range is exhausted — only an
        absent `next_message_id` does. Use `rebuild()` unless you are paging
        deliberately.
        """
        self._ensure_connected()
        return await self._sio.emit_and_wait(  # type: ignore[union-attr]
            C2SEvent.SESSION_HISTORY,
            {
                "max_messages": max_messages,
                "max_bytes": HISTORY_MAX_BYTES,
                "order": order,
                "from_message_id": from_message_id,
            },
            session_id=session_id,
        )

    async def rebuild(
        self, session_id: str, *, page_size: int = HISTORY_PAGE_SIZE, max_pages: int = 100,
    ) -> list[dict[str, Any]]:
        """Rebuild a session's messages from history, unconditionally.

        This is the recovery mechanism: run it on every join and every
        reconnect, and whenever a tracked session has been silent for an
        extended period. Paging stops only when the cursor is exhausted.

        Messages of every type are returned, including ones outside the
        supported scope; filtering is the caller's to do.
        """
        messages: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(max_pages):
            page = await self.get_history(session_id, max_messages=page_size, from_message_id=cursor)
            messages.extend(page.get("messages") or [])
            cursor = page.get("next_message_id") or None
            if not cursor:
                return messages
        _logger.warning(
            "rebuild(%s) stopped at the %d-page limit with the cursor still open; "
            "the returned history is incomplete.", session_id, max_pages,
        )
        return messages

    async def chat(
        self,
        session_id: str,
        content: str,
        *,
        attachments: list[dict[str, Any]] | None = None,
        referenced_sessions: list[dict[str, str]] | None = None,
        turn_timeout: float | None = None,
    ) -> AsyncGenerator[ChatEvent, None]:
        """Send a message and yield the events that follow.

        Events the SDK does not recognise are yielded unchanged alongside the
        rest; ignore what you do not handle.

        `turn_timeout` bounds the call in wall-clock seconds, keeping whatever
        arrived before it elapsed. Without one, a turn the session never closes
        is waited on indefinitely.
        """
        self._ensure_connected()
        async for event in self._chat.chat(  # type: ignore[union-attr]
            session_id, content,
            attachments=attachments,
            referenced_sessions=referenced_sessions,
            turn_timeout=turn_timeout,
        ):
            yield event

    def send_message(
        self,
        session_id: str,
        content: str,
        *,
        attachments: list[dict[str, Any]] | None = None,
        referenced_sessions: list[dict[str, str]] | None = None,
    ) -> None:
        """Send a message without waiting for events (fire-and-forget)."""
        self._ensure_connected()
        self._chat.send_message(  # type: ignore[union-attr]
            session_id, content,
            attachments=attachments,
            referenced_sessions=referenced_sessions,
        )

    async def listen(
        self, session_id: str, turn_timeout: float | None = None,
    ) -> AsyncGenerator[ChatEvent, None]:
        """Listen for events on a joined session without sending a message."""
        self._ensure_connected()
        async for event in self._chat._listen(  # type: ignore[union-attr]
            session_id, turn_timeout=turn_timeout,
        ):
            yield event

    async def subscribe(self, session_id: str) -> AsyncGenerator[ChatEvent, None]:
        """Persistent event stream for a session — yields events indefinitely.

        Unlike listen(), this never terminates on terminal states or timeouts.
        Designed for bidirectional use where sending and receiving are
        concurrent.
        """
        self._ensure_connected()
        queue: asyncio.Queue[ChatEvent] = asyncio.Queue()
        dedup = Deduplicator()

        def _handler(event_type: str, raw: dict[str, Any]) -> None:
            payload = raw.get("payload") or {}
            p_sid = payload.get("session_id")
            if p_sid and p_sid != session_id:
                return
            event = event_from_envelope(event_type, raw, session_id)
            if dedup.is_duplicate(event):
                return
            queue.put_nowait(event)

        remove = self._sio.add_event_handler(_handler)  # type: ignore[union-attr]
        try:
            while self.connected:
                try:
                    yield await asyncio.wait_for(queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue
        finally:
            remove()

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
        """Answer a `session:form_to_user` request.

        Submit only values the user supplied. The format has no representation
        for refusal and an empty submission is indistinguishable from empty
        answers, so a fabricated answer may be acted on. Sending nothing is
        safe.
        """
        self._ensure_connected()
        self._chat.send_form_response(session_id, message_id, form_data)  # type: ignore[union-attr]

    def emit_event(
        self, event_type: str, data: Any, session_id: str, message_id: str | None = None,
    ) -> None:
        """Send an arbitrary event, enveloped but otherwise unprocessed.

        The escape hatch for the unsupported surface. Anything sent through it
        may stop working without notice and without a version change.
        """
        self._ensure_connected()
        self._sio.emit(event_type, data, session_id, message_id)  # type: ignore[union-attr]

    @staticmethod
    def session_url(session_id: str) -> str:
        """Build the Pine AI web app URL for a session."""
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

    def rebuild(self, session_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        return self._run(self._async.rebuild(session_id, **kwargs))

    def chat_sync(
        self,
        session_id: str,
        content: str,
        *,
        attachments: list[dict[str, Any]] | None = None,
        referenced_sessions: list[dict[str, str]] | None = None,
        turn_timeout: float | None = None,
    ) -> list[ChatEvent]:
        """Send a message and return all events as a list (blocking)."""
        async def _collect() -> list[ChatEvent]:
            events = []
            async for event in self._async.chat(
                session_id, content,
                attachments=attachments,
                referenced_sessions=referenced_sessions,
                turn_timeout=turn_timeout,
            ):
                events.append(event)
            return events
        return self._run(_collect())

    def send_message(
        self,
        session_id: str,
        content: str,
        *,
        attachments: list[dict[str, Any]] | None = None,
        referenced_sessions: list[dict[str, str]] | None = None,
    ) -> None:
        """Send a message without waiting for events (fire-and-forget)."""
        self._async.send_message(
            session_id, content,
            attachments=attachments,
            referenced_sessions=referenced_sessions,
        )

    def send_form_response(self, session_id: str, message_id: str, form_data: dict[str, Any]) -> None:
        self._async.send_form_response(session_id, message_id, form_data)

    def emit_event(
        self, event_type: str, data: Any, session_id: str, message_id: str | None = None,
    ) -> None:
        self._async.emit_event(event_type, data, session_id, message_id)

    session_url = staticmethod(AsyncPineAI.session_url)
