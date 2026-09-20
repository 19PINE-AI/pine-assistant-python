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

import httpx

from pine_assistant.auth import Auth, SyncAuth
from pine_assistant.chat import ChatEngine, ChatEvent, Deduplicator, event_from_envelope
from pine_assistant.errors import ConnectionError
from pine_assistant.models.events import C2SEvent
from pine_assistant.models.form import FormSubmissionResult, FormToUserData, encode_form_answers
from pine_assistant.sessions import SessionsAPI, SyncSessionsAPI
from pine_assistant.transport.http import DEFAULT_API_BASE_PATH, DEFAULT_BASE_URL, HttpClient, SyncHttpClient
from pine_assistant.transport.socketio import SocketIOManager

DEVICE_ID_FILE = Path.home() / ".pine" / "device_id"
DEVICE_ID_ENV = "PINE_DEVICE_ID"

HISTORY_PAGE_SIZE = 30
HISTORY_MAX_BYTES = 5_242_880
HISTORY_MAX_MESSAGES = 100
HISTORY_MAX_PAGES = 100
FORM_RESPONSE_TIMEOUT_S = 10.0
FORM_RESPONSE_MAX_TIMEOUT_S = 30.0

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
        *,
        api_base_path: str = DEFAULT_API_BASE_PATH,
        http_client: httpx.AsyncClient | None = None,
        http_transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._base_url = base_url
        self._access_token = access_token
        self._user_id = user_id
        self._device_id = _get_or_create_device_id(device_id)
        self._transports = transports
        self._ready_timeout = ready_timeout

        self.http = HttpClient(
            base_url=base_url,
            token=access_token,
            api_base_path=api_base_path,
            client=http_client,
            transport=http_transport,
        )
        self.auth = Auth(self.http)
        self.sessions = SessionsAPI(self.http)

        self._sio: SocketIOManager | None = None
        self._chat: ChatEngine | None = None

    @property
    def connected(self) -> bool:
        return self._sio is not None and self._sio.connected

    async def __aenter__(self) -> "AsyncPineAI":
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

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
        self._chat = ChatEngine(self._sio, check_session_state=self._session_state)
        await self._sio.connect()

    async def disconnect(self) -> None:
        """Disconnect real-time Socket.IO only; the REST client stays usable."""
        if self._sio:
            await self._sio.disconnect()
            self._sio = None
            self._chat = None

    async def aclose(self) -> None:
        """Release Socket.IO and SDK-owned HTTP resources, even if either fails."""
        try:
            await self.disconnect()
        finally:
            await self.http.close()

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
        if isinstance(max_messages, bool) or not isinstance(max_messages, int) or not 1 <= max_messages <= HISTORY_MAX_MESSAGES:
            raise ValueError(f"max_messages must be between 1 and {HISTORY_MAX_MESSAGES}")
        if order.upper() not in ("ASC", "DESC"):
            raise ValueError("order must be ASC or DESC")
        return await self._sio.emit_and_wait(  # type: ignore[union-attr]
            C2SEvent.SESSION_HISTORY,
            {
                "max_messages": max_messages,
                "max_bytes": HISTORY_MAX_BYTES,
                "order": order.upper(),
                "from_message_id": from_message_id,
            },
            session_id=session_id,
        )

    async def rebuild(
        self, session_id: str, *, page_size: int = HISTORY_PAGE_SIZE, max_pages: int = HISTORY_MAX_PAGES,
    ) -> list[dict[str, Any]]:
        """Rebuild a session's messages from history, unconditionally.

        This is the recovery mechanism: run it on every join and every
        reconnect, and whenever a tracked session has been silent for an
        extended period. Paging stops only when the cursor is exhausted.

        Messages of every type are returned, including ones outside the
        supported scope; filtering is the caller's to do.
        """
        if isinstance(max_pages, bool) or not isinstance(max_pages, int) or not 1 <= max_pages <= HISTORY_MAX_PAGES:
            raise ValueError(f"max_pages must be between 1 and {HISTORY_MAX_PAGES}")
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

    async def _find_form_request(
        self, session_id: str, message_id: str, *, deadline: float,
    ) -> dict[str, Any]:
        """Locate one agent-authored form request in bounded persisted history."""
        cursor: str | None = None
        for _ in range(HISTORY_MAX_PAGES):
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("timed out locating the persisted form request")
            page = await asyncio.wait_for(
                self.get_history(session_id, max_messages=HISTORY_PAGE_SIZE, order="DESC", from_message_id=cursor),
                timeout=remaining,
            )
            for message in page.get("messages") or []:
                if not isinstance(message, dict):
                    continue
                payload = message.get("payload")
                metadata = message.get("metadata")
                source = metadata.get("source") if isinstance(metadata, dict) else None
                if (
                    message.get("type") == C2SEvent.SESSION_FORM_TO_USER
                    and isinstance(payload, dict)
                    and str(payload.get("message_id")) == message_id
                    and isinstance(metadata, dict)
                    and isinstance(source, dict)
                    and source.get("role") == "agent"
                    and isinstance(metadata.get("request_id"), str)
                    and metadata["request_id"]
                ):
                    return message
            cursor = page.get("next_message_id") or None
            if not cursor:
                break
        raise ValueError("form request is not available in bounded session history")

    async def submit_form_response(
        self,
        session_id: str,
        message_id: str,
        answers: dict[str, Any],
        *,
        timeout: float = FORM_RESPONSE_TIMEOUT_S,
    ) -> FormSubmissionResult:
        """Submit answers to a persisted agent form request.

        The original request is re-read from authenticated history. Its message
        ID and request ID are the only association used on the wire; callers
        cannot supply PII metadata or form definitions. The result reports only
        observed persistence and delivery, never a Socket.IO ACK.
        """
        self._ensure_connected()
        if not 0 < timeout <= FORM_RESPONSE_MAX_TIMEOUT_S:
            raise ValueError(f"timeout must be between 0 and {FORM_RESPONSE_MAX_TIMEOUT_S} seconds")
        deadline = asyncio.get_running_loop().time() + timeout
        original = await self._find_form_request(session_id, message_id, deadline=deadline)
        payload = original["payload"]
        metadata = original["metadata"]
        try:
            form = FormToUserData.model_validate(payload.get("data"))
            content = encode_form_answers(form.form, answers)
        except (TypeError, ValueError):
            # Do not attach answer values to failures: form values can be PII.
            raise ValueError("form response does not match the persisted form") from None
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError("timed out before form response could be submitted")
        return await self._chat.submit_form_response(  # type: ignore[union-attr]
            session_id, message_id, metadata["request_id"], content, timeout=remaining,
        )

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
        """Deprecated fire-and-forget form response.

        Use :meth:`submit_form_response`, which verifies the original form and
        reports persisted delivery. This legacy method intentionally remains
        synchronous for compatibility and cannot provide either guarantee.
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

    async def _session_state(self, session_id: str) -> dict[str, Any]:
        """Adapt the typed REST resource to the legacy realtime pre-check."""
        return (await self.sessions.get(session_id)).model_dump()


class PineAI:
    """Synchronous Pine REST client.

    REST resources return concrete values through ``httpx.Client``. Real-time
    Socket.IO requires an event loop and is intentionally available only on
    :class:`AsyncPineAI`; this client does not create a hidden loop or thread.
    """

    def __init__(
        self,
        access_token: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        *,
        api_base_path: str = DEFAULT_API_BASE_PATH,
        http_client: httpx.Client | None = None,
        http_transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.http = SyncHttpClient(
            base_url=base_url,
            token=access_token,
            api_base_path=api_base_path,
            client=http_client,
            transport=http_transport,
        )
        self.auth = SyncAuth(self.http)
        self.sessions = SyncSessionsAPI(self.http)

    def __enter__(self) -> "PineAI":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the SDK-owned synchronous HTTP client."""
        self.http.close()

    session_url = staticmethod(AsyncPineAI.session_url)
