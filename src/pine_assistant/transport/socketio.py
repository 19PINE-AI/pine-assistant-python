"""
Socket.IO connection manager — spec sections 3.1, 5.1.2.

Connection: wss://{baseUrl}/api/v2/socket.io/ with auth={token}.
Waits for `ready` event before resolving connect().
"""

import asyncio
import uuid
from typing import Any, Callable, Optional

import socketio

SOCKETIO_PATH = "/api/v2/socket.io/"


class SocketIOManager:
    def __init__(
        self,
        base_url: str,
        token: str,
        user_id: str,
        device_id: Optional[str] = None,
        transports: Optional[list[str]] = None,
        ready_timeout: float = 15.0,
    ):
        self._base_url = base_url
        self._token = token
        self._user_id = user_id
        self._device_id = device_id or str(uuid.uuid4())
        self._transports = transports or ["websocket"]
        self._ready_timeout = ready_timeout
        self._sio: Optional[socketio.AsyncClient] = None
        self._connected = False
        self._event_handlers: list[Callable[[str, dict[str, Any]], None]] = []
        self._joined_sessions: set[str] = set()

    @property
    def connected(self) -> bool:
        return self._connected and self._sio is not None and self._sio.connected

    @property
    def device_id(self) -> str:
        return self._device_id

    def add_event_handler(self, handler: Callable[[str, dict[str, Any]], None]) -> Callable[[], None]:
        """Add an event handler. Returns a cleanup function. Supports multiple concurrent handlers."""
        self._event_handlers.append(handler)
        def remove() -> None:
            try:
                self._event_handlers.remove(handler)
            except ValueError:
                pass
        return remove

    def on_event(self, handler: Optional[Callable[[str, dict[str, Any]], None]]) -> None:
        """Set a single event handler (replaces all). Use add_event_handler() for multi-session."""
        self._event_handlers.clear()
        if handler is not None:
            self._event_handlers.append(handler)

    async def connect(self) -> None:
        """Connect to Pine backend, wait for `ready` event — spec 5.1.2."""
        if self._sio and self._sio.connected:
            return

        self._sio = socketio.AsyncClient()
        ready_event = asyncio.Event()

        @self._sio.event
        async def connect() -> None:
            pass

        @self._sio.on("ready")
        async def on_ready(*_args: Any) -> None:
            self._connected = True
            if not ready_event.is_set():
                ready_event.set()
            else:
                # Reconnection: re-join all previously joined sessions
                for sid in list(self._joined_sessions):
                    self.emit("session:join", None, sid)

        @self._sio.on("*")
        async def on_any(event: str, data: Any) -> None:
            if event in ("connect", "disconnect", "connect_error", "ready"):
                return
            if self._event_handlers and isinstance(data, dict):
                for handler in list(self._event_handlers):
                    handler(event, data)

        @self._sio.event
        async def disconnect(_reason: str = "") -> None:
            self._connected = False

        await self._sio.connect(
            self._base_url,
            auth={"token": self._token},
            transports=self._transports,
            socketio_path=SOCKETIO_PATH,
        )

        try:
            await asyncio.wait_for(ready_event.wait(), timeout=self._ready_timeout)
        except asyncio.TimeoutError:
            await self._sio.disconnect()
            raise TimeoutError(f"Timed out waiting for 'ready' event after {self._ready_timeout}s")

    def emit(
        self,
        event_type: str,
        data: Any,
        session_id: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> None:
        """Emit a typed event with envelope wrapping.

        Schedules the async emit on the running event loop. Errors are logged
        rather than silently swallowed.
        """
        if not self._sio or not self._sio.connected:
            raise RuntimeError("Socket.IO not connected")
        # Track join/leave for reconnection
        if event_type == "session:join" and session_id:
            self._joined_sessions.add(session_id)
        if event_type == "session:leave" and session_id:
            self._joined_sessions.discard(session_id)
        from pine_assistant.transport.envelope import build_envelope
        envelope = build_envelope(
            event_type, data,
            user_id=self._user_id,
            device_id=self._device_id,
            session_id=session_id,
            message_id=message_id,
        )

        async def _do_emit() -> None:
            try:
                await self._sio.emit(event_type, envelope)  # type: ignore[union-attr]
            except Exception as e:
                import logging
                logging.getLogger("pine_assistant.transport.socketio").error(f"Emit failed for {event_type}: {e}")

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_do_emit())
        except RuntimeError:
            asyncio.ensure_future(_do_emit())

    async def emit_and_wait(
        self,
        event_type: str,
        data: Any,
        session_id: Optional[str] = None,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        """Emit and wait for a response event with matching session_id."""
        if not self._sio or not self._sio.connected:
            raise RuntimeError("Socket.IO not connected")
        from pine_assistant.transport.envelope import build_envelope
        request_id = str(uuid.uuid4())
        envelope = build_envelope(
            event_type, data,
            user_id=self._user_id,
            device_id=self._device_id,
            session_id=session_id,
            request_id=request_id,
        )

        result_event = asyncio.Event()
        result_data: dict[str, Any] = {}

        def response_handler(evt: str, raw: dict[str, Any]) -> None:
            if evt == event_type:
                payload = raw.get("payload", {})
                meta = raw.get("metadata", {})
                match_by_request = meta.get("request_id") == request_id
                match_by_session = (
                    session_id and
                    payload.get("session_id") == session_id and
                    meta.get("source", {}).get("role") != "user"
                )
                if match_by_request or match_by_session:
                    result_data.update(payload.get("data") or {})
                    result_event.set()

        remove_handler = self.add_event_handler(response_handler)
        await self._sio.emit(event_type, envelope)

        try:
            await asyncio.wait_for(result_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Timeout waiting for {event_type} response")
        finally:
            remove_handler()

        return result_data

    async def disconnect(self) -> None:
        self._connected = False
        if self._sio:
            await self._sio.disconnect()
            self._sio = None
