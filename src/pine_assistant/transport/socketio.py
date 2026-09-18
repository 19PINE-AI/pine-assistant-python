"""
Socket.IO connection manager — spec sections 3.1, 5.1.2.

Connection: wss://{baseUrl}/api/v2/socket.io/ with auth={token}.
Waits for `ready` event before resolving connect().
"""

import asyncio
import contextlib
import uuid
from collections.abc import Callable
from typing import Any

import socketio

from pine_assistant.errors import ConnectionError as PineConnectionError
from pine_assistant.models.form import FormSubmissionResult

SOCKETIO_PATH = "/api/v2/socket.io/"


def _format_connect_error(data: Any) -> str:
    """Render a server-supplied connect_error payload into a human-readable string."""
    if data is None:
        return "no payload"
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        msg = data.get("message") or data.get("error") or data.get("reason")
        if msg:
            return str(msg)
    return repr(data)


class SocketIOManager:
    def __init__(
        self,
        base_url: str,
        token: str,
        user_id: str,
        device_id: str | None = None,
        transports: list[str] | None = None,
        ready_timeout: float = 15.0,
    ):
        self._base_url = base_url
        self._token = token
        self._user_id = user_id
        self._device_id = device_id or str(uuid.uuid4())
        self._transports = transports or ["websocket"]
        self._ready_timeout = ready_timeout
        self._sio: socketio.AsyncClient | None = None
        self._connected = False
        self._event_handlers: list[Callable[[str, dict[str, Any]], None]] = []
        self._reconnect_handlers: list[Callable[[], None]] = []
        self._joined_sessions: set[str] = set()
        self._disconnect_event = asyncio.Event()
        self._connection_generation = 0
        self._form_submission_attempts: dict[tuple[str, str], int] = {}
        self._form_submission_pending_keys: set[tuple[str, str]] = set()

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
            with contextlib.suppress(ValueError):
                self._event_handlers.remove(handler)
        return remove

    def add_reconnect_handler(self, handler: Callable[[], None]) -> Callable[[], None]:
        """Register a callback fired after a reconnect has re-joined its sessions.

        A reconnect invalidates whatever the caller had accumulated: delivery
        may have stopped before the socket noticed, so state has to be rebuilt
        rather than resumed.
        """
        self._reconnect_handlers.append(handler)

        def remove() -> None:
            with contextlib.suppress(ValueError):
                self._reconnect_handlers.remove(handler)
        return remove

    async def connect(self) -> None:
        """Connect to Pine backend, wait for `ready` event — spec 5.1.2."""
        if self._sio and self._sio.connected:
            return

        disconnect_event = asyncio.Event()
        self._disconnect_event = disconnect_event
        self._sio = socketio.AsyncClient()
        ready_event = asyncio.Event()
        connect_error_data: dict[str, Any] = {}

        @self._sio.event
        async def connect() -> None:
            pass

        @self._sio.event
        async def connect_error(data: Any) -> None:
            connect_error_data["payload"] = data

        @self._sio.on("ready")
        async def on_ready(*_args: Any) -> None:
            self._connected = True
            if not ready_event.is_set():
                ready_event.set()
            else:
                self._connection_generation += 1
                self._disconnect_event = asyncio.Event()
                self._form_submission_attempts = {
                    key: generation
                    for key, generation in self._form_submission_attempts.items()
                    if key in self._form_submission_pending_keys
                }
                # Reconnection: re-join every previously joined session. State
                # is rebuilt from history rather than resumed from a cursor, so
                # the join carries since_revision "0" here too.
                for sid in list(self._joined_sessions):
                    self.emit("session:join", {"since_revision": "0"}, sid)
                for on_reconnect in list(self._reconnect_handlers):
                    on_reconnect()

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
            self._disconnect_event.set()

        try:
            await self._sio.connect(
                self._base_url,
                auth={"token": self._token},
                transports=self._transports,
                socketio_path=SOCKETIO_PATH,
                wait_timeout=self._ready_timeout,
            )
        except Exception:
            with contextlib.suppress(Exception):
                await self._sio.disconnect()
            self._sio = None
            self._connected = False
            self._disconnect_event.set()
            # python-socketio raises the generic "One or more namespaces
            # failed to connect" when the server replies with connect_error.
            # Surface the actual server-supplied reason if we captured one.
            if "payload" in connect_error_data:
                reason = _format_connect_error(connect_error_data["payload"])
                raise PineConnectionError(
                    f"Socket.IO connect rejected by server: {reason}"
                ) from None
            raise PineConnectionError("Socket.IO connect failed") from None

        try:
            await asyncio.wait_for(ready_event.wait(), timeout=self._ready_timeout)
        except asyncio.TimeoutError:
            with contextlib.suppress(Exception):
                await self._sio.disconnect()
            self._sio = None
            self._connected = False
            self._disconnect_event.set()
            # The Pine backend accepts the WebSocket but only emits 'ready' after
            # its own auth check. A timeout here almost always means the token or
            # user_id is rejected — surface that hint instead of a generic timeout.
            raise PineConnectionError(
                f"Socket.IO connected but no 'ready' event after {self._ready_timeout}s. "
                "This usually means access_token or user_id is invalid/expired — re-run the auth flow."
            ) from None
        self._connection_generation += 1
        # An earlier connection may still be unwinding after its disconnect.
        # Preserve those pending keys until their waiter has cleaned up, but a
        # completed attempt is scoped only to the connection that sent it.
        self._form_submission_attempts = {
            key: generation
            for key, generation in self._form_submission_attempts.items()
            if key in self._form_submission_pending_keys
        }

    def _track_membership(self, event_type: str, session_id: str | None) -> None:
        """Remember which sessions to re-join after a reconnect.

        Both emit paths run through here: joining goes out via emit_and_wait,
        so tracking only on the fire-and-forget path would leave every joined
        session unrestored after a drop.
        """
        if not session_id:
            return
        if event_type == "session:join":
            self._joined_sessions.add(session_id)
        elif event_type == "session:leave":
            self._joined_sessions.discard(session_id)

    def emit(
        self,
        event_type: str,
        data: Any,
        session_id: str | None = None,
        message_id: str | None = None,
    ) -> None:
        """Emit a typed event with envelope wrapping.

        Schedules the async emit on the running event loop. Errors are logged
        rather than silently swallowed.
        """
        if not self._sio or not self._sio.connected:
            raise RuntimeError("Socket.IO not connected")
        self._track_membership(event_type, session_id)
        from pine_assistant.transport.envelope import build_envelope
        envelope = build_envelope(
            event_type, data,
            user_id=self._user_id,
            device_id=self._device_id,
            session_id=session_id,
            message_id=message_id,
        )

        async def _do_emit() -> None:
            # The socket may be torn down between scheduling and execution —
            # e.g. user calls leave_session() then disconnect() back-to-back.
            # In that case the emit can't possibly succeed; skip silently
            # instead of logging a misleading "/ is not a connected namespace".
            if not self._sio or not self._sio.connected:
                return
            try:
                await self._sio.emit(event_type, envelope)  # type: ignore[union-attr]
            except Exception:
                import logging
                logging.getLogger("pine_assistant.transport.socketio").error("Emit failed for %s", event_type)

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_do_emit())
        except RuntimeError:
            asyncio.ensure_future(_do_emit())

    async def emit_and_wait(
        self,
        event_type: str,
        data: Any,
        session_id: str | None = None,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        """Emit and wait for the response with the exact request correlation.

        A session match alone is insufficient: concurrent requests in one room
        receive the same event name. The backend echoes the client request ID.
        """
        if not self._sio or not self._sio.connected:
            raise RuntimeError("Socket.IO not connected")
        self._track_membership(event_type, session_id)
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
        error_event = asyncio.Event()

        def response_handler(evt: str, raw: dict[str, Any]) -> None:
            payload = raw.get("payload")
            meta = raw.get("metadata")
            if not isinstance(payload, dict) or not isinstance(meta, dict):
                return
            if payload.get("session_id") != session_id or meta.get("request_id") != request_id:
                return
            if evt == event_type:
                data = payload.get("data")
                if isinstance(data, dict):
                    result_data.update(data)
                    result_event.set()
            elif evt == "session:error":
                error_event.set()

        remove_handler = self.add_event_handler(response_handler)
        waiters: list[asyncio.Task[bool]] = []
        try:
            await self._sio.emit(event_type, envelope)
            result_wait = asyncio.create_task(result_event.wait())
            error_wait = asyncio.create_task(error_event.wait())
            disconnect_wait = asyncio.create_task(self._disconnect_event.wait())
            waiters = [result_wait, error_wait, disconnect_wait]
            done, _ = await asyncio.wait(
                waiters, timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                raise TimeoutError(f"Timeout waiting for {event_type} response")
            if error_wait in done:
                raise RuntimeError(f"Server rejected {event_type}")
            if disconnect_wait in done:
                raise RuntimeError("Socket.IO disconnected while waiting for a response")
        except asyncio.TimeoutError:
            raise TimeoutError(f"Timeout waiting for {event_type} response") from None
        finally:
            for waiter in waiters:
                if not waiter.done():
                    waiter.cancel()
            if waiters:
                await asyncio.gather(*waiters, return_exceptions=True)
            remove_handler()

        return result_data

    async def emit_form_response(
        self,
        *,
        session_id: str,
        original_message_id: str,
        original_request_id: str,
        content: dict[str, Any],
        timeout: float,
    ) -> FormSubmissionResult:
        """Submit a verified form response and observe its durable outcome.

        Socket.IO ACKs only confirm transport handling. The backend instead
        directly emits a matching ``session:message_status`` whose request ID
        is the original form request and whose message ID is the persisted
        reply. It never retries: after a deadline callers get only what was
        observed.
        """
        if not self._sio or not self._sio.connected:
            raise RuntimeError("Socket.IO not connected")
        submission_key = (session_id, original_request_id)
        attempted_generation = self._form_submission_attempts.get(submission_key)
        if (
            submission_key in self._form_submission_pending_keys
            or attempted_generation == self._connection_generation
        ):
            raise RuntimeError("a response for this form has already been submitted on this connection")
        self._form_submission_attempts[submission_key] = self._connection_generation
        self._form_submission_pending_keys.add(submission_key)

        from pine_assistant.transport.envelope import build_envelope

        envelope = build_envelope(
            "session:form_to_user", {"content": content},
            user_id=self._user_id, device_id=self._device_id,
            session_id=session_id, message_id=original_message_id,
            request_id=original_request_id,
        )
        progress = asyncio.Event()
        persisted_message_id: str | None = None
        received_reason: str | None = None
        delivered = False
        terminal_failure = False
        disconnect_event = self._disconnect_event

        def response_handler(evt: str, raw: dict[str, Any]) -> None:
            nonlocal delivered, persisted_message_id, received_reason, terminal_failure
            payload = raw.get("payload")
            metadata = raw.get("metadata")
            if not isinstance(payload, dict) or payload.get("session_id") != session_id:
                return
            if evt == "session:message_status":
                data = payload.get("data")
                if not isinstance(data, dict) or data.get("request_id") != original_request_id:
                    return
                status = data.get("status")
                message_id = data.get("message_id")
                if status == "received" and isinstance(message_id, str) and message_id:
                    if persisted_message_id is not None and message_id != persisted_message_id:
                        return
                    persisted_message_id = message_id
                    reason = data.get("reason")
                    received_reason = reason if isinstance(reason, str) else None
                elif status == "delivered" and isinstance(message_id, str) and message_id:
                    if persisted_message_id is not None and message_id != persisted_message_id:
                        return
                    persisted_message_id = message_id
                    delivered = True
                else:
                    # A failed, unrecognised, or malformed receipt does not
                    # establish persistence. Do not surface its reason: it may
                    # have been composed from submitted form values.
                    if persisted_message_id is not None and message_id not in (None, persisted_message_id):
                        return
                    terminal_failure = True
                progress.set()
            elif (
                evt == "session:error"
                and isinstance(metadata, dict)
                and metadata.get("request_id") == original_request_id
            ):
                terminal_failure = True
                progress.set()

        remove_handler = self.add_event_handler(response_handler)
        waiters: list[asyncio.Task[bool]] = []
        try:
            await self._sio.emit("session:form_to_user", envelope)
            deadline = asyncio.get_running_loop().time() + timeout
            while not delivered and not terminal_failure and not disconnect_event.is_set():
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                progress.clear()
                if delivered or terminal_failure:
                    break
                receipt_wait = asyncio.create_task(progress.wait())
                connection_wait = asyncio.create_task(disconnect_event.wait())
                waiters = [receipt_wait, connection_wait]
                await asyncio.wait(waiters, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
                for waiter in waiters:
                    if not waiter.done():
                        waiter.cancel()
                await asyncio.gather(*waiters, return_exceptions=True)
                waiters = []
            if delivered:
                return FormSubmissionResult("delivered", persisted_message_id, original_request_id)
            if persisted_message_id is not None:
                return FormSubmissionResult("received", persisted_message_id, original_request_id, received_reason)
            return FormSubmissionResult("unknown", request_id=original_request_id)
        finally:
            for waiter in waiters:
                if not waiter.done():
                    waiter.cancel()
            if waiters:
                await asyncio.gather(*waiters, return_exceptions=True)
            remove_handler()
            self._form_submission_pending_keys.discard(submission_key)

    async def disconnect(self) -> None:
        self._connected = False
        self._disconnect_event.set()
        if self._sio:
            await self._sio.disconnect()
            self._sio = None
