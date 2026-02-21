"""
Chat engine — send messages and yield events via async generator.

All events are dispatched immediately as they arrive from the server.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncGenerator, Callable, Coroutine, Optional

from pine_assistant.models.events import C2SEvent, S2CEvent
from pine_assistant.transport.socketio import SocketIOManager

TERMINAL_STATES = {"task_finished", "task_cancelled", "task_stale"}
DEFAULT_IDLE_TIMEOUT_S = 120.0
DEFAULT_RESPONSE_IDLE_TIMEOUT_S = 2.0

SUBSTANTIVE_EVENTS = {
    S2CEvent.SESSION_TEXT, S2CEvent.SESSION_TEXT_PART,
    S2CEvent.SESSION_FORM_TO_USER,
    S2CEvent.SESSION_ASK_FOR_LOCATION, S2CEvent.SESSION_TASK_READY,
    S2CEvent.SESSION_TASK_FINISHED, S2CEvent.SESSION_INTERACTIVE_AUTH_CONFIRMATION,
    S2CEvent.SESSION_THREE_WAY_CALL, S2CEvent.SESSION_REWARD,
}


class ChatEvent:
    __slots__ = ("type", "session_id", "message_id", "data", "metadata")

    def __init__(self, type: str, session_id: str, data: Any,
                 message_id: Optional[str] = None, metadata: Optional[dict[str, Any]] = None):
        self.type = type
        self.session_id = session_id
        self.message_id = message_id
        self.data = data
        self.metadata = metadata

    def __repr__(self) -> str:
        return f"ChatEvent(type={self.type!r}, session_id={self.session_id!r})"


class ChatEngine:
    def __init__(
        self,
        sio: SocketIOManager,
        check_session_state: Optional[Callable[[str], Coroutine[Any, Any, dict[str, Any]]]] = None,
        idle_timeout_s: float = DEFAULT_IDLE_TIMEOUT_S,
        response_idle_timeout_s: float = DEFAULT_RESPONSE_IDLE_TIMEOUT_S,
    ):
        self._sio = sio
        self._check_session_state = check_session_state
        self._idle_timeout_s = idle_timeout_s
        self._response_idle_timeout_s = response_idle_timeout_s

    async def join_session(self, session_id: str) -> dict[str, Any]:
        """Join a session room — spec 5.1.1.
        Production handler reads payload.session_id (set by envelope builder).
        """
        return await self._sio.emit_and_wait(
            C2SEvent.SESSION_JOIN,
            None,  # payload.data is not used for join
            session_id=session_id,
        )

    def leave_session(self, session_id: str) -> None:
        """Leave a session room."""
        self._sio.emit(C2SEvent.SESSION_LEAVE, None, session_id)

    @staticmethod
    def _build_message_data(
        content: str,
        attachments: Optional[list[dict[str, Any]]] = None,
        referenced_sessions: Optional[list[dict[str, str]]] = None,
        action: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Build the session:message payload per spec 5.1.1."""
        from datetime import datetime
        data: dict[str, Any] = {
            "content": content,
            "attachments": attachments or [],
            "referenced_sessions": referenced_sessions or [],
            "client_now_date": datetime.now().isoformat(),
        }
        if action is not None:
            data["action"] = action
        return data

    async def chat(
        self,
        session_id: str,
        content: str,
        *,
        attachments: Optional[list[dict[str, Any]]] = None,
        referenced_sessions: Optional[list[dict[str, str]]] = None,
        action: Optional[dict[str, Any]] = None,
    ) -> AsyncGenerator[ChatEvent, None]:
        """Send a message and yield events with stream buffering.
        Production handler reads payload.data as {content, attachments, ...}.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=5)
        self._sio.emit(
            C2SEvent.SESSION_MESSAGE,
            self._build_message_data(content, attachments, referenced_sessions, action),
            session_id,
        )
        async for event in self._listen(session_id, _skip_state_precheck=True):
            if self._is_stale_event(event, cutoff):
                continue
            yield event

    @staticmethod
    def _is_stale_event(event: "ChatEvent", cutoff: datetime) -> bool:
        """Return True if the event's metadata timestamp predates the cutoff."""
        meta = event.metadata
        if not isinstance(meta, dict):
            return False
        ts_str = meta.get("timestamp")
        if not ts_str:
            return False
        try:
            event_ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            return event_ts < cutoff
        except (ValueError, TypeError):
            return False

    def send_message(
        self,
        session_id: str,
        content: str,
        *,
        attachments: Optional[list[dict[str, Any]]] = None,
        referenced_sessions: Optional[list[dict[str, str]]] = None,
        action: Optional[dict[str, Any]] = None,
    ) -> None:
        """Fire-and-forget message send (no event listening)."""
        self._sio.emit(
            C2SEvent.SESSION_MESSAGE,
            self._build_message_data(content, attachments, referenced_sessions, action),
            session_id,
        )

    async def _listen(
        self, session_id: str, *, _skip_state_precheck: bool = False,
    ) -> AsyncGenerator[ChatEvent, None]:
        """Listen for events — all events dispatched immediately."""
        if not _skip_state_precheck and self._check_session_state:
            try:
                session = await self._check_session_state(session_id)
                if session.get("state") in TERMINAL_STATES:
                    yield ChatEvent(type=S2CEvent.SESSION_STATE, session_id=session_id, data={"content": session["state"]})
                    return
            except Exception:
                pass  # best effort

        queue: asyncio.Queue[Optional[ChatEvent]] = asyncio.Queue()
        done = False
        received_agent_response = False

        def handler(event: str, raw: dict[str, Any]) -> None:
            nonlocal done, received_agent_response
            payload = raw.get("payload", {})
            p_session_id = payload.get("session_id")
            if p_session_id and p_session_id != session_id:
                return

            queue.put_nowait(ChatEvent(
                type=event, session_id=session_id,
                message_id=payload.get("message_id"),
                data=payload.get("data"),
                metadata=raw.get("metadata"),
            ))
            if event in SUBSTANTIVE_EVENTS:
                received_agent_response = True
            if event == S2CEvent.SESSION_INPUT_STATE and isinstance(payload.get("data"), dict):
                if payload["data"].get("content") == "waiting_input" and received_agent_response:
                    done = True
                    queue.put_nowait(None)
            if event == S2CEvent.SESSION_STATE and isinstance(payload.get("data"), dict):
                state = payload["data"].get("content", "")
                if state in TERMINAL_STATES:
                    done = True
                    queue.put_nowait(None)

        remove_handler = self._sio.add_event_handler(handler)

        try:
            while not done:
                timeout = self._response_idle_timeout_s if received_agent_response else self._idle_timeout_s
                try:
                    evt = await asyncio.wait_for(queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    if received_agent_response:
                        break
                    if self._check_session_state:
                        try:
                            session = await self._check_session_state(session_id)
                            if session.get("state") in TERMINAL_STATES:
                                yield ChatEvent(type=S2CEvent.SESSION_STATE, session_id=session_id, data={"content": session["state"]})
                                break
                        except Exception:
                            pass
                    continue
                if evt is None:
                    break
                yield evt
            while not queue.empty():
                evt = queue.get_nowait()
                if evt is not None:
                    yield evt
        finally:
            remove_handler()

    def send_form_response(self, session_id: str, message_id: str, form_data: dict[str, Any]) -> None:
        """Production handler reads payload.data.content as form key-value pairs."""
        self._sio.emit(C2SEvent.SESSION_FORM_TO_USER, {"content": form_data}, session_id, message_id)

    def send_auth_confirmation(self, session_id: str, message_id: str, data: dict[str, Any]) -> None:
        """Production handler reads payload.data.content as confirmation data."""
        self._sio.emit(C2SEvent.SESSION_INTERACTIVE_AUTH_CONFIRMATION, {"content": data}, session_id, message_id)

    def send_location_response(self, session_id: str, message_id: str, latitude: str, longitude: str) -> None:
        """Production handler reads payload.data.content as {latitude, longitude}."""
        self._sio.emit(C2SEvent.SESSION_ASK_FOR_LOCATION, {"content": {"latitude": latitude, "longitude": longitude}}, session_id, message_id)

    def send_location_selection(self, session_id: str, message_id: str, places: list[dict[str, Any]]) -> None:
        """Production handler reads payload.data.list as place objects."""
        self._sio.emit(C2SEvent.SESSION_LOCATION_SELECTION, {"list": places}, session_id, message_id)
