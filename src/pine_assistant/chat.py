"""
Chat engine — send messages and yield complete events via async generator.

Stream buffering:
- Tier 1: session:text_part buffered until final: true
- Tier 2: Non-streaming events yield immediately
- Tier 3: session:work_log_part debounced (3s silence)
"""

import asyncio
import time
from typing import Any, AsyncGenerator, Callable, Coroutine, Optional

from pine_assistant.models.events import C2SEvent, S2CEvent
from pine_assistant.transport.socketio import SocketIOManager

TERMINAL_STATES = {"task_finished", "task_cancelled", "task_stale"}
DEFAULT_IDLE_TIMEOUT_S = 120.0

# Events that are buffered/debounced — NOT dispatched immediately
BUFFERED_EVENTS = {S2CEvent.SESSION_TEXT_PART, S2CEvent.SESSION_WORK_LOG_PART}

# Substantive response events — track for waiting_input termination gating
SUBSTANTIVE_EVENTS = {
    S2CEvent.SESSION_TEXT, S2CEvent.SESSION_FORM_TO_USER,
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


class TextPartBuffer:
    """Buffer text_part chunks per message_id, flush on final: true."""

    def __init__(self) -> None:
        self._parts: dict[str, list[str]] = {}

    def collect(self, message_id: str, content: str, final: bool) -> Optional[str]:
        if message_id not in self._parts:
            self._parts[message_id] = []
        if content:
            self._parts[message_id].append(content)
        if final:
            merged = "".join(self._parts.pop(message_id, []))
            return merged
        return None


class ChatEngine:
    def __init__(
        self,
        sio: SocketIOManager,
        check_session_state: Optional[Callable[[str], Coroutine[Any, Any, dict[str, Any]]]] = None,
        idle_timeout_s: float = DEFAULT_IDLE_TIMEOUT_S,
    ):
        self._sio = sio
        self._check_session_state = check_session_state
        self._idle_timeout_s = idle_timeout_s

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
        self._sio.emit(
            C2SEvent.SESSION_MESSAGE,
            self._build_message_data(content, attachments, referenced_sessions, action),
            session_id,
        )
        async for event in self._listen(session_id):
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
        """Fire-and-forget message send (no event listening)."""
        self._sio.emit(
            C2SEvent.SESSION_MESSAGE,
            self._build_message_data(content, attachments, referenced_sessions, action),
            session_id,
        )

    async def _listen(self, session_id: str) -> AsyncGenerator[ChatEvent, None]:
        """Listen for events with stream buffering."""
        # Check session state before entering loop — don't hang on completed sessions
        if self._check_session_state:
            try:
                session = await self._check_session_state(session_id)
                if session.get("state") in TERMINAL_STATES:
                    yield ChatEvent(type=S2CEvent.SESSION_STATE, session_id=session_id, data={"content": session["state"]})
                    return
            except Exception:
                pass  # best effort

        text_buffer = TextPartBuffer()
        queue: asyncio.Queue[Optional[ChatEvent]] = asyncio.Queue()
        done = False
        received_agent_response = False

        # Work log debounce state
        wl_timers: dict[str, asyncio.TimerHandle] = {}
        wl_buffers: dict[str, dict[str, Any]] = {}

        def flush_wl(step_id: str) -> None:
            buf = wl_buffers.pop(step_id, None)
            wl_timers.pop(step_id, None)
            if buf:
                queue.put_nowait(ChatEvent(
                    type=S2CEvent.SESSION_WORK_LOG_PART,
                    session_id=session_id,
                    data={"step_id": step_id, "text": buf.get("text", ""), "status": buf.get("status")},
                ))

        def handler(event: str, raw: dict[str, Any]) -> None:
            nonlocal done
            payload = raw.get("payload", {})
            p_session_id = payload.get("session_id")
            if p_session_id and p_session_id != session_id:
                return  # Not our session — other handlers will process it

            message_id = payload.get("message_id")
            data = payload.get("data")
            metadata = raw.get("metadata")

            # Tier 1: text_part
            if event == S2CEvent.SESSION_TEXT_PART:
                if isinstance(data, dict):
                    content = data.get("content", "")
                    final = data.get("final", False)
                    merged = text_buffer.collect(message_id or "unknown", content, final)
                    if merged is not None:
                        queue.put_nowait(ChatEvent(
                            type=S2CEvent.SESSION_TEXT, session_id=session_id,
                            message_id=message_id, data={"content": merged}, metadata=metadata,
                        ))
                return

            # Tier 3: work_log_part debounce
            if event == S2CEvent.SESSION_WORK_LOG_PART:
                if isinstance(data, dict):
                    step_id = data.get("step_id", "unknown")
                    existing = wl_buffers.get(step_id, {"text": ""})
                    existing["text"] = existing.get("text", "") + (data.get("text_delta", "") or "")
                    if data.get("status"):
                        existing["status"] = data["status"]
                    wl_buffers[step_id] = existing
                    old_timer = wl_timers.pop(step_id, None)
                    if old_timer:
                        old_timer.cancel()
                    loop = asyncio.get_running_loop()
                    wl_timers[step_id] = loop.call_later(3.0, flush_wl, step_id)
                return

            # All other events: dispatch immediately (pass-through for agent)
            nonlocal received_agent_response
            queue.put_nowait(ChatEvent(
                type=event, session_id=session_id,
                message_id=message_id, data=data, metadata=metadata,
            ))
            if event in SUBSTANTIVE_EVENTS:
                received_agent_response = True
            if event == S2CEvent.SESSION_INPUT_STATE and isinstance(data, dict):
                if data.get("content") == "waiting_input" and received_agent_response:
                    done = True
                    queue.put_nowait(None)
            if event == S2CEvent.SESSION_STATE and isinstance(data, dict):
                state = data.get("content", "")
                if state in TERMINAL_STATES:
                    done = True
                    queue.put_nowait(None)

        remove_handler = self._sio.add_event_handler(handler)

        try:
            while not done:
                try:
                    evt = await asyncio.wait_for(queue.get(), timeout=self._idle_timeout_s)
                except asyncio.TimeoutError:
                    # Idle timeout — check session state via REST
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
            for t in wl_timers.values():
                t.cancel()
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
