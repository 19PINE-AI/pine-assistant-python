"""
Chat engine — send messages and yield events via async generator.

Every event reaches the caller, whether or not the SDK recognises it. Only the
supported surface drives control flow: what terminates a turn, what counts as a
response, and what gets deduplicated are all decided from scope events.
"""

import asyncio
from collections.abc import AsyncGenerator, Callable, Coroutine
from typing import Any

from pine_assistant.models.events import C2SEvent, S2CEvent
from pine_assistant.models.session import ACCEPTING_INPUT
from pine_assistant.transport.socketio import SocketIOManager

TERMINAL_STATES = {"task_finished", "task_cancelled", "task_stale"}
DEFAULT_IDLE_TIMEOUT_S = 120.0
DEFAULT_RESPONSE_IDLE_TIMEOUT_S = 2.0

# Joining always rebuilds from history rather than resuming a cursor: the
# incremental mechanism is gated and may be unavailable, an unconditional
# rebuild is not.
FULL_REBUILD_REVISION = "0"

# An agent response, for the purpose of deciding a turn has begun. Scope events
# only — a turn must not hinge on an event we do not maintain.
SUBSTANTIVE_EVENTS = frozenset({
    S2CEvent.SESSION_TEXT,
    S2CEvent.SESSION_TEXT_PART,
    S2CEvent.SESSION_RICH_CONTENT,
    S2CEvent.SESSION_FORM_TO_USER,
    S2CEvent.SESSION_TASK_READY,
    S2CEvent.SESSION_TASK_FINISHED,
    S2CEvent.SESSION_TOOL_STATUS,
    S2CEvent.SESSION_RESTRICTION,
})


class ChatEvent:
    __slots__ = ("type", "session_id", "message_id", "data", "metadata", "event_id")

    def __init__(self, type: str, session_id: str, data: Any,
                 message_id: str | None = None, metadata: dict[str, Any] | None = None,
                 event_id: str | None = None):
        self.type = type
        self.session_id = session_id
        self.message_id = message_id
        self.data = data
        self.metadata = metadata
        self.event_id = event_id

    def __repr__(self) -> str:
        return f"ChatEvent(type={self.type!r}, session_id={self.session_id!r})"


def event_from_envelope(event_type: str, raw: dict[str, Any], session_id: str) -> ChatEvent:
    """Build a ChatEvent from a raw envelope, carrying the payload through as-is."""
    payload = raw.get("payload") or {}
    metadata = raw.get("metadata")
    return ChatEvent(
        type=event_type,
        session_id=session_id,
        message_id=payload.get("message_id"),
        data=payload.get("data"),
        metadata=metadata,
        event_id=(metadata or {}).get("event_id") if isinstance(metadata, dict) else None,
    )


class Deduplicator:
    """Suppresses events already seen.

    Keyed on the event identifier together with the message type — never the
    identifier alone, which collides across types. An event with no identifier
    cannot be keyed and is always passed through.
    """

    def __init__(self) -> None:
        self._seen: set[tuple[str, str]] = set()

    def is_duplicate(self, event: ChatEvent) -> bool:
        if not event.event_id:
            return False
        key = (event.event_id, event.type)
        if key in self._seen:
            return True
        self._seen.add(key)
        return False


class ChatEngine:
    def __init__(
        self,
        sio: SocketIOManager,
        check_session_state: Callable[[str], Coroutine[Any, Any, dict[str, Any]]] | None = None,
        idle_timeout_s: float = DEFAULT_IDLE_TIMEOUT_S,
        response_idle_timeout_s: float = DEFAULT_RESPONSE_IDLE_TIMEOUT_S,
    ):
        self._sio = sio
        self._check_session_state = check_session_state
        self._idle_timeout_s = idle_timeout_s
        self._response_idle_timeout_s = response_idle_timeout_s

    async def join_session(self, session_id: str) -> dict[str, Any]:
        """Enter a session and retrieve its current state.

        `since_revision` is always "0" and the incremental-synchronization
        fields in the response are ignored; callers rebuild from history.
        """
        return await self._sio.emit_and_wait(
            C2SEvent.SESSION_JOIN,
            {"since_revision": FULL_REBUILD_REVISION},
            session_id=session_id,
        )

    def leave_session(self, session_id: str) -> None:
        """Leave a session room.

        Room management, outside the supported protocol scope; a connection
        tracks one session, so disconnecting is the alternative.
        """
        self._sio.emit("session:leave", None, session_id)

    @staticmethod
    def _build_message_data(
        content: str,
        attachments: list[dict[str, Any]] | None = None,
        referenced_sessions: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        from datetime import datetime
        return {
            "content": content,
            "attachments": attachments or [],
            "referenced_sessions": referenced_sessions or [],
            "client_now_date": datetime.now().isoformat(),
        }

    async def chat(
        self,
        session_id: str,
        content: str,
        *,
        attachments: list[dict[str, Any]] | None = None,
        referenced_sessions: list[dict[str, str]] | None = None,
    ) -> AsyncGenerator[ChatEvent, None]:
        """Send a message and yield the events that follow."""
        self._sio.emit(
            C2SEvent.SESSION_MESSAGE,
            self._build_message_data(content, attachments, referenced_sessions),
            session_id,
        )
        async for event in self._listen(session_id, _skip_state_precheck=True):
            yield event

    def send_message(
        self,
        session_id: str,
        content: str,
        *,
        attachments: list[dict[str, Any]] | None = None,
        referenced_sessions: list[dict[str, str]] | None = None,
    ) -> None:
        """Fire-and-forget message send (no event listening)."""
        self._sio.emit(
            C2SEvent.SESSION_MESSAGE,
            self._build_message_data(content, attachments, referenced_sessions),
            session_id,
        )

    async def _listen(
        self, session_id: str, *, _skip_state_precheck: bool = False,
    ) -> AsyncGenerator[ChatEvent, None]:
        """Yield events for a session until the turn ends."""
        if not _skip_state_precheck and self._check_session_state:
            try:
                session = await self._check_session_state(session_id)
                if session.get("state") in TERMINAL_STATES:
                    yield ChatEvent(type=S2CEvent.SESSION_STATE, session_id=session_id,
                                    data={"content": session["state"]})
                    return
            except Exception:
                pass  # best effort

        queue: asyncio.Queue[ChatEvent | None] = asyncio.Queue()
        dedup = Deduplicator()
        done = False
        received_agent_response = False

        def handler(event: str, raw: dict[str, Any]) -> None:
            nonlocal done, received_agent_response
            payload = raw.get("payload") or {}
            p_session_id = payload.get("session_id")
            if p_session_id and p_session_id != session_id:
                return

            chat_event = event_from_envelope(event, raw, session_id)
            if dedup.is_duplicate(chat_event):
                return
            queue.put_nowait(chat_event)

            if event in SUBSTANTIVE_EVENTS:
                received_agent_response = True
            data = payload.get("data")
            if (event == S2CEvent.SESSION_INPUT_STATE and isinstance(data, dict)
                    and data.get("content") == ACCEPTING_INPUT and received_agent_response):
                done = True
                queue.put_nowait(None)
            if (event == S2CEvent.SESSION_STATE and isinstance(data, dict)
                    and data.get("content", "") in TERMINAL_STATES):
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
                                yield ChatEvent(type=S2CEvent.SESSION_STATE, session_id=session_id,
                                                data={"content": session["state"]})
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
        """Answer a `session:form_to_user` request.

        Never submit values the user did not supply: the format defines no
        representation for refusal, an empty submission is indistinguishable
        from empty answers, and the agent may act on it. Sending nothing is
        safe.
        """
        self._sio.emit(C2SEvent.SESSION_FORM_TO_USER, {"content": form_data}, session_id, message_id)
