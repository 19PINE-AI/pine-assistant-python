"""
Socket.IO event types inside the supported protocol scope.

Only the supported surface is modelled here. The server emits many more events;
they reach callers verbatim through the same stream and carry no compatibility
guarantee — see `pine_assistant.is_supported_event`.
"""

import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        pass


class C2SEvent(StrEnum):
    """Client-to-server events inside the supported scope."""
    SESSION_JOIN = "session:join"
    SESSION_HISTORY = "session:history"
    SESSION_MESSAGE = "session:message"
    SESSION_FORM_TO_USER = "session:form_to_user"


class S2CEvent(StrEnum):
    """Server-to-client events inside the supported scope."""

    # Connection and session
    READY = "ready"
    SESSION_JOIN = "session:join"
    SESSION_HISTORY = "session:history"
    SESSION_ERROR = "session:error"

    # Conversation
    SESSION_MESSAGE = "session:message"
    SESSION_TEXT = "session:text"
    SESSION_TEXT_PART = "session:text_part"
    SESSION_RICH_CONTENT = "session:rich_content"
    SESSION_LLM_THINKING = "session:llm_thinking"

    # Session state
    SESSION_STATE = "session:state"
    SESSION_INPUT_STATE = "session:input_state"
    SESSION_MESSAGE_STATUS = "session:message_status"
    SESSION_REQUIRED_ACTION = "session:required_action"
    SESSION_UPDATE_TITLE = "session:update_title"
    SESSION_RESTRICTION = "session:restriction"

    # Interaction
    SESSION_FORM_TO_USER = "session:form_to_user"

    # Task and result
    SESSION_TASK_READY = "session:task_ready"
    SESSION_TASK_FINISHED = "session:task_finished"
    SESSION_TOOL_STATUS = "session:tool_status"


SUPPORTED_EVENTS = frozenset(e.value for e in S2CEvent) | frozenset(e.value for e in C2SEvent)


def is_supported_event(event_type: str) -> bool:
    """Whether an event carries the compatibility guarantee.

    An unsupported event is still delivered — the SDK never drops one — but it
    may be renamed, have its payload changed, or cease to be emitted without
    notice. Tolerating one is required; depending on one is not.
    """
    return event_type in SUPPORTED_EVENTS
