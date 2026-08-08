"""
pine-assistant — Pine AI SDK for Python.

Let Pine AI handle your digital chores.
Socket.IO + REST client for the Pine AI backend.

The SDK models the supported protocol scope. Events outside it are delivered
verbatim but carry no compatibility guarantee: tolerate them, do not depend on
them. `is_supported_event` tells the two apart.
"""

from pine_assistant.auth import Auth
from pine_assistant.chat import ChatEvent
from pine_assistant.client import AsyncPineAI, PineAI
from pine_assistant.errors import AuthError, ConnectionError, PineAIError, SessionError
from pine_assistant.models.events import (
    SUPPORTED_EVENTS,
    C2SEvent,
    S2CEvent,
    is_supported_event,
)
from pine_assistant.models.session import InputState, InputStateCode
from pine_assistant.sessions import SessionsAPI

__version__ = "0.4.0"
__all__ = [
    "PineAI",
    "AsyncPineAI",
    "Auth",
    "SessionsAPI",
    "ChatEvent",
    "PineAIError",
    "AuthError",
    "SessionError",
    "ConnectionError",
    "C2SEvent",
    "S2CEvent",
    "SUPPORTED_EVENTS",
    "is_supported_event",
    "InputState",
    "InputStateCode",
]
