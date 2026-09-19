"""
pine-assistant — Pine AI SDK for Python.

Let Pine AI handle your digital chores.
Socket.IO + REST client for the Pine AI backend.

The SDK models the supported protocol scope. Events outside it are delivered
verbatim but carry no compatibility guarantee: tolerate them, do not depend on
them. `is_supported_event` tells the two apart.
"""

from pine_assistant.auth import Auth, SyncAuth
from pine_assistant.chat import ChatEvent
from pine_assistant.client import AsyncPineAI, PineAI
from pine_assistant.errors import AuthError, ConnectionError, PineAIError, SessionError
from pine_assistant.models.auth import AuthIdentity, AuthTicket, RedeemedTicket
from pine_assistant.models.events import (
    SUPPORTED_EVENTS,
    C2SEvent,
    S2CEvent,
    is_supported_event,
)
from pine_assistant.models.session import (
    SessionInfo,
    SessionListResponse,
    SessionMessageStatus,
    SessionOutcome,
    SessionOutcomeRating,
    SessionOutcomesPage,
)
from pine_assistant.sessions import SessionsAPI, SyncSessionsAPI

__version__ = "0.5.0rc1"
__all__ = [
    "PineAI",
    "AsyncPineAI",
    "Auth",
    "SyncAuth",
    "SessionsAPI",
    "SyncSessionsAPI",
    "AuthIdentity",
    "AuthTicket",
    "RedeemedTicket",
    "SessionInfo",
    "SessionListResponse",
    "SessionMessageStatus",
    "SessionOutcome",
    "SessionOutcomeRating",
    "SessionOutcomesPage",
    "ChatEvent",
    "PineAIError",
    "AuthError",
    "SessionError",
    "ConnectionError",
    "C2SEvent",
    "S2CEvent",
    "SUPPORTED_EVENTS",
    "is_supported_event",
]
