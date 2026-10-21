"""
pine-assistant — Pine AI SDK for Python.

Let Pine AI handle your digital chores.
Socket.IO + REST client for the Pine AI backend.

The SDK models the supported protocol scope. Events outside it are delivered
verbatim but carry no compatibility guarantee: tolerate them, do not depend on
them. `is_supported_event` tells the two apart.
"""

from pine_assistant.app_settings import AppSettingsAPI, SyncAppSettingsAPI
from pine_assistant.auth import Auth, SyncAuth
from pine_assistant.chat import ChatEvent
from pine_assistant.client import AsyncPineAI, PineAI
from pine_assistant.errors import AuthError, ConnectionError, PineAIError, PlatformError, SessionError
from pine_assistant.models.app_settings import AssistantIdentity, AssistantIdentityCatalog
from pine_assistant.models.auth import AuthIdentity, AuthTicket, RedeemedTicket
from pine_assistant.models.events import (
    SUPPORTED_EVENTS,
    C2SEvent,
    S2CEvent,
    is_supported_event,
)
from pine_assistant.models.platform import ManagedUser, ManagedUserAssistant
from pine_assistant.models.session import (
    CallerID,
    SessionConfig,
    SessionHistoryMessage,
    SessionHistoryPage,
    SessionInfo,
    SessionListResponse,
    SessionMessageStatus,
    SessionOutcome,
    SessionOutcomeRating,
    SessionOutcomesPage,
)
from pine_assistant.platform_api import (
    ManagedUsersAPI,
    PlatformAPI,
    SyncManagedUsersAPI,
    SyncPlatformAPI,
    validate_external_id,
    validate_user_id,
)
from pine_assistant.sessions import SessionsAPI, SyncSessionsAPI, validate_client_now_date
from pine_assistant.transport.http import API_KEY_PREFIXES, MANAGED_USER_HEADER

__version__ = "0.5.0rc3"
__all__ = [
    "PineAI",
    "AsyncPineAI",
    "Auth",
    "SyncAuth",
    "SessionsAPI",
    "SyncSessionsAPI",
    "AppSettingsAPI",
    "SyncAppSettingsAPI",
    "PlatformAPI",
    "SyncPlatformAPI",
    "ManagedUsersAPI",
    "SyncManagedUsersAPI",
    "ManagedUser",
    "ManagedUserAssistant",
    "AssistantIdentity",
    "AssistantIdentityCatalog",
    "API_KEY_PREFIXES",
    "MANAGED_USER_HEADER",
    "validate_external_id",
    "validate_user_id",
    "validate_client_now_date",
    "AuthIdentity",
    "AuthTicket",
    "RedeemedTicket",
    "SessionInfo",
    "SessionConfig",
    "CallerID",
    "SessionHistoryMessage",
    "SessionHistoryPage",
    "SessionListResponse",
    "SessionMessageStatus",
    "SessionOutcome",
    "SessionOutcomeRating",
    "SessionOutcomesPage",
    "ChatEvent",
    "PineAIError",
    "AuthError",
    "SessionError",
    "PlatformError",
    "ConnectionError",
    "C2SEvent",
    "S2CEvent",
    "SUPPORTED_EVENTS",
    "is_supported_event",
]
