"""
Master Envelope — spec section 4.1.
"""

from typing import Any

from pydantic import BaseModel


class UserSource(BaseModel):
    role: str  # "user" | "agent" | "system"
    user_id: str | None = None
    device_id: str | None = None
    plat: str | None = None       # Platform identifier (production field)
    version: str | None = None    # App version (production field)


class MessageMetadata(BaseModel):
    event_id: str
    request_id: str | None = None
    timestamp: str
    source: UserSource
    is_volatile: bool = False


class SessionMessagePayload(BaseModel):
    session_id: str | None = None
    message_id: str | None = None
    quoted_message_id: str | None = None
    type: str | None = None
    data: Any | None = None


class MessageEnvelope(BaseModel):
    metadata: MessageMetadata
    type: str
    payload: SessionMessagePayload
