"""
Master Envelope — spec section 4.1.
"""

from typing import Any, Optional
from pydantic import BaseModel


class UserSource(BaseModel):
    role: str  # "user" | "agent" | "system"
    user_id: Optional[str] = None
    device_id: Optional[str] = None
    plat: Optional[str] = None       # Platform identifier (production field)
    version: Optional[str] = None    # App version (production field)


class MessageMetadata(BaseModel):
    event_id: str
    request_id: Optional[str] = None
    timestamp: str
    source: UserSource
    is_volatile: bool = False


class SessionMessagePayload(BaseModel):
    session_id: Optional[str] = None
    message_id: Optional[str] = None
    quoted_message_id: Optional[str] = None
    type: Optional[str] = None
    data: Optional[Any] = None


class MessageEnvelope(BaseModel):
    metadata: MessageMetadata
    type: str
    payload: SessionMessagePayload
