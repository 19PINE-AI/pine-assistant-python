"""
Envelope construction and parsing — spec section 4.1.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pine_assistant.models.envelope import MessageEnvelope, MessageMetadata, SessionMessagePayload, UserSource


def build_envelope(
    event_type: str,
    data: Any,
    user_id: str,
    device_id: str,
    session_id: Optional[str] = None,
    message_id: Optional[str] = None,
    request_id: Optional[str] = None,
    is_volatile: bool = False,
) -> dict[str, Any]:
    """Build a C2S message envelope as a dict ready for Socket.IO emit."""
    envelope = MessageEnvelope(
        metadata=MessageMetadata(
            event_id=str(uuid.uuid4()),
            request_id=request_id or str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=UserSource(role="user", user_id=user_id, device_id=device_id),
            is_volatile=is_volatile,
        ),
        type=event_type,
        payload=SessionMessagePayload(
            session_id=session_id,
            message_id=message_id,
            type=event_type,
            data=data,
        ),
    )
    return envelope.model_dump()


def parse_envelope(raw: dict[str, Any]) -> Optional[MessageEnvelope]:
    """Parse an S2C message envelope. Returns None if invalid."""
    try:
        return MessageEnvelope.model_validate(raw)
    except Exception:
        return None
