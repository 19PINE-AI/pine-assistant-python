"""Typed responses for the Pine Platform API."""

from pydantic import BaseModel, ConfigDict, field_validator

from pine_assistant.models.session import SessionInfo


class ManagedUser(BaseModel):
    """One end user of a Platform integration, addressed by its ``external_id``.

    ``id`` is the Pine user ID that ``/api/v2`` and Socket.IO payloads carry;
    the Platform API never accepts it as input.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    external_id: str
    email: str
    name: str
    phone: str | None
    created_at: str

    @field_validator("id", mode="before")
    @classmethod
    def normalize_id(cls, value: object) -> str:
        return SessionInfo.normalize_id(value)
