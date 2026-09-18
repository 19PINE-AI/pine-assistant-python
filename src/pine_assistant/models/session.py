"""
Session models — the REST session object.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator


class SessionInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    type: str | None = None
    title: str = ""
    # Expiry is carried here and nowhere on the Socket.IO surface.
    is_stale: bool | None = None
    is_processed: bool | None = None
    state: str = "init"
    version: str | None = None
    created_at: str = ""
    updated_at: str = ""

    @field_validator("id", mode="before")
    @classmethod
    def normalize_id(cls, value: Any) -> str:
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError("id must be a string or integer")
        value = str(value).strip()
        if not value.isdecimal() or int(value) <= 0:
            raise ValueError("id must be a positive decimal identifier")
        return value


class SessionListResponse(BaseModel):
    sessions: list[SessionInfo]
    total: int
    limit: int
    offset: int
