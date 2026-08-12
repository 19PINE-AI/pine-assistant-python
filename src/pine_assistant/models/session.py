"""
Session models — the REST session object.
"""

from pydantic import BaseModel


class SessionInfo(BaseModel):
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


class SessionListResponse(BaseModel):
    sessions: list[SessionInfo]
    total: int
    limit: int
    offset: int
