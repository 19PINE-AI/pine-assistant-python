"""
Session models — spec section 4.3.
"""

from typing import Any, Optional
from pydantic import BaseModel


class SessionInfo(BaseModel):
    id: str
    type: Optional[str] = None
    title: str = ""
    is_stale: Optional[bool] = None
    is_processed: Optional[bool] = None
    state: str = "init"
    version: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""


class SessionListResponse(BaseModel):
    sessions: list[SessionInfo]
    total: int
    limit: int
    offset: int
