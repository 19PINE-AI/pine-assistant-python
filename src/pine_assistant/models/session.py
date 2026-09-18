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


class SessionMessageStatus(BaseModel):
    """The synchronous acknowledgement of a REST session-message send."""

    model_config = ConfigDict(extra="allow")

    message_id: str | None = None
    request_id: str | None = None
    status: str
    reason: str | None = None
    details: dict[str, Any] | None = None
    revision: str | None = None

    @field_validator("message_id", "revision", mode="before")
    @classmethod
    def normalize_optional_id(cls, value: Any) -> str | None:
        if value is None:
            return None
        return SessionInfo.normalize_id(value)


class SessionOutcomeRating(BaseModel):
    model_config = ConfigDict(extra="allow")

    stars: int
    reasons: list[str] = []
    comment: str = ""
    rated_at: str = ""


class SessionOutcome(BaseModel):
    """A persisted, newest-first ``session:outcome_updated`` record."""

    model_config = ConfigDict(extra="allow")

    outcome_id: str
    session_id: str
    kind: str = ""
    importance: str = ""
    milestone_fact: str
    evidence_excerpts: list[str] = []
    brief: str = ""
    story_description: str = ""
    outcome_narrative: str = ""
    share_draft: str = ""
    engage_prompt: str = ""
    engage_call_objective: str = ""
    estimated_time_saved_min: int = 0
    previous_briefs: list[str] = []
    created_at: str
    rating: SessionOutcomeRating | None = None

    @field_validator("outcome_id", "session_id", mode="before")
    @classmethod
    def normalize_id(cls, value: Any) -> str:
        return SessionInfo.normalize_id(value)


class SessionOutcomesPage(BaseModel):
    model_config = ConfigDict(extra="allow")

    items: list[SessionOutcome]
    next_cursor: str | None = None
    total: int | None = None

    @field_validator("next_cursor", mode="before")
    @classmethod
    def normalize_cursor(cls, value: Any) -> str | None:
        if value is None:
            return None
        return SessionInfo.normalize_id(value)
