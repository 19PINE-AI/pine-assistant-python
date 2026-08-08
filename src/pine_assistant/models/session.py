"""
Session models — REST session objects and the `session:input_state` payload.
"""

import sys

from pydantic import BaseModel

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        pass


class SessionInfo(BaseModel):
    id: str
    type: str | None = None
    title: str = ""
    # Expiry is carried here and nowhere on the Socket.IO surface: an expired
    # session presents only as a disabled composer, with no code that tells it
    # apart from other causes.
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


class InputStateCode(StrEnum):
    """Reason codes on `session:input_state`."""
    DEFAULT = "default"
    TASK_READY = "task_ready"
    TASK_PROCESSING = "task_processing"
    PROFILE_UPDATE_REQUIRED = "profile_update_required"
    SESSION_SUMMARY = "session_summary"
    PHONE_VERIFICATION_REQUIRED = "phone_verification_required"


ACCEPTING_INPUT = "waiting_input"


class InputState(BaseModel):
    """`session:input_state` payload.

    The blocking condition is read from `code`, never inferred from which other
    events did or did not arrive — the events that elaborate on a condition are
    mostly outside the supported scope.
    """
    content: str = ""
    detail: str = ""
    code: str = ""

    @property
    def accepting_input(self) -> bool:
        return self.content == ACCEPTING_INPUT

    @property
    def blocked(self) -> bool:
        return not self.accepting_input

    @property
    def awaiting_credits(self) -> bool:
        """Blocked on an unconfirmed credit charge.

        The cost is carried by `session:task_ready`. When the balance covers it
        the server starts the task itself; when it does not, the session waits
        here until the balance is restored.
        """
        return self.blocked and self.code == InputStateCode.TASK_READY

    @property
    def needs_phone_verification(self) -> bool:
        """Blocked on phone verification.

        A provisioning prerequisite: it has no in-session remedy, and the event
        that explains it is outside the supported scope.
        """
        return self.blocked and self.code == InputStateCode.PHONE_VERIFICATION_REQUIRED
