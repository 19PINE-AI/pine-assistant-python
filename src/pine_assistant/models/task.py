"""
Task models — `session:task_finished`, `session:tool_status`,
`session:llm_thinking`, `session:restriction`.
"""

from typing import Any

from pydantic import BaseModel


class Achievement(BaseModel):
    id: str = ""
    title: str = ""
    description: str | None = None
    icon_url: str | None = None
    rarity: str | None = None
    is_new: bool | None = None


class TaskCompletionSummary(BaseModel):
    """Quantified outcome. `brief` is its only textual field."""
    brief: str | None = None
    time_saved_minutes: int | None = None
    hold_time_avoided_mins: int | None = None
    calls_made: int | None = None
    call_duration_mins: int | None = None
    emails_sent: int | None = None
    web_tasks_completed: int | None = None
    silos_conquered: int | None = None
    obstacles_overcome: int | None = None
    money_saved: float | None = None
    money_saved_currency: str | None = None
    credits_invested: int | None = None
    achievements: list[Achievement] | None = None


class TaskCompletion(BaseModel):
    """The task's conclusion.

    The textual result is `result_title`, `result_description` and
    `outcome_narrative`; `summary` is quantified.
    """
    result_title: str = ""
    result_description: str | None = None
    outcome_narrative: str | None = None
    summary: TaskCompletionSummary | None = None


class TaskFinishedData(BaseModel):
    """`session:task_finished` payload."""
    status: str = ""
    completion: TaskCompletion | None = None


class ToolStatusSummary(BaseModel):
    """Outcome of one tool operation. `text` is its only textual field."""
    text: str | None = None
    duration: int | None = None
    actions_count: int | None = None
    credits_consumed: int | None = None


class ToolStatusData(BaseModel):
    """`session:tool_status` payload — the record of one asynchronous operation.

    Outbound calls are reported here: `target` is the number called, and the
    duration, credits and textual outcome are on `summary`. Live updates reuse
    the same `message_id`.

    Distinct from the task-level result. A `tool_call` step in
    `session:llm_thinking` describes the same operation and must not be
    presented as a second entry.
    """
    operation_id: str = ""
    tool_name: str | None = None
    provider: str | None = None
    target: str | None = None
    start_time: str | None = None
    status: str | None = None
    summary: ToolStatusSummary | None = None


class ToolCallHistoryEntry(BaseModel):
    tool_name: str | None = None
    status: str | None = None
    title: str | None = None
    content: str | None = None


class LLMThinkingData(BaseModel):
    """`session:llm_thinking` payload — reasoning and tool-call trace.

    Search has no dedicated event; search activity appears here as a `tool_call`
    step.
    """
    type: str = ""
    title: str | None = None
    content: str | None = None
    tool_name: str | None = None
    status: str | None = None
    turn_id: str | None = None
    thinking_id: str | None = None
    final: bool = False
    history: list[ToolCallHistoryEntry] | None = None


class RestrictionData(BaseModel):
    """`session:restriction` payload — the only statement that a task will not
    complete."""
    level: str = ""
    reason: str | None = None
    message: str | None = None


class MessageStatusData(BaseModel):
    """`session:message_status` payload — the only means of telling a rejected
    or rate-limited message from one still being processed."""
    status: str = ""
    message_id: str | None = None
    request_id: str | None = None
    reason: str | None = None
    details: dict[str, Any] | None = None


class RichContentData(BaseModel):
    """`session:rich_content` payload — a structured document.

    Its content is not repeated in `session:text`; ignoring this event loses the
    content entirely.
    """
    title: str = ""
    content: str = ""
    subtitle: str | None = None
    type: str | None = None
    message_to_user: str | None = None
