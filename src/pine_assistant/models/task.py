"""
Task models — spec 5.1.2 session:work_log, session:task_finished, session:task_ready.
"""

from typing import Any, Optional
from pydantic import BaseModel


class TaskReadyData(BaseModel):
    """session:task_ready payload"""
    required: int = 0
    suggested: Optional[int] = None
    confirmed: bool = False


class WorkLogStep(BaseModel):
    """Work log step — spec 5.1.2 session:work_log"""
    id: str = ""
    step_type: str = ""
    step_title: str = ""
    step_details: Optional[str] = None
    status: str = ""
    start_time: Optional[int] = None
    data: Optional[dict[str, Any]] = None
    can_retry: Optional[bool] = None
    can_cancel: Optional[bool] = None
    is_collapsed: Optional[bool] = None


class WorkLogData(BaseModel):
    """session:work_log payload.data"""
    steps: list[WorkLogStep] = []


class WorkLogPartData(BaseModel):
    """session:work_log_part payload.data"""
    step_id: str = ""
    text_delta: Optional[str] = None
    data_delta: Optional[dict[str, Any]] = None
    status: Optional[str] = None


class Achievement(BaseModel):
    id: str = ""
    title: str = ""
    description: Optional[str] = None
    icon_url: Optional[str] = None
    rarity: Optional[str] = None
    is_new: Optional[bool] = None


class TaskCompletionSummary(BaseModel):
    time_saved_minutes: Optional[int] = None
    hold_time_avoided_mins: Optional[int] = None
    calls_made: Optional[int] = None
    call_duration_mins: Optional[int] = None
    emails_sent: Optional[int] = None
    web_tasks_completed: Optional[int] = None
    money_saved: Optional[float] = None
    money_saved_currency: Optional[str] = None
    credits_invested: Optional[int] = None
    achievements: Optional[list[Achievement]] = None


class TaskCompletion(BaseModel):
    result_title: str = ""
    result_description: Optional[str] = None
    summary: Optional[TaskCompletionSummary] = None
    share_text: Optional[str] = None
    engage_enabled: Optional[bool] = None
    engage_prompt: Optional[str] = None
    engage_status: Optional[str] = None


class TaskFinishedData(BaseModel):
    """session:task_finished payload.data"""
    status: str = ""
    completion: Optional[TaskCompletion] = None


class ThinkingStep(BaseModel):
    """session:thinking step"""
    kind: str = ""
    title: Optional[str] = None
    status: Optional[str] = None
    content: Optional[str] = None
    thinking_data: Optional[dict[str, Any]] = None


class InteractiveAuthData(BaseModel):
    """session:interactive_auth_confirmation payload.data (S2C)"""
    confirmation_id: Optional[str] = None
    message_to_user: str = ""
    verification_types: Optional[list[str]] = None
    verification_guidance: Optional[dict[str, Any]] = None
    scheduled_time: Optional[str] = None
    scheduled_call_reminder: Optional[bool] = None
    user_phone: Optional[str] = None
    pine_caller_id: Optional[str] = None
    caller_first_name: Optional[str] = None
    caller_last_name: Optional[str] = None
    expires_at: Optional[str] = None


class ThreeWayCallData(BaseModel):
    """session:three_way_call payload.data"""
    title: Optional[str] = None
    content: Optional[str] = None
    caller_id_number: Optional[str] = None
    caller_first_name: Optional[str] = None
    caller_last_name: Optional[str] = None
