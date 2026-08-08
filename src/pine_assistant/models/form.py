"""
Form models — `session:form_to_user`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class FormField(BaseModel):
    name: str
    type: str = "text"
    label: str | None = None
    description: str | None = None
    placeholder: str | None = None
    source: str | None = None
    is_required: bool | None = None
    pii_level: str | None = None
    prefilled: str | None = None
    options: list[str] | None = None


class FormData(BaseModel):
    fields: list[FormField] = []
    content: dict[str, Any] | None = None
    is_submitted: bool = False


class FormToUserData(BaseModel):
    """`session:form_to_user` payload — how a task gathers the account details
    it needs to act."""
    message_to_user: str = ""
    form: FormData = FormData()
