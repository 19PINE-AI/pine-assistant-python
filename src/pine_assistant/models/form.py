"""
Form models — spec 5.1.2 session:form_to_user.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class FormField(BaseModel):
    name: str
    type: str = "text"
    label: Optional[str] = None
    placeholder: Optional[str] = None
    is_required: Optional[bool] = None
    pii_level: Optional[str] = None
    prefilled: Optional[str] = None
    options: Optional[list[str]] = None


class FormData(BaseModel):
    fields: list[FormField] = []
    content: Optional[dict[str, Any]] = None
    is_submitted: bool = False


class FormToUserData(BaseModel):
    """S2C session:form_to_user payload.data"""
    message_to_user: str = ""
    form: FormData = FormData()


class AskForLocationData(BaseModel):
    """S2C session:ask_for_location payload.data"""
    message_to_user: str = ""
    form: FormData = FormData()


class LocationSelectionData(BaseModel):
    """S2C session:location_selection payload.data"""
    message_to_user: str = ""
    locations: list[dict[str, Any]] = Field(default_factory=list, alias="list")
    selected: list[dict[str, Any]] = Field(default_factory=list)
    limit: int = 0

    model_config = {"populate_by_name": True}
