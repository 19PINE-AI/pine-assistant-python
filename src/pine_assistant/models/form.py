"""
Form models — `session:form_to_user`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class FormShowWhen(BaseModel):
    """Visibility rule used by the web form renderer."""

    field: str
    equals: list[Any]


class FormField(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    type: str = "text"
    label: str | None = None
    description: str | None = None
    placeholder: str | None = None
    source: str | None = None
    is_required: bool | None = None
    pii_level: str | None = None
    prefilled: Any | None = None
    options: list[Any] | None = None
    show_when: FormShowWhen | None = None


class FormData(BaseModel):
    fields: list[FormField] = Field(default_factory=list)
    content: dict[str, Any] | None = None
    is_submitted: bool = False


class FormToUserData(BaseModel):
    """`session:form_to_user` payload — how a task gathers the account details
    it needs to act."""
    message_to_user: str = ""
    form: FormData = Field(default_factory=FormData)


@dataclass(frozen=True)
class FormSubmissionResult:
    """What the server was observed to do with a form reply.

    ``delivered`` and ``received`` come from a matching backend message-status
    receipt, which carries the persisted reply's message ID. ``received`` means
    the reply was persisted but delivery was not observed. ``unknown`` is
    intentionally conservative.
    """

    status: Literal["delivered", "received", "unknown"]
    message_id: str | None = None
    request_id: str | None = None
    reason: str | None = dataclass_field(default=None, repr=False)


def encode_form_answers(form: FormData, answers: dict[str, Any]) -> dict[str, Any]:
    """Validate and encode answers exactly as the web form submits them.

    Field definitions and PII levels are server-owned. This only accepts
    declared, visible fields; arrays use compact JSON, matching JSON.stringify.
    """
    if not isinstance(answers, dict):
        raise ValueError("form answers must be a mapping")

    fields = {field.name: field for field in form.fields}
    if not fields or len(fields) != len(form.fields) or not all(fields):
        raise ValueError("form has invalid field names")
    if any(name not in fields for name in answers):
        raise ValueError("form answers contain a field not requested by the server")

    def visible(field: FormField) -> bool:
        rule = field.show_when
        if rule is None:
            return True
        parent = answers.get(rule.field)
        if parent is None:
            return False
        # JavaScript's Array.prototype.includes is type-strict: True does not
        # equal 1. It also compares object values by identity, so JSON-shaped
        # object/array rules are never considered visible across this boundary.
        return any(type(parent) is type(candidate) and parent == candidate for candidate in rule.equals)

    encoded: dict[str, Any] = {}
    for name, value in answers.items():
        if not visible(fields[name]):
            raise ValueError("form answers contain a field that is not visible")
        encoded[name] = json.dumps(value, separators=(",", ":")) if isinstance(value, list) else value

    for field in form.fields:
        if not field.is_required or not visible(field):
            continue
        value = answers.get(field.name)
        if value is None or (isinstance(value, str) and not value.strip()) or (
            isinstance(value, list) and not value
        ):
            raise ValueError("a required visible form field is missing")
    return encoded
