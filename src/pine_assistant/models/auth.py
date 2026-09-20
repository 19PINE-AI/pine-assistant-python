"""Typed responses for Pine authentication resources."""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class AuthIdentity(BaseModel):
    user_id: str

    @field_validator("user_id", mode="before")
    @classmethod
    def normalize_user_id(cls, value: Any) -> str:
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError("user_id must be a string or integer")
        value = str(value).strip()
        if not value.isdecimal() or int(value) <= 0:
            raise ValueError("user_id must be a positive decimal identifier")
        return value


class AuthTicket(BaseModel):
    ticket: str = Field(repr=False)
    expires_in: int

    @field_validator("ticket")
    @classmethod
    def valid_ticket(cls, value: str) -> str:
        if len(value) != 43:
            raise ValueError("ticket must be 43 characters")
        return value


class RedeemedTicket(BaseModel):
    access_token: str = Field(repr=False)

    @field_validator("access_token")
    @classmethod
    def non_empty_token(cls, value: str) -> str:
        if not value:
            raise ValueError("access_token must not be empty")
        return value
