"""
Pine Platform API — resources of a partner, authenticated by its secret key.

A partner's secret key (``pine_sk_live_...`` / ``pine_sk_test_...``) belongs on
its servers only. These requests act as the partner itself and never carry a
``Pine-Managed-User-Id`` header.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote

from pydantic import ValidationError

from pine_assistant.errors import PineAIError, PlatformError
from pine_assistant.models.platform import ManagedUser
from pine_assistant.transport.http import HttpClient, SyncHttpClient

_ASSISTANT_KEYS = frozenset({"identity_id", "name"})
_EXTERNAL_ID = re.compile(r"[A-Za-z0-9_:|-][A-Za-z0-9._:|-]{0,127}")
_MANAGED_USERS = "/platform/v1/managed-users"
_USER_ID = re.compile(r"[1-9][0-9]{0,18}")
_MAX_USER_ID = 9223372036854775807


def validate_external_id(value: str, field: str = "external_id") -> str:
    """Check the backend's external_id rule; the value is never echoed on failure."""
    if not isinstance(value, str) or not _EXTERNAL_ID.fullmatch(value):
        raise ValueError(
            f"{field} must be 1-128 letters, digits, '.', '_', ':', '|' or '-' and must not start with '.'"
        )
    return value


def validate_user_id(value: str, field: str = "user_id") -> str:
    """Check the backend's Pine user ID rule; the value is never echoed on failure."""
    if not isinstance(value, str) or not _USER_ID.fullmatch(value) or int(value) > _MAX_USER_ID:
        raise ValueError(
            f'{field} must be a string such as "123456789": the decimal Pine user ID, '
            "with no sign, spaces or leading zeros"
        )
    return value


def _assistant(assistant: dict[str, str]) -> dict[str, str]:
    if (
        not isinstance(assistant, dict)
        or not assistant.keys() <= _ASSISTANT_KEYS
        or not all(isinstance(value, str) for value in assistant.values())
    ):
        raise ValueError("assistant must be a dict with only the string members identity_id and name")
    return dict(assistant)


def _optional_members(**members: Any) -> dict[str, Any]:
    body = {key: value for key, value in members.items() if value is not None}
    if "assistant" in body:
        body["assistant"] = _assistant(body["assistant"])
    return body


def _create_body(
    external_id: str, email: str, name: str, phone: str | None, assistant: dict[str, str] | None,
) -> dict[str, Any]:
    return {
        "external_id": validate_external_id(external_id),
        "email": email,
        "name": name,
        **_optional_members(phone=phone, assistant=assistant),
    }


def _path(external_id: str) -> str:
    return f"{_MANAGED_USERS}/{quote(validate_external_id(external_id), safe='')}"


def _platform_failure(error: PineAIError) -> PlatformError:
    return PlatformError("Pine Platform API request failed", error.code, status_code=error.status_code)


def _parse_managed_user(payload: object) -> ManagedUser:
    try:
        return ManagedUser.model_validate(payload)
    except ValidationError:
        raise PlatformError("Pine Platform API returned an invalid managed user", "invalid_response") from None


class ManagedUsersAPI:
    def __init__(self, http: HttpClient):
        self._http = http

    async def create(
        self,
        external_id: str,
        *,
        email: str,
        name: str,
        phone: str | None = None,
        assistant: dict[str, str] | None = None,
    ) -> ManagedUser:
        """Create the managed user for ``external_id``, or return the existing one unchanged.

        Idempotent per partner and ``external_id``: a repeated call returns
        the stored user even when ``email``, ``name``, ``phone`` or
        ``assistant`` differ. ``phone`` is an E.164 number you have verified.
        ``assistant`` is ``{"identity_id": ..., "name": ...}``; ``name`` needs
        ``identity_id``.
        """
        body = _create_body(external_id, email, name, phone, assistant)
        try:
            payload = await self._http.post(_MANAGED_USERS, body, as_partner=True)
        except PineAIError as exc:
            raise _platform_failure(exc) from None
        return _parse_managed_user(payload)

    async def get(self, external_id: str) -> ManagedUser:
        """Return the managed user for ``external_id``; another partner's user is a 404."""
        path = _path(external_id)
        try:
            payload = await self._http.get(path, as_partner=True)
        except PineAIError as exc:
            raise _platform_failure(exc) from None
        return _parse_managed_user(payload)

    async def update(
        self,
        external_id: str,
        *,
        email: str | None = None,
        name: str | None = None,
        phone: str | None = None,
        assistant: dict[str, str] | None = None,
    ) -> ManagedUser:
        """Change only the members given and return the stored user.

        ``None`` leaves a member unchanged; ``phone=""`` removes the phone.
        ``assistant`` takes ``identity_id``, ``name`` or both; a name alone
        renames the current identity. The agent uses the new profile from the
        user's next message.
        """
        body = _optional_members(email=email, name=name, phone=phone, assistant=assistant)
        path = _path(external_id)
        try:
            payload = await self._http.patch(path, body, as_partner=True)
        except PineAIError as exc:
            raise _platform_failure(exc) from None
        return _parse_managed_user(payload)


class SyncManagedUsersAPI:
    def __init__(self, http: SyncHttpClient):
        self._http = http

    def create(
        self,
        external_id: str,
        *,
        email: str,
        name: str,
        phone: str | None = None,
        assistant: dict[str, str] | None = None,
    ) -> ManagedUser:
        body = _create_body(external_id, email, name, phone, assistant)
        try:
            payload = self._http.post(_MANAGED_USERS, body, as_partner=True)
        except PineAIError as exc:
            raise _platform_failure(exc) from None
        return _parse_managed_user(payload)

    def get(self, external_id: str) -> ManagedUser:
        path = _path(external_id)
        try:
            payload = self._http.get(path, as_partner=True)
        except PineAIError as exc:
            raise _platform_failure(exc) from None
        return _parse_managed_user(payload)

    def update(
        self,
        external_id: str,
        *,
        email: str | None = None,
        name: str | None = None,
        phone: str | None = None,
        assistant: dict[str, str] | None = None,
    ) -> ManagedUser:
        body = _optional_members(email=email, name=name, phone=phone, assistant=assistant)
        path = _path(external_id)
        try:
            payload = self._http.patch(path, body, as_partner=True)
        except PineAIError as exc:
            raise _platform_failure(exc) from None
        return _parse_managed_user(payload)


class PlatformAPI:
    """Platform API resources for a client built with ``api_key``."""

    def __init__(self, http: HttpClient):
        self.managed_users = ManagedUsersAPI(http)


class SyncPlatformAPI:
    def __init__(self, http: SyncHttpClient):
        self.managed_users = SyncManagedUsersAPI(http)
