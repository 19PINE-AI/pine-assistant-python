"""
Pine Platform API — enterprise (tenant) resources authenticated by a secret key.

A tenant's secret key (``pine_sk_live_...`` / ``pine_sk_test_...``) belongs on
its servers only. These requests act as the tenant itself and never carry a
``Pine-Managed-User`` header.
"""

from __future__ import annotations

import re
from urllib.parse import quote

from pydantic import ValidationError

from pine_assistant.errors import PineAIError, PlatformError
from pine_assistant.models.platform import ManagedUser
from pine_assistant.transport.http import HttpClient, SyncHttpClient

API_KEY_PREFIX = "pine_sk_"
_EXTERNAL_ID = re.compile(r"[A-Za-z0-9_:|-][A-Za-z0-9._:|-]{0,127}")
_MANAGED_USERS = "/platform/v1/managed-users"


def validate_external_id(value: str, field: str = "external_id") -> str:
    """Check the backend's external_id rule; the value is never echoed on failure."""
    if not isinstance(value, str) or not _EXTERNAL_ID.fullmatch(value):
        raise ValueError(
            f"{field} must be 1-128 letters, digits, '.', '_', ':', '|' or '-' and must not start with '.'"
        )
    return value


def _create_body(external_id: str, email: str, name: str, phone: str | None) -> dict[str, str]:
    body = {"external_id": validate_external_id(external_id), "email": email, "name": name}
    if phone is not None:
        body["phone"] = phone
    return body


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

    async def create(self, external_id: str, *, email: str, name: str, phone: str | None = None) -> ManagedUser:
        """Create the managed user for ``external_id``, or return the existing one unchanged.

        Idempotent per integration and ``external_id``: a repeated call returns
        the stored user even when ``email``, ``name`` or ``phone`` differ.
        ``phone`` is an E.164 number you have verified.
        """
        body = _create_body(external_id, email, name, phone)
        try:
            payload = await self._http.post(_MANAGED_USERS, body, as_tenant=True)
        except PineAIError as exc:
            raise _platform_failure(exc) from None
        return _parse_managed_user(payload)

    async def get(self, external_id: str) -> ManagedUser:
        """Return the managed user for ``external_id``; another integration's user is a 404."""
        path = _path(external_id)
        try:
            payload = await self._http.get(path, as_tenant=True)
        except PineAIError as exc:
            raise _platform_failure(exc) from None
        return _parse_managed_user(payload)


class SyncManagedUsersAPI:
    def __init__(self, http: SyncHttpClient):
        self._http = http

    def create(self, external_id: str, *, email: str, name: str, phone: str | None = None) -> ManagedUser:
        body = _create_body(external_id, email, name, phone)
        try:
            payload = self._http.post(_MANAGED_USERS, body, as_tenant=True)
        except PineAIError as exc:
            raise _platform_failure(exc) from None
        return _parse_managed_user(payload)

    def get(self, external_id: str) -> ManagedUser:
        path = _path(external_id)
        try:
            payload = self._http.get(path, as_tenant=True)
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
