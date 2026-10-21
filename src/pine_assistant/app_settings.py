"""
App settings — the assistant identity catalog.

A client with ``api_key`` alone reads the catalog anonymously; a managed-user
or user-token client reads it as that user.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from pine_assistant.errors import PineAIError
from pine_assistant.models.app_settings import AssistantIdentityCatalog
from pine_assistant.transport.http import HttpClient, SyncHttpClient

_ASSISTANT_IDENTITIES = "/v2/app/settings/assistance_identities"
_CATALOG: dict[str, str | int] = {"identities": "catalog"}


def _parse_catalog(payload: Any) -> AssistantIdentityCatalog:
    try:
        return AssistantIdentityCatalog.model_validate(payload)
    except ValidationError:
        raise PineAIError("invalid_response", "Pine API returned an invalid assistant identity catalog") from None


class AppSettingsAPI:
    def __init__(self, http: HttpClient):
        self._http = http

    async def assistant_identities(self) -> AssistantIdentityCatalog:
        """List the assistant identities to choose from; pass an active one's ``id`` as ``identity_id``."""
        payload = await self._http.get(_ASSISTANT_IDENTITIES, authenticated=not self._http.key_only, params=_CATALOG)
        return _parse_catalog(payload)


class SyncAppSettingsAPI:
    def __init__(self, http: SyncHttpClient):
        self._http = http

    def assistant_identities(self) -> AssistantIdentityCatalog:
        payload = self._http.get(_ASSISTANT_IDENTITIES, authenticated=not self._http.key_only, params=_CATALOG)
        return _parse_catalog(payload)
