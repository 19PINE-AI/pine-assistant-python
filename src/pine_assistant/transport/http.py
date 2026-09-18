"""
REST HTTP client for Pine AI — spec sections 4.1, 4.3.
"""

from typing import Any

import httpx

from pine_assistant.errors import PineAIError

DEFAULT_BASE_URL = "https://www.19pine.ai"
DEFAULT_API_BASE_PATH = "/api"
_USER_AGENT = "pine-assistant-sdk"

# Public HTTPX timeout inputs accepted by both sync and async clients.  HTTPX's
# internal ``TimeoutTypes`` alias is not exported in every supported release.
TimeoutConfig = httpx.Timeout | float | None | tuple[float] | tuple[float, float] | tuple[float, float, float] | tuple[float, float, float, float]


def _api_url(base_url: str, api_base_path: str) -> str:
    if not base_url:
        raise ValueError("base_url must not be empty")
    path = api_base_path.strip()
    if not path:
        return base_url.rstrip("/")
    return f"{base_url.rstrip('/')}/{path.strip('/')}"


def _error_code(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except (ValueError, TypeError):
        return "http_error"
    if not isinstance(payload, dict):
        return "http_error"
    error = payload.get("error")
    if isinstance(error, dict) and isinstance(error.get("code"), str):
        return error["code"]
    if isinstance(payload.get("code"), str):
        return payload["code"]
    return "http_error"


def _json_data(response: httpx.Response) -> Any:
    if not 200 <= response.status_code < 300:
        raise PineAIError(
            _error_code(response),
            f"Pine API request failed with HTTP {response.status_code}",
            status_code=response.status_code,
        )
    if response.status_code == 204:
        return None
    try:
        payload = response.json()
    except (ValueError, TypeError) as exc:
        raise PineAIError("invalid_response", "Pine API returned invalid JSON") from exc
    if isinstance(payload, dict) and "status" in payload and "data" in payload:
        return payload["data"]
    return payload


class HttpClient:
    """Async HTTP client; injected clients remain caller-owned."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        token: str | None = None,
        *,
        api_base_path: str = DEFAULT_API_BASE_PATH,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: TimeoutConfig = 30.0,
    ):
        if client is not None and transport is not None:
            raise ValueError("pass either client or transport, not both")
        self._api_base_url = _api_url(base_url, api_base_path)
        self._token = token
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
            timeout=timeout,
            transport=transport,
            follow_redirects=False,
        )

    def set_token(self, token: str | None) -> None:
        self._token = token

    def _auth_headers(self, authenticated: bool, token: str | None = None) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        actual_token = self._token if token is None else token
        if authenticated and actual_token:
            headers["Authorization"] = f"Bearer {actual_token}"
        return headers

    def _url(self, path: str) -> str:
        # Absolute paths keep an injected client's base_url from changing the
        # destination, and redirects are refused to protect bearer credentials.
        return f"{self._api_base_url}/{path.lstrip('/')}"

    async def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        params: dict[str, str | int] | None = None,
        authenticated: bool = True,
        token: str | None = None,
        files: Any = None,
    ) -> Any:
        actual_token = self._token if token is None else token
        headers = self._auth_headers(authenticated, token)
        if files is not None:
            # httpx must generate the multipart boundary itself.
            headers.pop("Content-Type", None)
        try:
            request = self._client.build_request(
                method, self._url(path), json=body, params=params, headers=headers, files=files,
            )
            # An injected client may carry default authentication. Never let it
            # authenticate an anonymous request or a request for another user.
            if authenticated and actual_token:
                request.headers["Authorization"] = f"Bearer {actual_token}"
            else:
                request.headers.pop("Authorization", None)
            response = await self._client.send(request, follow_redirects=False, auth=None)
        except httpx.TimeoutException as exc:
            raise PineAIError("timeout", "Pine API request timed out") from exc
        except httpx.RequestError as exc:
            raise PineAIError("connection_error", "Pine API request could not be completed") from exc
        return _json_data(response)

    async def get(self, path: str, authenticated: bool = True, *, token: str | None = None,
                  params: dict[str, str | int] | None = None) -> Any:
        return await self._request("GET", path, authenticated=authenticated, token=token, params=params)

    async def post(self, path: str, body: dict[str, Any] | None = None, authenticated: bool = True,
                   *, token: str | None = None) -> Any:
        return await self._request("POST", path, body=body, authenticated=authenticated, token=token)

    async def put(self, path: str, body: dict[str, Any] | None = None, authenticated: bool = True,
                  *, token: str | None = None) -> Any:
        return await self._request("PUT", path, body=body, authenticated=authenticated, token=token)

    async def delete(self, path: str, params: dict[str, str | int] | None = None, authenticated: bool = True,
                     *, token: str | None = None) -> Any:
        return await self._request("DELETE", path, params=params, authenticated=authenticated, token=token)

    async def upload(self, path: str, file_path: str, authenticated: bool = True) -> Any:
        """Upload a file via multipart form data."""
        import os
        with open(file_path, "rb") as file_handle:
            return await self._request(
                "POST", path, authenticated=authenticated,
                files={"files": (os.path.basename(file_path), file_handle)},
            )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class SyncHttpClient:
    """Synchronous REST transport with the same serialization and error rules."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        token: str | None = None,
        *,
        api_base_path: str = DEFAULT_API_BASE_PATH,
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: TimeoutConfig = 30.0,
    ) -> None:
        if client is not None and transport is not None:
            raise ValueError("pass either client or transport, not both")
        self._api_base_url = _api_url(base_url, api_base_path)
        self._token = token
        self._owns_client = client is None
        self._client = client or httpx.Client(
            headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
            timeout=timeout,
            transport=transport,
            follow_redirects=False,
        )

    def set_token(self, token: str | None) -> None:
        self._token = token

    def _auth_headers(self, authenticated: bool, token: str | None = None) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        actual_token = self._token if token is None else token
        if authenticated and actual_token:
            headers["Authorization"] = f"Bearer {actual_token}"
        return headers

    def _url(self, path: str) -> str:
        return f"{self._api_base_url}/{path.lstrip('/')}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        params: dict[str, str | int] | None = None,
        authenticated: bool = True,
        token: str | None = None,
        files: Any = None,
    ) -> Any:
        actual_token = self._token if token is None else token
        headers = self._auth_headers(authenticated, token)
        if files is not None:
            headers.pop("Content-Type", None)
        try:
            request = self._client.build_request(
                method, self._url(path), json=body, params=params, headers=headers, files=files,
            )
            if authenticated and actual_token:
                request.headers["Authorization"] = f"Bearer {actual_token}"
            else:
                request.headers.pop("Authorization", None)
            response = self._client.send(request, follow_redirects=False, auth=None)
        except httpx.TimeoutException as exc:
            raise PineAIError("timeout", "Pine API request timed out") from exc
        except httpx.RequestError as exc:
            raise PineAIError("connection_error", "Pine API request could not be completed") from exc
        return _json_data(response)

    def get(self, path: str, authenticated: bool = True, *, token: str | None = None,
            params: dict[str, str | int] | None = None) -> Any:
        return self._request("GET", path, authenticated=authenticated, token=token, params=params)

    def post(self, path: str, body: dict[str, Any] | None = None, authenticated: bool = True,
             *, token: str | None = None) -> Any:
        return self._request("POST", path, body=body, authenticated=authenticated, token=token)

    def put(self, path: str, body: dict[str, Any] | None = None, authenticated: bool = True,
            *, token: str | None = None) -> Any:
        return self._request("PUT", path, body=body, authenticated=authenticated, token=token)

    def delete(self, path: str, params: dict[str, str | int] | None = None, authenticated: bool = True,
               *, token: str | None = None) -> Any:
        return self._request("DELETE", path, params=params, authenticated=authenticated, token=token)

    def upload(self, path: str, file_path: str, authenticated: bool = True) -> Any:
        import os
        with open(file_path, "rb") as file_handle:
            return self._request(
                "POST", path, authenticated=authenticated,
                files={"files": (os.path.basename(file_path), file_handle)},
            )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
