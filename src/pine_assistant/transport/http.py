"""
REST HTTP client for Pine AI — spec sections 4.1, 4.3.
"""

from typing import Any, Optional

import httpx

from pine_assistant.errors import PineAIError

DEFAULT_BASE_URL = "https://www.19pine.ai"


class HttpClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, token: Optional[str] = None):
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._client = httpx.AsyncClient(
            base_url=f"{self._base_url}/api",
            headers={"User-Agent": "pine-assistant-sdk/0.1.0", "Accept": "application/json"},
            timeout=30.0,
        )

    def set_token(self, token: str) -> None:
        self._token = token

    def _auth_headers(self, authenticated: bool) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if authenticated and self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    @staticmethod
    def _unwrap(json_data: Any) -> Any:
        """Unwrap the standard Pine API response: { "status": "success", "data": <actual_data> }"""
        if isinstance(json_data, dict) and "status" in json_data and "data" in json_data:
            return json_data["data"]
        return json_data

    async def get(self, path: str, authenticated: bool = True) -> Any:
        resp = await self._client.get(path, headers=self._auth_headers(authenticated))
        if resp.status_code >= 400:
            raise PineAIError("http_error", f"HTTP {resp.status_code}: {resp.text[:200]}")
        return self._unwrap(resp.json())

    async def post(self, path: str, body: Optional[dict[str, Any]] = None, authenticated: bool = True) -> Any:
        resp = await self._client.post(path, json=body, headers=self._auth_headers(authenticated))
        if resp.status_code >= 400:
            raise PineAIError("http_error", f"HTTP {resp.status_code}: {resp.text[:200]}")
        return self._unwrap(resp.json())

    async def put(self, path: str, body: Optional[dict[str, Any]] = None, authenticated: bool = True) -> Any:
        resp = await self._client.put(path, json=body, headers=self._auth_headers(authenticated))
        if resp.status_code >= 400:
            raise PineAIError("http_error", f"HTTP {resp.status_code}: {resp.text[:200]}")
        return self._unwrap(resp.json())

    async def delete(self, path: str, params: Optional[dict[str, str]] = None, authenticated: bool = True) -> Any:
        resp = await self._client.delete(path, params=params, headers=self._auth_headers(authenticated))
        if resp.status_code >= 400:
            raise PineAIError("http_error", f"HTTP {resp.status_code}: {resp.text[:200]}")
        return self._unwrap(resp.json())

    async def upload(self, path: str, file_path: str, authenticated: bool = True) -> Any:
        """Upload a file via multipart form data."""
        import os
        filename = os.path.basename(file_path)
        with open(file_path, "rb") as f:
            files = {"files": (filename, f)}
            headers = {}
            if authenticated and self._token:
                headers["Authorization"] = f"Bearer {self._token}"
            resp = await self._client.post(path, files=files, headers=headers)
        if resp.status_code >= 400:
            raise PineAIError("http_error", f"HTTP {resp.status_code}: {resp.text[:200]}")
        return self._unwrap(resp.json())

    async def close(self) -> None:
        await self._client.aclose()
