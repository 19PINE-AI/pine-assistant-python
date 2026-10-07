"""Platform API and managed-user mode against a fake backend on 127.0.0.1."""

import json
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
import socketio
from aiohttp import web

from pine_assistant import (
    API_KEY_PREFIXES,
    MANAGED_USER_HEADER,
    AsyncPineAI,
    ManagedUser,
    PineAI,
    PineAIError,
    PlatformError,
    SessionError,
    validate_external_id,
)

API_KEY = "pine_sk_test_c2VjcmV0LWtleS1mb3ItdGVzdHM"
EXTERNAL_ID = "acme:user|42-x"
ENCODED_EXTERNAL_ID = "acme%3Auser%7C42-x"
PINE_USER_ID = "4242"


def success(data):
    return {"status": "success", "data": data}


def managed_user(**overrides):
    return {
        "id": PINE_USER_ID,
        "external_id": EXTERNAL_ID,
        "email": "ada@example.com",
        "name": "Ada Lovelace",
        "phone": None,
        "created_at": "2026-10-07T00:00:00Z",
        **overrides,
    }


STORED_USER = managed_user(email="stored@example.com", name="Stored Name", phone="+14155550199")


@dataclass
class FakeBackend:
    base_url: str = ""
    requests: list[dict[str, Any]] = field(default_factory=list)
    handshakes: list[Any] = field(default_factory=list)
    handshake_clients: list[str | None] = field(default_factory=list)
    envelopes: list[dict[str, Any]] = field(default_factory=list)
    error_status: int | None = None
    error_code: str = "platform_failure"
    existing: bool = False

    def app(self) -> web.Application:
        sio = socketio.AsyncServer(async_mode="aiohttp")
        app = web.Application()
        sio.attach(app, socketio_path="/api/v2/socket.io/")

        @sio.event
        async def connect(sid, environ, auth):
            self.handshakes.append(auth)
            self.handshake_clients.append(environ.get("HTTP_PINE_CLIENT"))
            await sio.emit("ready", {}, to=sid)

        @sio.on("session:history")
        async def history(sid, envelope):
            self.envelopes.append(envelope)
            await sio.emit(
                "session:history",
                {
                    "metadata": {"request_id": envelope["metadata"]["request_id"]},
                    "payload": {"session_id": envelope["payload"]["session_id"], "data": {"messages": []}},
                },
                to=sid,
            )

        async def record(request: web.Request) -> web.Response | None:
            body = await request.text()
            self.requests.append({
                "method": request.method,
                "path": request.raw_path,
                "headers": dict(request.headers),
                "body": json.loads(body) if body else None,
            })
            if self.error_status is not None:
                # A hostile body: none of it may surface through the SDK.
                return web.json_response(
                    {"status": "error", "error": {"code": self.error_code, "message": f"{API_KEY} {EXTERNAL_ID}"}},
                    status=self.error_status,
                )
            return None

        async def create_user(request):
            failure = await record(request)
            if failure:
                return failure
            if self.existing:
                # The stored user is returned unchanged, whatever the request carried.
                return web.json_response(success(STORED_USER), status=200)
            return web.json_response(success(managed_user(**self.requests[-1]["body"])), status=201)

        async def get_user(request):
            return await record(request) or web.json_response(success(managed_user()))

        async def auth_me(request):
            return await record(request) or web.json_response(success({"user_id": PINE_USER_ID}))

        async def sessions(request):
            return await record(request) or web.json_response(
                success({"sessions": [{"id": "7"}], "total": 1, "limit": 30, "offset": 0})
            )

        app.router.add_post("/api/platform/v1/managed-users", create_user)
        app.router.add_get("/api/platform/v1/managed-users/{external_id}", get_user)
        app.router.add_get("/api/v2/auth/me", auth_me)
        app.router.add_get("/api/v2/sessions", sessions)
        app.router.add_post("/api/v2/sessions", sessions)
        return app


@pytest.fixture
async def backend():
    fake = FakeBackend()
    runner = web.AppRunner(fake.app())
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0][:2]
    fake.base_url = f"http://{host}:{port}"
    try:
        yield fake
    finally:
        await runner.cleanup()


def _assert_no_secrets(text: str) -> None:
    assert API_KEY not in text
    assert EXTERNAL_ID not in text


@pytest.mark.asyncio
async def test_managed_user_rest_requests_carry_key_header_and_client_name(backend):
    async with AsyncPineAI(
        api_key=API_KEY, managed_user=EXTERNAL_ID, client_name="mcp", base_url=backend.base_url,
    ) as client:
        listed = await client.sessions.list()

    assert listed.sessions[0].id == "7"
    [request] = backend.requests
    assert request["headers"]["Authorization"] == f"Bearer {API_KEY}"
    assert request["headers"]["Pine-Managed-User"] == EXTERNAL_ID
    assert request["headers"]["Pine-Client"] == "mcp"


@pytest.mark.asyncio
async def test_managed_user_connect_resolves_user_once_and_sends_handshake(backend):
    async with AsyncPineAI(api_key=API_KEY, managed_user=EXTERNAL_ID, base_url=backend.base_url) as client:
        await client.connect()
        await client.get_history("7")
        await client.disconnect()
        await client.connect()

    assert [request["path"] for request in backend.requests] == ["/api/v2/auth/me"]
    assert backend.requests[0]["headers"]["Pine-Managed-User"] == EXTERNAL_ID
    assert "Pine-Client" not in backend.requests[0]["headers"]
    assert backend.handshakes == [{"token": API_KEY, "managed_user": EXTERNAL_ID}] * 2
    assert backend.handshake_clients == [None, None]
    assert backend.envelopes[0]["metadata"]["source"]["user_id"] == PINE_USER_ID


@pytest.mark.asyncio
async def test_client_name_is_sent_on_the_socketio_handshake(backend):
    async with AsyncPineAI(
        api_key=API_KEY, managed_user=EXTERNAL_ID, client_name="mcp", base_url=backend.base_url,
    ) as client:
        await client.connect()
    assert backend.requests[0]["headers"]["Pine-Client"] == "mcp"
    assert backend.handshake_clients == ["mcp"]


@pytest.mark.asyncio
async def test_user_token_handshake_is_unchanged(backend):
    async with AsyncPineAI(access_token="user-token", user_id="17", base_url=backend.base_url) as client:
        await client.connect()
    assert backend.handshakes == [{"token": "user-token"}]
    assert backend.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("existing", "phone"), [(False, "+14155550100"), (True, None)])
async def test_create_managed_user_is_typed_and_acts_as_the_tenant(backend, existing, phone):
    backend.existing = existing
    async with AsyncPineAI(api_key=API_KEY, managed_user="someone-else", base_url=backend.base_url) as client:
        user = await client.platform.managed_users.create(
            EXTERNAL_ID, email="ada@example.com", name="Ada Lovelace", phone=phone,
        )

    assert isinstance(user, ManagedUser)
    assert user.id == PINE_USER_ID
    assert user.external_id == EXTERNAL_ID
    if existing:
        assert user == ManagedUser.model_validate(STORED_USER)
    else:
        assert (user.email, user.name, user.phone) == ("ada@example.com", "Ada Lovelace", phone)
    [request] = backend.requests
    assert request["method"] == "POST"
    assert request["path"] == "/api/platform/v1/managed-users"
    expected = {"external_id": EXTERNAL_ID, "email": "ada@example.com", "name": "Ada Lovelace"}
    assert request["body"] == (expected | {"phone": phone} if phone else expected)
    assert request["headers"]["Authorization"] == f"Bearer {API_KEY}"
    assert "Pine-Managed-User" not in request["headers"]


@pytest.mark.asyncio
async def test_get_managed_user_percent_encodes_the_external_id(backend):
    async with AsyncPineAI(api_key=API_KEY, base_url=backend.base_url) as client:
        user = await client.platform.managed_users.get(EXTERNAL_ID)

    assert user.id == PINE_USER_ID
    assert backend.requests[0]["path"] == f"/api/platform/v1/managed-users/{ENCODED_EXTERNAL_ID}"
    assert "Pine-Managed-User" not in backend.requests[0]["headers"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 429, 500])
async def test_platform_errors_keep_code_and_status_without_upstream_body(backend, status):
    backend.error_status = status
    async with AsyncPineAI(api_key=API_KEY, base_url=backend.base_url) as client:
        with pytest.raises(PlatformError) as excinfo:
            await client.platform.managed_users.get(EXTERNAL_ID)

    assert excinfo.value.status_code == status
    assert excinfo.value.code == "platform_failure"
    assert excinfo.value.__cause__ is None
    _assert_no_secrets(str(excinfo.value))
    _assert_no_secrets(repr(excinfo.value))


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 404])
async def test_managed_user_resolution_failure_is_typed_and_does_not_connect(backend, status):
    backend.error_status = status
    async with AsyncPineAI(api_key=API_KEY, managed_user=EXTERNAL_ID, base_url=backend.base_url) as client:
        with pytest.raises(PineAIError) as excinfo:
            await client.connect()
    assert excinfo.value.status_code == status
    assert backend.handshakes == []
    _assert_no_secrets(str(excinfo.value))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"api_key": API_KEY, "managed_user": EXTERNAL_ID, "access_token": "user-token"},
        {"api_key": API_KEY, "managed_user": EXTERNAL_ID, "user_id": "17"},
        {"managed_user": EXTERNAL_ID},
        {"api_key": "not-a-pine-key", "managed_user": EXTERNAL_ID},
        {"api_key": "pine_sk_prod_c2VjcmV0", "managed_user": EXTERNAL_ID},
        {"api_key": "pine_sk_c2VjcmV0"},
        {"api_key": API_KEY, "managed_user": ".leading-dot"},
        {"api_key": API_KEY, "managed_user": "x" * 129},
        {"api_key": API_KEY, "managed_user": "has space"},
        {"api_key": API_KEY, "client_name": "bad name"},
    ],
)
def test_invalid_platform_configuration_is_rejected_without_echoing_values(kwargs):
    client_types = (AsyncPineAI,) if "user_id" in kwargs else (AsyncPineAI, PineAI)
    for client_type in client_types:
        with pytest.raises(ValueError) as excinfo:
            client_type(base_url="https://pine.test", **kwargs)
        _assert_no_secrets(str(excinfo.value))
        for value in kwargs.values():
            assert value not in str(excinfo.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("external_id", ["", "a/b", "../x", ".hidden", "x" * 129])
async def test_invalid_external_id_is_rejected_before_any_request(backend, external_id):
    async with AsyncPineAI(api_key=API_KEY, base_url=backend.base_url) as client:
        with pytest.raises(ValueError):
            await client.platform.managed_users.get(external_id)
        with pytest.raises(ValueError):
            await client.platform.managed_users.create(external_id, email="a@example.com", name="A")
    assert backend.requests == []


@pytest.mark.asyncio
async def test_injected_client_cannot_add_or_override_the_managed_user():
    seen = []

    async def handler(request):
        seen.append(request)
        if request.url.path.startswith("/api/platform/"):
            return httpx.Response(200, json=success(managed_user()))
        return httpx.Response(200, json=success({"user_id": "1"}))

    injected = httpx.AsyncClient(
        headers={"Pine-Managed-User": "injected"}, transport=httpx.MockTransport(handler),
    )
    async with AsyncPineAI(access_token="user-token", base_url="https://pine.test", http_client=injected) as client:
        await client.auth.me()
    async with AsyncPineAI(
        api_key=API_KEY, managed_user=EXTERNAL_ID, base_url="https://pine.test", http_client=injected,
    ) as client:
        await client.auth.me()
        await client.platform.managed_users.get(EXTERNAL_ID)
    await injected.aclose()

    assert "Pine-Managed-User" not in seen[0].headers
    assert seen[1].headers["Pine-Managed-User"] == EXTERNAL_ID
    assert "Pine-Managed-User" not in seen[2].headers


def test_sync_client_supports_platform_and_managed_user_requests():
    seen = []

    def handler(request):
        seen.append(request)
        if request.url.path.startswith("/api/platform/"):
            return httpx.Response(201, json=success(managed_user()))
        return httpx.Response(200, json=success({"sessions": [], "total": 0, "limit": 30, "offset": 0}))

    with PineAI(
        api_key=API_KEY, managed_user=EXTERNAL_ID, client_name="mcp", base_url="https://pine.test",
        http_transport=httpx.MockTransport(handler),
    ) as client:
        created = client.platform.managed_users.create(EXTERNAL_ID, email="ada@example.com", name="Ada Lovelace")
        client.sessions.list()

    assert created.id == PINE_USER_ID
    assert seen[0].url.raw_path == b"/api/platform/v1/managed-users"
    assert "Pine-Managed-User" not in seen[0].headers
    assert seen[1].headers["Pine-Managed-User"] == EXTERNAL_ID
    assert all(request.headers["Authorization"] == f"Bearer {API_KEY}" for request in seen)
    assert all(request.headers["Pine-Client"] == "mcp" for request in seen)


@pytest.mark.asyncio
@pytest.mark.parametrize(("status", "code"), [(403, "platform_route_not_allowed"), (429, "rate_limited")])
async def test_user_scoped_failures_keep_code_and_status_without_upstream_body(backend, status, code):
    backend.error_status, backend.error_code = status, code
    async with AsyncPineAI(api_key=API_KEY, managed_user=EXTERNAL_ID, base_url=backend.base_url) as client:
        with pytest.raises(SessionError) as excinfo:
            await client.sessions.list()
    assert (excinfo.value.code, excinfo.value.status_code) == (code, status)
    _assert_no_secrets(str(excinfo.value))


@pytest.mark.asyncio
async def test_managed_user_is_never_sent_with_a_token_other_than_the_key(backend):
    async with AsyncPineAI(api_key=API_KEY, managed_user=EXTERNAL_ID, base_url=backend.base_url) as client:
        client.http.set_token("user-jwt")
        await client.sessions.list()
        await client.http.get("/v2/sessions", token=API_KEY)
        await client.http.get("/v2/sessions", token="other-jwt")

    first, second, third = (request["headers"] for request in backend.requests)
    assert first["Authorization"] == "Bearer user-jwt" and MANAGED_USER_HEADER not in first
    assert second["Authorization"] == f"Bearer {API_KEY}" and second[MANAGED_USER_HEADER] == EXTERNAL_ID
    assert third["Authorization"] == "Bearer other-jwt" and MANAGED_USER_HEADER not in third


@pytest.mark.asyncio
async def test_caller_headers_cannot_override_identity(backend):
    forged = {"Authorization": "Bearer forged", MANAGED_USER_HEADER: "someone-else", "Pine-Client": "forged"}
    async with AsyncPineAI(api_key=API_KEY, managed_user=EXTERNAL_ID, base_url=backend.base_url) as client:
        await client.http.post("/v2/sessions", {}, headers=forged)
    async with AsyncPineAI(access_token="user-token", base_url=backend.base_url) as client:
        await client.http.post("/v2/sessions", {}, headers=forged)

    managed, user = (request["headers"] for request in backend.requests)
    assert managed["Authorization"] == f"Bearer {API_KEY}"
    assert managed[MANAGED_USER_HEADER] == EXTERNAL_ID
    assert "Pine-Client" not in managed
    assert user["Authorization"] == "Bearer user-token"
    assert MANAGED_USER_HEADER not in user and "Pine-Client" not in user


def test_public_platform_names():
    assert API_KEY_PREFIXES == ("pine_sk_live_", "pine_sk_test_")
    assert MANAGED_USER_HEADER == "Pine-Managed-User"
    assert validate_external_id(EXTERNAL_ID) == EXTERNAL_ID
