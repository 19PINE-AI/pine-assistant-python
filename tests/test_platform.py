"""Platform API and managed-user mode against a fake backend on 127.0.0.1."""

import asyncio
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
    AuthError,
    ConnectionError,
    ManagedUser,
    PineAI,
    PlatformError,
    SessionError,
    validate_external_id,
    validate_user_id,
)
from pine_assistant.transport.http import SyncHttpClient

API_KEY = "pine_sk_test_c2VjcmV0LWtleS1mb3ItdGVzdHM"
EXTERNAL_ID = "acme:user|42-x"
ENCODED_EXTERNAL_ID = "acme%3Auser%7C42-x"
PINE_USER_ID = "730194412688371"


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
    envelopes: list[dict[str, Any]] = field(default_factory=list)
    error_status: int | None = None
    error_code: str = "platform_failure"
    existing: bool = False
    refuse_handshake: bool = False

    def app(self) -> web.Application:
        sio = socketio.AsyncServer(async_mode="aiohttp")
        app = web.Application()
        sio.attach(app, socketio_path="/api/v2/socket.io/")

        @sio.event
        async def connect(sid, environ, auth):
            self.handshakes.append(auth)
            if self.refuse_handshake:
                # Like the Pine backend: accept the connection, then close it
                # with no connect_error and no code.
                async def close():
                    await asyncio.sleep(0.05)
                    await sio.disconnect(sid)

                sio.start_background_task(close)
            else:
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
                    {"status": "error", "error": {"code": self.error_code, "message": f"{API_KEY} {EXTERNAL_ID} {PINE_USER_ID}"}},
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

        async def read_history(request):
            return await record(request) or web.json_response(
                success({"messages": [{"message_id": "9", "type": "session:text", "version": 2, "data": "{}"}]})
            )

        app.router.add_post("/api/platform/v1/managed-users", create_user)
        app.router.add_get("/api/platform/v1/managed-users/{external_id}", get_user)
        app.router.add_get("/api/v2/auth/me", auth_me)
        app.router.add_get("/api/v2/sessions", sessions)
        app.router.add_get("/api/v2/sessions/{id}/messages", read_history)
        app.router.add_post("/api/v2/sessions", sessions)
        app.router.add_post("/api/v2/auth/tickets", sessions)
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
    assert PINE_USER_ID not in text


@pytest.mark.asyncio
async def test_managed_user_rest_requests_carry_the_key_and_the_header(backend):
    async with AsyncPineAI(api_key=API_KEY, user_id=PINE_USER_ID, base_url=backend.base_url) as client:
        listed = await client.sessions.list()

    assert listed.sessions[0].id == "7"
    [request] = backend.requests
    assert request["headers"]["Authorization"] == f"Bearer {API_KEY}"
    assert request["headers"]["Pine-Managed-User-Id"] == PINE_USER_ID
    assert "Pine-Managed-User" not in request["headers"]


@pytest.mark.asyncio
async def test_managed_user_reads_session_history_over_rest(backend):
    async with AsyncPineAI(api_key=API_KEY, user_id=PINE_USER_ID, base_url=backend.base_url) as client:
        page = await client.sessions.history("7")

    assert [message.message_id for message in page.messages] == ["9"]
    [request] = backend.requests
    assert request["path"] == "/api/v2/sessions/7/messages?limit=30"
    assert request["headers"]["Authorization"] == f"Bearer {API_KEY}"
    assert request["headers"][MANAGED_USER_HEADER] == PINE_USER_ID


@pytest.mark.asyncio
async def test_managed_user_connect_sends_the_id_in_handshake_and_envelope_without_auth_me(backend):
    async with AsyncPineAI(api_key=API_KEY, user_id=PINE_USER_ID, base_url=backend.base_url) as client:
        await client.connect()
        await client.get_history("7")
        await client.disconnect()
        await client.connect()

    assert backend.requests == []
    assert backend.handshakes == [{"token": API_KEY, "managed_user_id": PINE_USER_ID}] * 2
    assert all(isinstance(auth["managed_user_id"], str) for auth in backend.handshakes)
    assert backend.envelopes[0]["metadata"]["source"]["user_id"] == PINE_USER_ID


@pytest.mark.asyncio
async def test_user_token_handshake_is_unchanged(backend):
    async with AsyncPineAI(access_token="user-token", user_id="17", base_url=backend.base_url) as client:
        await client.connect()
    assert backend.handshakes == [{"token": "user-token"}]
    assert backend.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("existing", "phone"), [(False, "+14155550100"), (True, None)])
async def test_create_managed_user_is_typed_and_acts_as_the_partner(backend, existing, phone):
    backend.existing = existing
    async with AsyncPineAI(api_key=API_KEY, user_id="999", base_url=backend.base_url) as client:
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
    assert MANAGED_USER_HEADER not in request["headers"]


@pytest.mark.asyncio
async def test_get_managed_user_percent_encodes_the_external_id(backend):
    async with AsyncPineAI(api_key=API_KEY, base_url=backend.base_url) as client:
        user = await client.platform.managed_users.get(EXTERNAL_ID)

    assert user.id == PINE_USER_ID
    assert backend.requests[0]["path"] == f"/api/platform/v1/managed-users/{ENCODED_EXTERNAL_ID}"
    assert MANAGED_USER_HEADER not in backend.requests[0]["headers"]


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
@pytest.mark.parametrize(
    ("kwargs", "hint"),
    [
        ({"api_key": API_KEY, "user_id": PINE_USER_ID}, "Check api_key and user_id."),
        ({"access_token": "user-token", "user_id": "17"}, "re-run the auth flow"),
    ],
)
async def test_refused_handshake_raises_connection_error_with_a_credentials_hint(backend, kwargs, hint):
    backend.refuse_handshake = True
    async with AsyncPineAI(base_url=backend.base_url, ready_timeout=0.5, **kwargs) as client:
        with pytest.raises(ConnectionError) as excinfo:
            await client.connect()
        assert not client.connected

    assert len(backend.handshakes) == 1
    message = str(excinfo.value)
    assert "no 'ready' event after 0.5s" in message
    assert hint in message
    _assert_no_secrets(message)


@pytest.mark.asyncio
async def test_key_only_client_refuses_user_scoped_calls_before_any_request(backend):
    async with AsyncPineAI(api_key=API_KEY, base_url=backend.base_url) as client:
        for call in (client.sessions.list, client.auth.me, client.connect):
            with pytest.raises(ValueError) as excinfo:
                await call()
            assert "pass user_id" in str(excinfo.value)
            _assert_no_secrets(str(excinfo.value))
        await client.platform.managed_users.get(EXTERNAL_ID)

    with PineAI(api_key=API_KEY, base_url=backend.base_url) as client, pytest.raises(ValueError) as excinfo:
        client.sessions.list()
    assert "pass user_id" in str(excinfo.value)

    assert [request["path"] for request in backend.requests] == [
        f"/api/platform/v1/managed-users/{ENCODED_EXTERNAL_ID}",
    ]
    assert backend.handshakes == []


@pytest.mark.asyncio
@pytest.mark.parametrize("override", [{"access_token": "user-token"}, {"user_id": "17"}, {"user_id": PINE_USER_ID}])
async def test_key_mode_connect_refuses_identity_overrides(backend, override):
    async with AsyncPineAI(api_key=API_KEY, user_id=PINE_USER_ID, base_url=backend.base_url) as client:
        with pytest.raises(ValueError) as excinfo:
            await client.connect(**override)
    _assert_no_secrets(str(excinfo.value))
    assert backend.handshakes == []


INVALID_USER_IDS = ["0", "01", "+1", " 1", "1 ", "1\n", "", "9223372036854775808", "\u0661", "usr_1", 42, True, None]


@pytest.mark.parametrize("value", INVALID_USER_IDS)
def test_validate_user_id_rejects_non_canonical_ids_without_echoing(value):
    with pytest.raises(ValueError) as excinfo:
        validate_user_id(value)
    message = str(excinfo.value)
    assert 'user_id must be a string such as "123456789"' in message
    if value not in ("", None, True):
        assert str(value) not in message


@pytest.mark.parametrize("value", ["1", "123456789", PINE_USER_ID, "9223372036854775807"])
def test_validate_user_id_accepts_canonical_ids(value):
    assert validate_user_id(value) == value


@pytest.mark.parametrize(
    "kwargs",
    [
        {"api_key": API_KEY, "user_id": PINE_USER_ID, "access_token": "user-token"},
        {"api_key": API_KEY, "access_token": "user-token"},
        {"api_key": "not-a-pine-key", "user_id": PINE_USER_ID},
        {"api_key": "pine_sk_prod_c2VjcmV0", "user_id": PINE_USER_ID},
        {"api_key": "pine_sk_c2VjcmV0"},
        {"api_key": API_KEY, "user_id": EXTERNAL_ID},
        {"api_key": API_KEY, "user_id": "0730194412688371"},
        {"api_key": API_KEY, "user_id": "9223372036854775808"},
        {"api_key": API_KEY, "user_id": 730194412688371},
        {"api_key": API_KEY, "user_id": True},
        {"access_token": API_KEY},
        {"access_token": API_KEY, "user_id": PINE_USER_ID},
    ],
)
def test_invalid_platform_configuration_is_rejected_without_echoing_values(kwargs):
    for client_type in (AsyncPineAI, PineAI):
        with pytest.raises(ValueError) as excinfo:
            client_type(base_url="https://pine.test", **kwargs)
        _assert_no_secrets(str(excinfo.value))
        for value in kwargs.values():
            if value is not True:
                assert str(value) not in str(excinfo.value)


@pytest.mark.parametrize("removed", [{"managed_user": EXTERNAL_ID}, {"client_name": "mcp"}])
def test_removed_constructor_arguments_are_gone(removed):
    for client_type in (AsyncPineAI, PineAI):
        with pytest.raises(TypeError):
            client_type(api_key=API_KEY, user_id=PINE_USER_ID, **removed)


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
        headers={MANAGED_USER_HEADER: "1"}, transport=httpx.MockTransport(handler),
    )
    async with AsyncPineAI(access_token="user-token", base_url="https://pine.test", http_client=injected) as client:
        await client.auth.me()
    async with AsyncPineAI(
        api_key=API_KEY, user_id=PINE_USER_ID, base_url="https://pine.test", http_client=injected,
    ) as client:
        await client.auth.me()
        await client.platform.managed_users.get(EXTERNAL_ID)
    await injected.aclose()

    assert MANAGED_USER_HEADER not in seen[0].headers
    assert seen[1].headers[MANAGED_USER_HEADER] == PINE_USER_ID
    assert MANAGED_USER_HEADER not in seen[2].headers


def test_transport_takes_key_mode_from_api_key_not_from_the_token_text():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=success({"sessions": [], "total": 0, "limit": 30, "offset": 0}))

    transport = httpx.MockTransport(handler)
    token_client = SyncHttpClient(base_url="https://pine.test", token=API_KEY, transport=transport)
    with pytest.raises(ValueError, match="needs a client built with api_key"):
        token_client.get("/platform/v1/managed-users/x", as_partner=True)
    token_client.get("/v2/sessions")
    token_client.set_token("user-jwt")

    key_client = SyncHttpClient(base_url="https://pine.test", api_key=API_KEY, transport=transport)
    with pytest.raises(ValueError, match="pass user_id"):
        key_client.get("/v2/sessions")
    with pytest.raises(ValueError, match="cannot change its token"):
        key_client.set_token("user-jwt")
    with pytest.raises(ValueError, match="either token or api_key"):
        SyncHttpClient(base_url="https://pine.test", token="user-jwt", api_key=API_KEY, transport=transport)

    [request] = seen
    assert request.headers["Authorization"] == f"Bearer {API_KEY}"
    assert MANAGED_USER_HEADER not in request.headers


def test_sync_client_supports_platform_and_managed_user_requests():
    seen = []

    def handler(request):
        seen.append(request)
        if request.url.path.startswith("/api/platform/"):
            return httpx.Response(201, json=success(managed_user()))
        return httpx.Response(200, json=success({"sessions": [], "total": 0, "limit": 30, "offset": 0}))

    with PineAI(
        api_key=API_KEY, user_id=PINE_USER_ID, base_url="https://pine.test",
        http_transport=httpx.MockTransport(handler),
    ) as client:
        created = client.platform.managed_users.create(EXTERNAL_ID, email="ada@example.com", name="Ada Lovelace")
        client.sessions.list()

    assert created.id == PINE_USER_ID
    assert seen[0].url.raw_path == b"/api/platform/v1/managed-users"
    assert MANAGED_USER_HEADER not in seen[0].headers
    assert seen[1].headers[MANAGED_USER_HEADER] == PINE_USER_ID
    assert all(request.headers["Authorization"] == f"Bearer {API_KEY}" for request in seen)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "code"),
    [
        (400, "invalid_request"),
        (401, "unauthorized"),
        (402, "platform_spend_limit_reached"),
        (403, "platform_access_suspended"),
        (403, "platform_managed_user_unavailable"),
        (403, "platform_route_not_allowed"),
        (404, "platform_managed_user_not_found"),
        (409, "platform_session_limit_reached"),
        (409, "platform_session_state_conflict"),
        (429, "too_many_requests"),
    ],
)
async def test_user_scoped_failures_keep_code_and_status_without_upstream_body(backend, status, code):
    backend.error_status, backend.error_code = status, code
    async with AsyncPineAI(api_key=API_KEY, user_id=PINE_USER_ID, base_url=backend.base_url) as client:
        with pytest.raises(SessionError) as excinfo:
            await client.sessions.list()
    assert (excinfo.value.code, excinfo.value.status_code) == (code, status)
    _assert_no_secrets(str(excinfo.value))


@pytest.mark.asyncio
async def test_managed_user_is_never_sent_with_a_token_other_than_the_key(backend):
    async with AsyncPineAI(api_key=API_KEY, user_id=PINE_USER_ID, base_url=backend.base_url) as client:
        await client.http.get("/v2/sessions", token=API_KEY)
        await client.http.get("/v2/sessions", token="other-jwt")
        await client.http.get("/v2/sessions", token="pine_sk_test_another-key")

    first, second, third = (request["headers"] for request in backend.requests)
    assert first["Authorization"] == f"Bearer {API_KEY}" and first[MANAGED_USER_HEADER] == PINE_USER_ID
    assert second["Authorization"] == "Bearer other-jwt" and MANAGED_USER_HEADER not in second
    assert third["Authorization"] == "Bearer pine_sk_test_another-key" and MANAGED_USER_HEADER not in third


@pytest.mark.asyncio
@pytest.mark.parametrize("user_id", [PINE_USER_ID, None])
async def test_key_client_refuses_a_user_token_before_any_request(backend, user_id):
    async with AsyncPineAI(api_key=API_KEY, user_id=user_id, base_url=backend.base_url) as client:
        with pytest.raises(ValueError) as set_token:
            client.http.set_token("user-jwt")
        with pytest.raises(ValueError) as verified:
            await client.auth.verify_code("ada@example.com", "1234", "request-token")
    with PineAI(api_key=API_KEY, user_id=user_id, base_url=backend.base_url) as client, pytest.raises(ValueError):
        client.auth.verify_code("ada@example.com", "1234", "request-token")

    for excinfo in (set_token, verified):
        assert "cannot change its token" in str(excinfo.value)
        _assert_no_secrets(str(excinfo.value))
    assert backend.requests == []


@pytest.mark.asyncio
async def test_platform_calls_need_a_key_client(backend):
    async with AsyncPineAI(access_token="user-token", user_id="17", base_url=backend.base_url) as client:
        with pytest.raises(ValueError) as fetched:
            await client.platform.managed_users.get(EXTERNAL_ID)
        with pytest.raises(ValueError) as created:
            await client.platform.managed_users.create(EXTERNAL_ID, email="a@example.com", name="A")
    for excinfo in (fetched, created):
        assert "needs a client built with api_key" in str(excinfo.value)
    with PineAI(access_token="user-token", base_url=backend.base_url) as client, pytest.raises(ValueError):
        client.platform.managed_users.get(EXTERNAL_ID)
    assert backend.requests == []


@pytest.mark.asyncio
async def test_caller_headers_cannot_override_identity(backend):
    forged = {"Authorization": "Bearer forged", MANAGED_USER_HEADER: "1"}
    async with AsyncPineAI(api_key=API_KEY, user_id=PINE_USER_ID, base_url=backend.base_url) as client:
        await client.http.post("/v2/sessions", {}, headers=forged)
    async with AsyncPineAI(access_token="user-token", base_url=backend.base_url) as client:
        await client.http.post("/v2/sessions", {}, headers=forged)

    managed, user = (request["headers"] for request in backend.requests)
    assert managed["Authorization"] == f"Bearer {API_KEY}"
    assert managed[MANAGED_USER_HEADER] == PINE_USER_ID
    assert user["Authorization"] == "Bearer user-token"
    assert MANAGED_USER_HEADER not in user


def test_public_platform_names():
    assert API_KEY_PREFIXES == ("pine_sk_live_", "pine_sk_test_")
    assert MANAGED_USER_HEADER == "Pine-Managed-User-Id"
    assert validate_external_id(EXTERNAL_ID) == EXTERNAL_ID
    assert validate_user_id(PINE_USER_ID) == PINE_USER_ID


AUTH_ATTEMPT_ID = "a" * 22
CODE_CHALLENGE = "c" * 43


@pytest.mark.asyncio
async def test_documented_managed_user_restrictions_raise_the_resource_error_type(backend):
    async with AsyncPineAI(api_key=API_KEY, user_id=PINE_USER_ID, base_url=backend.base_url) as client:
        backend.error_status, backend.error_code = 403, "platform_route_not_allowed"
        with pytest.raises(AuthError) as minted:
            await client.auth.mint_ticket(
                purpose="mcp", client_id="client", auth_attempt_id=AUTH_ATTEMPT_ID, code_challenge=CODE_CHALLENGE,
            )
        backend.error_status, backend.error_code = 404, "platform_managed_user_not_found"
        with pytest.raises(SessionError) as listed:
            await client.sessions.list()
        with pytest.raises(AuthError) as identity:
            await client.auth.me()

    assert (minted.value.code, minted.value.status_code) == ("platform_route_not_allowed", 403)
    for excinfo in (listed, identity):
        assert (excinfo.value.code, excinfo.value.status_code) == ("platform_managed_user_not_found", 404)
        _assert_no_secrets(str(excinfo.value))
