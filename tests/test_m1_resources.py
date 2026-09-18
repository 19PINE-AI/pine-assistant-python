import asyncio
import json

import httpx
import pytest

from pine_assistant import AsyncPineAI, AuthError, PineAI, SessionError


def success(data):
    return {"status": "success", "data": data}


def session(session_id="7"):
    return {"id": session_id, "title": "A session", "state": "init"}


@pytest.mark.asyncio
async def test_async_m1_reads_use_typed_values_and_read_only_list_contract():
    seen = []

    async def handler(request):
        seen.append(request)
        if request.url.path.endswith("/auth/me"):
            return httpx.Response(200, json=success({"user_id": "42"}))
        if request.url.path.endswith("/sessions"):
            return httpx.Response(200, json=success({
                "sessions": [session()], "total": 1, "limit": 10, "offset": 0,
            }))
        return httpx.Response(200, json=success(session()))

    async with AsyncPineAI(
        access_token="user-a",
        base_url="https://pine.test",
        http_transport=httpx.MockTransport(handler),
    ) as client:
        assert (await client.auth.me()).user_id == "42"
        listed = await client.sessions.list(limit=10)
        assert listed.total == 1
        assert listed.sessions[0].id == "7"
        assert (await client.sessions.get(7)).id == "7"

    assert all(request.headers["Authorization"] == "Bearer user-a" for request in seen)
    assert dict(seen[1].url.params)["ensure_copilot"] == "false"
    assert all(request.url.path.startswith("/api/v2/") for request in seen)


@pytest.mark.asyncio
async def test_ticket_redeem_uses_service_bearer_only_for_that_request():
    requests = []

    async def handler(request):
        requests.append(request)
        if request.url.path.endswith("/tickets"):
            return httpx.Response(200, json=success({"ticket": "t" * 43, "expires_in": 120}))
        return httpx.Response(200, json=success({"access_token": "pine-token"}))

    async with AsyncPineAI(
        access_token="browser-token",
        base_url="https://pine.test",
        http_transport=httpx.MockTransport(handler),
    ) as client:
        ticket = await client.auth.mint_ticket(
            purpose="mcp", client_id="pine-mcp", auth_attempt_id="a" * 22, code_challenge="c" * 43,
        )
        assert ticket.expires_in == 120
        result = await client.auth.redeem_ticket(
            ticket=ticket.ticket,
            auth_attempt_id="a" * 22,
            code_verifier="v" * 43,
            service_token="service-token",
        )
        assert result.access_token == "pine-token"
        assert "pine-token" not in repr(result)

    assert requests[0].headers["Authorization"] == "Bearer browser-token"
    assert requests[1].headers["Authorization"] == "Bearer service-token"
    assert json.loads(requests[1].content) == {
        "ticket": "t" * 43, "auth_attempt_id": "a" * 22, "code_verifier": "v" * 43,
    }


@pytest.mark.asyncio
async def test_safe_errors_retain_status_and_code_without_upstream_body():
    async def handler(_request):
        return httpx.Response(403, json={
            "status": "error", "error": {"code": "unauthorized", "message": "email=a@pine.test"},
        })

    async with AsyncPineAI(base_url="https://pine.test", http_transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(AuthError) as excinfo:
            await client.auth.me()
    assert excinfo.value.status_code == 403
    assert excinfo.value.code == "unauthorized"
    assert "a@pine.test" not in str(excinfo.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [401, 500])
async def test_auth_statuses_are_retained(status_code):
    async def handler(_request):
        return httpx.Response(status_code, json={"status": "error", "error": {"code": "backend_code"}})

    async with AsyncPineAI(base_url="https://pine.test", http_transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(AuthError) as excinfo:
            await client.auth.me()
    assert excinfo.value.status_code == status_code
    assert excinfo.value.code == "backend_code"


@pytest.mark.asyncio
async def test_timeout_and_invalid_session_are_safe_errors():
    async def timeout_handler(_request):
        raise httpx.ReadTimeout("sensitive request URL")

    async with AsyncPineAI(base_url="https://pine.test", http_transport=httpx.MockTransport(timeout_handler)) as client:
        with pytest.raises(AuthError) as excinfo:
            await client.auth.me()
    assert excinfo.value.code == "timeout"
    assert "sensitive" not in str(excinfo.value)

    async def invalid_handler(_request):
        return httpx.Response(200, json=success({"id": "not-a-session"}))

    async with AsyncPineAI(base_url="https://pine.test", http_transport=httpx.MockTransport(invalid_handler)) as client:
        with pytest.raises(SessionError, match="invalid session"):
            await client.sessions.get("7")


@pytest.mark.asyncio
async def test_injected_client_cannot_redirect_or_leak_default_identity():
    seen = []

    async def handler(request):
        seen.append(request)
        return httpx.Response(302, headers={"Location": "https://attacker.test/identity"})

    injected = httpx.AsyncClient(
        base_url="https://wrong.test/other",
        headers={"Authorization": "Bearer leaked"},
        auth=httpx.BasicAuth("wrong", "wrong"),
        follow_redirects=True,
        transport=httpx.MockTransport(handler),
    )
    client = AsyncPineAI(access_token="right", base_url="https://pine.test", http_client=injected)
    with pytest.raises(AuthError) as excinfo:
        await client.auth.me()
    await client.aclose()
    assert not injected.is_closed
    assert seen[0].url == httpx.URL("https://pine.test/api/v2/auth/me")
    assert seen[0].headers["Authorization"] == "Bearer right"
    assert len(seen) == 1
    assert excinfo.value.status_code == 302
    await injected.aclose()


@pytest.mark.asyncio
async def test_disconnect_only_closes_realtime_and_aclose_closes_owned_http():
    async def handler(_request):
        return httpx.Response(200, json=success({"user_id": "1"}))

    class Socket:
        connected = True
        disconnected = False

        async def disconnect(self):
            self.disconnected = True
            self.connected = False

    client = AsyncPineAI(base_url="https://pine.test", http_transport=httpx.MockTransport(handler))
    socket = Socket()
    client._sio = socket  # exercise the production lifecycle boundary
    await client.disconnect()
    assert socket.disconnected
    assert not client.http._client.is_closed
    await client.aclose()
    assert client.http._client.is_closed


@pytest.mark.asyncio
async def test_aclose_closes_http_even_when_realtime_cleanup_fails():
    async def handler(_request):
        return httpx.Response(200, json=success({"user_id": "1"}))

    class BrokenSocket:
        connected = True

        async def disconnect(self):
            raise RuntimeError("socket cleanup failed")

    client = AsyncPineAI(base_url="https://pine.test", http_transport=httpx.MockTransport(handler))
    client._sio = BrokenSocket()
    with pytest.raises(RuntimeError, match="socket cleanup failed"):
        await client.aclose()
    assert client.http._client.is_closed


@pytest.mark.asyncio
async def test_concurrent_clients_keep_distinct_bearers():
    async def handler(request):
        return httpx.Response(200, json=success({"user_id": request.headers["Authorization"].split()[-1]}))

    shared = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    first = AsyncPineAI(access_token="101", base_url="https://pine.test", http_client=shared)
    second = AsyncPineAI(access_token="202", base_url="https://pine.test", http_client=shared)
    try:
        a, b = await asyncio.gather(first.auth.me(), second.auth.me())
        assert (a.user_id, b.user_id) == ("101", "202")
    finally:
        await first.aclose()
        await second.aclose()
        assert not shared.is_closed
        await shared.aclose()


def test_sync_resources_return_values_without_an_async_loop():
    seen = []

    def handler(request):
        seen.append(request)
        if request.url.path.endswith("/auth/me"):
            return httpx.Response(200, json=success({"user_id": 9}))
        return httpx.Response(200, json=success({
            "sessions": [session(8)], "total": 1, "limit": 1, "offset": 0,
        }))

    injected = httpx.Client(base_url="https://wrong.test", transport=httpx.MockTransport(handler))
    client = PineAI(access_token="sync-user", base_url="https://pine.test", http_client=injected)
    assert client.auth.me().user_id == "9"
    assert client.sessions.list(limit=1).sessions[0].id == "8"
    client.close()
    assert not injected.is_closed
    assert all(request.headers["Authorization"] == "Bearer sync-user" for request in seen)
    injected.close()


def test_sync_client_preserves_the_session_url_utility():
    session_id = "session-42"
    assert PineAI.session_url(session_id) == AsyncPineAI.session_url(session_id)
    assert PineAI.session_url(session_id) == "https://www.19pine.ai/app/chat/session-42"


@pytest.mark.asyncio
async def test_upload_uses_multipart_content_type(tmp_path):
    attachment = tmp_path / "note.txt"
    attachment.write_text("hello")

    async def handler(request):
        assert request.headers["Content-Type"].startswith("multipart/form-data; boundary=")
        assert b"hello" in request.content
        return httpx.Response(200, json=success([]))

    async with AsyncPineAI(
        access_token="user", base_url="https://pine.test", http_transport=httpx.MockTransport(handler),
    ) as client:
        assert await client.sessions.upload_attachment(str(attachment)) == []
