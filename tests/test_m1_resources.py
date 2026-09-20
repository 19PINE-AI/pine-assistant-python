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
async def test_async_session_message_serializes_supported_fields_and_preserves_statuses():
    requests = []
    statuses = iter(["received", "delivered", "delivery_failed"])

    async def handler(request):
        requests.append(request)
        return httpx.Response(200, json=success({
            "message_id": "18", "request_id": "click-18", "status": next(statuses), "revision": "24",
        }))

    async with AsyncPineAI(
        access_token="user", base_url="https://pine.test", http_transport=httpx.MockTransport(handler),
    ) as client:
        sent = []
        for _ in range(3):
            sent.append(await client.sessions.send_message(
                7,
                "Continue",
                type="plain_text",
                action={"action_type": "engage_call", "payload": {"number": "555"}},
                attachments=[{"id": "upload-1"}],
                library_artifact_ids=["artifact-1"],
                referenced_sessions=[{"session_id": "6", "title": "Earlier work"}],
                quote={"text": "The prior quote"},
                client_now_date="2026-09-18",
                request_id="click-18",
            ))

    assert [result.status for result in sent] == ["received", "delivered", "delivery_failed"]
    assert all((result.message_id, result.request_id, result.revision) == ("18", "click-18", "24") for result in sent)
    assert len(requests) == 3
    assert all(request.method == "POST" and request.url.path == "/api/v2/sessions/7/messages" for request in requests)
    assert all(request.headers["X-Request-ID"] == "click-18" for request in requests)
    assert json.loads(requests[0].content) == {
        "content": "Continue",
        "type": "plain_text",
        "action": {"action_type": "engage_call", "payload": {"number": "555"}},
        "attachments": [{"id": "upload-1"}],
        "library_artifact_ids": ["artifact-1"],
        "referenced_sessions": [{"session_id": "6", "title": "Earlier work"}],
        "quote": {"text": "The prior quote"},
        "client_now_date": "2026-09-18",
    }


@pytest.mark.asyncio
async def test_async_session_outcomes_preserve_the_full_page_and_cursor():
    async def handler(request):
        assert request.url.path == "/api/v2/sessions/7/outcomes"
        assert dict(request.url.params) == {"limit": "2", "before": "18"}
        return httpx.Response(200, json=success({
            "items": [{
                "outcome_id": "17", "session_id": "7", "kind": "positive", "importance": "major",
                "milestone_fact": "A call was booked", "evidence_excerpts": ["confirmed"], "brief": "Booked.",
                "story_description": "Pine called the provider.", "outcome_narrative": "Pine booked the call.",
                "share_draft": "Call booked", "engage_prompt": "Discuss the booking",
                "engage_call_objective": "Explain the confirmation", "estimated_time_saved_min": 12,
                "previous_briefs": ["Earlier update"], "created_at": "2026-09-18T01:02:03Z",
                "rating": {"stars": 5, "reasons": ["clear"], "comment": "thanks", "rated_at": "2026-09-18T02:03:04Z"},
                "source": "agent", "type": "session:outcome_updated", "id": "17",
            }],
            "next_cursor": "17", "total": 3,
        }))

    async with AsyncPineAI(base_url="https://pine.test", http_transport=httpx.MockTransport(handler)) as client:
        page = await client.sessions.outcomes("7", limit=2, before=18)

    outcome = page.items[0]
    assert (page.next_cursor, page.total) == ("17", 3)
    assert outcome.outcome_narrative == "Pine booked the call."
    assert outcome.rating and outcome.rating.stars == 5
    assert outcome.model_extra == {"source": "agent", "type": "session:outcome_updated", "id": "17"}


@pytest.mark.asyncio
async def test_session_message_errors_keep_codes_and_sanitize_input_state_without_retrying():
    attempts = 0

    async def conflict_handler(_request):
        nonlocal attempts
        attempts += 1
        return httpx.Response(409, json={
            "status": "error",
            "data": {"content": "input disabled", "detail": "task is processing", "code": "task_processing"},
            "debug": "sensitive upstream body",
        })

    async with AsyncPineAI(base_url="https://pine.test", http_transport=httpx.MockTransport(conflict_handler)) as client:
        with pytest.raises(SessionError) as excinfo:
            await client.sessions.send_message("7", "Continue")
    assert attempts == 1
    assert excinfo.value.status_code == 409
    assert excinfo.value.code == "task_processing"
    assert excinfo.value.details == {"input_state": {
        "content": "input disabled", "detail": "task is processing", "code": "task_processing",
    }}
    assert "sensitive" not in str(excinfo.value)

    async def timeout_handler(_request):
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("sensitive request URL")

    async with AsyncPineAI(base_url="https://pine.test", http_transport=httpx.MockTransport(timeout_handler)) as client:
        with pytest.raises(SessionError) as timeout:
            await client.sessions.send_message("7", "Continue")
    assert attempts == 2
    assert timeout.value.code == "timeout"
    assert "sensitive" not in str(timeout.value)


@pytest.mark.asyncio
async def test_async_end_task_uses_close_contract_and_keeps_typed_errors_without_retrying():
    requests = []

    async def success_handler(request):
        requests.append(request)
        assert request.method == "POST"
        assert request.url.path == "/api/v2/sessions/7/close"
        return httpx.Response(
            200,
            json=success({"session": session("7") | {"state": "task_finished", "finished_status": "user_closed"}}),
        )

    async with AsyncPineAI(base_url="https://pine.test", http_transport=httpx.MockTransport(success_handler)) as client:
        closed = await client.sessions.end_task(7)
        with pytest.raises(ValueError):
            await client.sessions.end_task(True)
    assert closed.id == "7"
    assert closed.state == "task_finished"
    assert closed.finished_status == "user_closed"
    assert len(requests) == 1

    async def rejected_handler(_request):
        return httpx.Response(403, json={"status": "error", "error": {"code": "session_access_denied"}})

    async with AsyncPineAI(base_url="https://pine.test", http_transport=httpx.MockTransport(rejected_handler)) as client:
        with pytest.raises(SessionError) as rejected:
            await client.sessions.end_task("7")
    assert (rejected.value.code, rejected.value.status_code) == ("session_access_denied", 403)

    async def timeout_handler(_request):
        raise httpx.ReadTimeout("sensitive request URL")

    async with AsyncPineAI(base_url="https://pine.test", http_transport=httpx.MockTransport(timeout_handler)) as client:
        with pytest.raises(SessionError) as timeout:
            await client.sessions.end_task("7")
    assert timeout.value.code == "timeout"
    assert "sensitive" not in str(timeout.value)


def test_sync_end_task_uses_close_contract_and_validates_session_id():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json=success({"session": session("7") | {"state": "task_finished", "finished_status": "user_closed"}}),
        )

    with PineAI(base_url="https://pine.test", http_transport=httpx.MockTransport(handler)) as client:
        closed = client.sessions.end_task("7")
        with pytest.raises(ValueError):
            client.sessions.end_task("bad")
    assert closed.id == "7"
    assert closed.finished_status == "user_closed"
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert requests[0].url.path == "/api/v2/sessions/7/close"


def test_sync_session_send_and_outcomes_match_async_rest_contract():
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path.endswith("/messages"):
            return httpx.Response(200, json=success({
                "message_id": "18", "request_id": "sync-18", "status": "delivered", "revision": "24",
            }))
        return httpx.Response(200, json=success({
            "items": [{"outcome_id": "17", "session_id": "7", "milestone_fact": "Booked", "created_at": "now"}],
            "total": 1,
        }))

    with PineAI(access_token="user", base_url="https://pine.test", http_transport=httpx.MockTransport(handler)) as client:
        sent = client.sessions.send_message("7", "Continue", request_id="sync-18")
        page = client.sessions.outcomes("7")

    assert (sent.status, sent.message_id, sent.revision) == ("delivered", "18", "24")
    assert page.items[0].milestone_fact == "Booked"
    assert requests[0].headers["X-Request-ID"] == "sync-18"
    assert json.loads(requests[0].content) == {"content": "Continue"}
    assert dict(requests[1].url.params) == {"limit": "20"}


@pytest.mark.parametrize("session_id", [0, "bad", True])
def test_session_message_and_outcomes_validate_session_ids(session_id):
    client = PineAI(base_url="https://pine.test", http_transport=httpx.MockTransport(lambda _request: None))
    try:
        with pytest.raises(ValueError):
            client.sessions.send_message(session_id, "Continue")
        with pytest.raises(ValueError):
            client.sessions.outcomes(session_id)
    finally:
        client.close()


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
