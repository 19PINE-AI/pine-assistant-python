"""Managed-user update, assistant, session config, identity catalog and the user's time zone."""

import json
import re
from datetime import datetime, timedelta, tzinfo
from datetime import timezone as dt_timezone
from zoneinfo import ZoneInfo

import httpx
import pytest

from pine_assistant import (
    MANAGED_USER_HEADER,
    AssistantIdentityCatalog,
    AsyncPineAI,
    ManagedUser,
    ManagedUserAssistant,
    PineAI,
    PlatformError,
    SessionConfig,
    SessionError,
)
from pine_assistant.chat import ChatEngine

API_KEY = "pine_sk_test_c2VjcmV0LWtleS1mb3ItdGVzdHM"
EXTERNAL_ID = "acme:user|42-x"
ENCODED_EXTERNAL_ID = "acme%3Auser%7C42-x"
PINE_USER_ID = "730194412688371"
BASE_URL = "https://pine.test"


def success(data):
    return {"status": "success", "data": data}


def managed_user(**overrides):
    return {
        "id": PINE_USER_ID,
        "external_id": EXTERNAL_ID,
        "email": "ada@example.com",
        "name": "Ada Lovelace",
        "phone": None,
        "assistant": {"identity_id": "agt_ava", "name": "Zoe Park"},
        "created_at": "2026-10-21T00:00:00Z",
        **overrides,
    }


CATALOG = {
    "version": "0123456789abcdef",
    "identities": [
        {"id": "agt_ava", "first_name": "Ava", "last_name": "Reed", "locale": "en-US", "status": "active"},
        {"id": "agt_old", "first_name": "Old", "status": "deprecated"},
    ],
}


class Recorder:
    def __init__(self, response: httpx.Response | None = None):
        self.requests: list[httpx.Request] = []
        self.response = response

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.response is not None:
            return self.response
        if request.url.path.startswith("/api/platform/"):
            return httpx.Response(200, json=success(managed_user()))
        if request.url.path.endswith("/config"):
            body = json.loads(request.content) if request.content else {}
            return httpx.Response(200, json=success(body))
        if request.url.path == "/api/v2/app/settings/assistance_identities":
            return httpx.Response(200, json=success(CATALOG))
        return httpx.Response(200, json=success({"message_id": "18", "status": "received"}))

    def body(self, index: int = -1):
        return json.loads(self.requests[index].content)


def async_client(recorder: Recorder, user_id: str | None = None) -> AsyncPineAI:
    return AsyncPineAI(
        api_key=API_KEY, user_id=user_id, base_url=BASE_URL, http_transport=httpx.MockTransport(recorder),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "body"),
    [
        ({}, {}),
        ({"phone": ""}, {"phone": ""}),
        ({"email": "new@example.com", "name": "Ada King"}, {"email": "new@example.com", "name": "Ada King"}),
        ({"phone": "+14155550100", "email": None}, {"phone": "+14155550100"}),
        ({"assistant": {}}, {"assistant": {}}),
        ({"assistant": {"name": "Zoe Park"}}, {"assistant": {"name": "Zoe Park"}}),
        (
            {"assistant": {"identity_id": "agt_ava", "name": "Zoe Park"}},
            {"assistant": {"identity_id": "agt_ava", "name": "Zoe Park"}},
        ),
    ],
)
async def test_update_sends_only_the_given_members_as_the_partner(kwargs, body):
    recorder = Recorder()
    async with async_client(recorder, user_id=PINE_USER_ID) as client:
        user = await client.platform.managed_users.update(EXTERNAL_ID, **kwargs)

    assert isinstance(user, ManagedUser)
    assert user.assistant == ManagedUserAssistant(identity_id="agt_ava", name="Zoe Park")
    [request] = recorder.requests
    assert request.method == "PATCH"
    assert request.url.raw_path == f"/api/platform/v1/managed-users/{ENCODED_EXTERNAL_ID}".encode()
    assert request.headers["Authorization"] == f"Bearer {API_KEY}"
    assert MANAGED_USER_HEADER not in request.headers
    assert recorder.body() == body


@pytest.mark.asyncio
async def test_create_sends_the_assistant():
    recorder = Recorder()
    async with async_client(recorder) as client:
        await client.platform.managed_users.create(
            EXTERNAL_ID, email="ada@example.com", name="Ada Lovelace", assistant={"identity_id": "agt_ava"},
        )

    assert recorder.body() == {
        "external_id": EXTERNAL_ID,
        "email": "ada@example.com",
        "name": "Ada Lovelace",
        "assistant": {"identity_id": "agt_ava"},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "assistant",
    [{"identity": "agt_ava"}, {"identity_id": None}, {"name": 7}, ["agt_ava"], "agt_ava"],
)
async def test_invalid_assistant_is_refused_before_any_request(assistant):
    recorder = Recorder()
    async with async_client(recorder) as client:
        with pytest.raises(ValueError, match="assistant must be a dict"):
            await client.platform.managed_users.update(EXTERNAL_ID, assistant=assistant)
        with pytest.raises(ValueError, match="assistant must be a dict"):
            await client.platform.managed_users.create(
                EXTERNAL_ID, email="ada@example.com", name="Ada", assistant=assistant,
            )

    assert recorder.requests == []


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"assistant": None}, None),
        ({"assistant": {"identity_id": "agt_ava", "name": "Ava Reed"}}, ManagedUserAssistant(identity_id="agt_ava", name="Ava Reed")),
    ],
)
def test_managed_user_assistant_parses(overrides, expected):
    assert ManagedUser.model_validate(managed_user(**overrides)).assistant == expected


def test_managed_user_without_assistant_member_still_parses():
    payload = managed_user()
    del payload["assistant"]
    assert ManagedUser.model_validate(payload).assistant is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "code"),
    [(400, "invalid_request"), (403, "platform_access_suspended"), (404, "platform_managed_user_not_found")],
)
async def test_update_errors_keep_code_and_status_without_upstream_body(status, code):
    recorder = Recorder(httpx.Response(
        status, json={"status": "error", "error": {"code": code, "message": f"{API_KEY} {EXTERNAL_ID}"}},
    ))
    async with async_client(recorder) as client:
        with pytest.raises(PlatformError) as excinfo:
            await client.platform.managed_users.update(EXTERNAL_ID, phone="")

    assert (excinfo.value.code, excinfo.value.status_code) == (code, status)
    assert API_KEY not in repr(excinfo.value)
    assert EXTERNAL_ID not in repr(excinfo.value)


def test_sync_update_and_config():
    recorder = Recorder()
    with PineAI(
        api_key=API_KEY, user_id=PINE_USER_ID, base_url=BASE_URL, http_transport=httpx.MockTransport(recorder),
    ) as client:
        user = client.platform.managed_users.update(EXTERNAL_ID, assistant={"name": "Zoe Park"})
        config = client.sessions.update_config(7, caller_id="platform")

    assert user.assistant is not None
    assert config == SessionConfig(caller_id="platform")
    assert [request.method for request in recorder.requests] == ["PATCH", "PUT"]
    assert MANAGED_USER_HEADER not in recorder.requests[0].headers
    assert recorder.requests[1].headers[MANAGED_USER_HEADER] == PINE_USER_ID


@pytest.mark.asyncio
@pytest.mark.parametrize("caller_id", ["user", "platform"])
async def test_update_config_puts_the_caller_id_as_the_managed_user(caller_id):
    recorder = Recorder()
    async with async_client(recorder, user_id=PINE_USER_ID) as client:
        config = await client.sessions.update_config("7", caller_id=caller_id)

    assert config.caller_id == caller_id
    [request] = recorder.requests
    assert (request.method, request.url.path) == ("PUT", "/api/v2/sessions/7/config")
    assert request.headers[MANAGED_USER_HEADER] == PINE_USER_ID
    assert recorder.body() == {"caller_id": caller_id}


@pytest.mark.asyncio
@pytest.mark.parametrize("caller_id", ["pine", "", None, "PLATFORM"])
async def test_invalid_caller_id_is_refused_before_any_request(caller_id):
    recorder = Recorder()
    async with async_client(recorder, user_id=PINE_USER_ID) as client:
        with pytest.raises(ValueError, match="caller_id"):
            await client.sessions.update_config(7, caller_id=caller_id)

    assert recorder.requests == []


@pytest.mark.asyncio
async def test_config_push_failure_keeps_code_and_status():
    recorder = Recorder(httpx.Response(
        503, json={"status": "error", "error": {"code": "session_config_push_failed", "message": "retry"}},
    ))
    async with async_client(recorder, user_id=PINE_USER_ID) as client:
        with pytest.raises(SessionError) as excinfo:
            await client.sessions.update_config(7, caller_id="platform")

    assert (excinfo.value.code, excinfo.value.status_code) == ("session_config_push_failed", 503)


@pytest.mark.asyncio
async def test_key_only_client_cannot_touch_session_config():
    recorder = Recorder()
    async with async_client(recorder) as client:
        with pytest.raises(ValueError):
            await client.sessions.get_config(7)

    assert recorder.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "caller_id", "extra"),
    [({}, None, {}), ({"caller_id": "platform"}, "platform", {}), ({"caller_id": "user", "debug": 1}, "user", {"debug": 1})],
)
async def test_get_config_parses_set_and_unset_values(payload, caller_id, extra):
    recorder = Recorder(httpx.Response(200, json=success(payload)))
    async with async_client(recorder, user_id=PINE_USER_ID) as client:
        config = await client.sessions.get_config(7)

    assert config.caller_id == caller_id
    assert config.model_extra == extra
    assert recorder.requests[0].method == "GET"


@pytest.mark.asyncio
async def test_key_only_client_reads_the_catalog_anonymously():
    recorder = Recorder()
    async with async_client(recorder) as client:
        catalog = await client.app_settings.assistant_identities()

    assert isinstance(catalog, AssistantIdentityCatalog)
    assert catalog.version == "0123456789abcdef"
    assert [(identity.id, identity.status) for identity in catalog.identities] == [
        ("agt_ava", "active"), ("agt_old", "deprecated"),
    ]
    [request] = recorder.requests
    assert request.url.path == "/api/v2/app/settings/assistance_identities"
    assert request.url.params["identities"] == "catalog"
    assert "Authorization" not in request.headers
    assert MANAGED_USER_HEADER not in request.headers


def test_managed_user_client_reads_the_catalog_as_the_user():
    recorder = Recorder()
    with PineAI(
        api_key=API_KEY, user_id=PINE_USER_ID, base_url=BASE_URL, http_transport=httpx.MockTransport(recorder),
    ) as client:
        catalog = client.app_settings.assistant_identities()

    assert catalog.identities[0].id == "agt_ava"
    [request] = recorder.requests
    assert request.headers["Authorization"] == f"Bearer {API_KEY}"
    assert request.headers[MANAGED_USER_HEADER] == PINE_USER_ID


RFC3339_SECONDS = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}")


class NoOffset(tzinfo):
    def utcoffset(self, dt):
        return None

    def dst(self, dt):
        return None


def assert_now_in(value, zone):
    assert RFC3339_SECONDS.fullmatch(value)
    sent = datetime.fromisoformat(value)
    assert sent.utcoffset() == datetime.now(zone).utcoffset()
    assert abs((datetime.now(zone) - sent).total_seconds()) < 5


@pytest.mark.parametrize(
    ("timezone", "offset"),
    [
        ("Asia/Tokyo", "+09:00"),
        ("Asia/Kolkata", "+05:30"),
        ("UTC", "+00:00"),
        (dt_timezone(timedelta(hours=-4)), "-04:00"),
        (ZoneInfo("Asia/Shanghai"), "+08:00"),
    ],
)
@pytest.mark.asyncio
async def test_rest_message_carries_the_current_time_in_the_timezone(timezone, offset):
    recorder = Recorder()
    async with async_client(recorder, user_id=PINE_USER_ID) as client:
        await client.sessions.send_message(7, "hi", timezone=timezone)

    sent = recorder.body()["client_now_date"]
    assert sent.endswith(offset)
    assert_now_in(sent, timezone if isinstance(timezone, tzinfo) else ZoneInfo(timezone))


def test_daylight_saving_zone_uses_its_current_offset():
    recorder = Recorder()
    with PineAI(
        api_key=API_KEY, user_id=PINE_USER_ID, base_url=BASE_URL, http_transport=httpx.MockTransport(recorder),
    ) as client:
        client.sessions.send_message(7, "hi", timezone="America/New_York")

    assert_now_in(recorder.body()["client_now_date"], ZoneInfo("America/New_York"))


@pytest.mark.asyncio
@pytest.mark.parametrize("timezone", ["Mars/Olympus", "", "../etc/passwd", "EST5EDT/x", 330, NoOffset()])
async def test_invalid_timezone_is_refused_before_any_request(timezone):
    recorder = Recorder()
    async with async_client(recorder, user_id=PINE_USER_ID) as client:
        with pytest.raises(ValueError, match="timezone"):
            await client.sessions.send_message(7, "hi", timezone=timezone)

    assert recorder.requests == []


@pytest.mark.asyncio
async def test_rest_message_omits_client_now_date_without_timezone():
    recorder = Recorder()
    async with async_client(recorder, user_id=PINE_USER_ID) as client:
        await client.sessions.send_message(7, "hi")

    assert recorder.body() == {"content": "hi"}


def test_socketio_message_carries_client_now_date_only_with_a_timezone():
    assert ChatEngine._build_message_data("hi") == {"content": "hi", "attachments": [], "referenced_sessions": []}
    assert_now_in(ChatEngine._build_message_data("hi", timezone="Asia/Tokyo")["client_now_date"], ZoneInfo("Asia/Tokyo"))
    with pytest.raises(ValueError, match="timezone"):
        ChatEngine._build_message_data("hi", timezone="Mars/Olympus")
