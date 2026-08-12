"""Live tests against the real Pine AI service.

Excluded from CI: they need a token and every run spends credits. Their job is
to check the supported surface against a running server, and to be the source
the fixtures are recorded from.

    PINE_INTEGRATION=1 PINE_ACCESS_TOKEN=... PINE_USER_ID=... pytest tests/integration -v
"""

import asyncio
import contextlib
import os

import pytest

from pine_assistant import AsyncPineAI, S2CEvent, is_supported_event

SKIP = not os.environ.get("PINE_INTEGRATION")
ACCESS_TOKEN = os.environ.get("PINE_ACCESS_TOKEN", "")
USER_ID = os.environ.get("PINE_USER_ID", "")
BASE_URL = os.environ.get("PINE_BASE_URL", "https://www.19pine.ai")

pytestmark = pytest.mark.skipif(SKIP, reason="PINE_INTEGRATION not set")

PROMPT = "Ask what time Saturday's dinner starts."

# NANP reserves 555-0100 through 555-0199 for fictional use — nothing routes
# there, so the call fails without reaching anyone.
UNROUTABLE_NUMBER = "+1 415-555-0199"
CALL_PROMPT = (
    f"Please call {UNROUTABLE_NUMBER} and ask what time Saturday's dinner starts. "
    "Make exactly one attempt — if it does not connect, stop and tell me. Do not retry."
)
CALL_TIMEOUT_S = 300.0


def make_client() -> AsyncPineAI:
    return AsyncPineAI(access_token=ACCESS_TOKEN, user_id=USER_ID, base_url=BASE_URL)


@pytest.fixture
async def session():
    """A connected client on a fresh session, torn down afterwards."""
    client = make_client()
    await client.connect()
    created = await client.sessions.create()
    sid = created["id"]
    await client.join_session(sid)
    try:
        yield client, sid
    finally:
        client.leave_session(sid)
        with contextlib.suppress(Exception):
            await client.sessions.delete(sid)
        await client.disconnect()


class TestConnection:
    async def test_connects_and_receives_ready(self):
        client = make_client()
        await client.connect()
        assert client.connected
        await client.disconnect()

    async def test_rejects_an_invalid_token(self):
        client = AsyncPineAI(access_token="invalid", user_id=USER_ID, base_url=BASE_URL)
        with pytest.raises(Exception):
            await client.connect()


class TestSessionLifecycle:
    async def test_create_list_get_delete(self):
        client = make_client()
        created = await client.sessions.create()
        sid = created["id"]
        assert created["state"] == "init"

        listed = await client.sessions.list(limit=50)
        assert sid in [s["id"] for s in listed["sessions"]]

        fetched = await client.sessions.get(sid)
        assert fetched["id"] == sid
        # Expiry lives here and nowhere on the Socket.IO surface.
        assert "is_stale" in fetched

        await client.sessions.delete(sid)


class TestSupportedSurface:
    async def test_a_turn_produces_a_substantive_response(self, session):
        """Streamed text, a complete message, a rich document, or a form.

        A turn often ends on streaming increments alone: the composer reopens
        once the agent has finished speaking, and the complete `session:text`
        is the durable record, read back from history rather than awaited here.
        """
        client, sid = session
        events = [e async for e in client.chat(sid, PROMPT)]

        types = {e.type for e in events}
        assert types & {
            S2CEvent.SESSION_TEXT.value,
            S2CEvent.SESSION_TEXT_PART.value,
            S2CEvent.SESSION_RICH_CONTENT.value,
            S2CEvent.SESSION_FORM_TO_USER.value,
        }, f"no substantive response; saw {sorted(types)}"

    async def test_unsupported_events_arrive_without_breaking_the_turn(self, session):
        """The server keeps emitting outside the scope, and the SDK keeps
        handing those events over rather than failing on them."""
        client, sid = session
        events = [e async for e in client.chat(sid, PROMPT)]

        unsupported = sorted({e.type for e in events if not is_supported_event(e.type)})
        print(f"  unsupported events observed: {unsupported}")
        assert events, "the turn produced nothing at all"

    async def test_rebuild_returns_the_conversation(self, session):
        client, sid = session
        async for _ in client.chat(sid, PROMPT):
            pass

        messages = await client.rebuild(sid)
        assert isinstance(messages, list)
        assert messages, "history came back empty after a turn"


class TestErrors:
    async def test_get_nonexistent_session(self):
        client = make_client()
        with pytest.raises(Exception):
            await client.sessions.get("999999999999")


class TestOutboundCall:
    """One real call task, placed to a number that cannot connect.

    This is the only test here that starts a task, so it is the only one that
    reaches `session:task_ready`, `session:tool_status` and
    `session:task_finished` — the events that report what a task did. It costs
    credits and takes a few minutes.
    """

    async def test_a_call_that_cannot_connect_reports_through_tool_status(self, session):
        client, sid = session

        async for _ in client.chat(sid, CALL_PROMPT):
            pass

        # The task runs after the turn ends, so its events arrive outside it.
        calls: list[dict] = []
        finished: list[dict] = []

        async def watch():
            async for event in client.subscribe(sid):
                data = event.data if isinstance(event.data, dict) else {}
                if event.type == S2CEvent.SESSION_TOOL_STATUS:
                    calls.append(data)
                elif event.type == S2CEvent.SESSION_TASK_FINISHED:
                    finished.append(data)
                    return

        # A timeout is not a failure by itself — the assertions below say what
        # had to have arrived by then.
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(watch(), timeout=CALL_TIMEOUT_S)

        assert calls, f"no session:tool_status within {CALL_TIMEOUT_S:.0f}s"
        assert any(c.get("tool_name") == "phone_call" for c in calls)

        last = calls[-1]
        assert last.get("status") in ("failed", "completed"), last.get("status")
        assert (last.get("summary") or {}).get("text"), "the call reported no outcome"

        assert finished, "the task never reported a result"
        completion = finished[-1].get("completion") or {}
        assert completion.get("result_title")
        assert completion.get("outcome_narrative")
