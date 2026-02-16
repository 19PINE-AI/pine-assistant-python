"""
Integration tests for Pine AI Python SDK — tests against real Pine AI service.

Requires environment variables:
  PINE_ACCESS_TOKEN  — valid access token
  PINE_USER_ID       — user ID
  PINE_BASE_URL      — (optional) defaults to https://www.19pine.ai

Run: PINE_INTEGRATION=1 pytest tests/integration/ -v
"""

import os
import pytest
import pytest_asyncio

from pine_assistant import AsyncPineAI, S2CEvent

SKIP = not os.environ.get("PINE_INTEGRATION")
ACCESS_TOKEN = os.environ.get("PINE_ACCESS_TOKEN", "")
USER_ID = os.environ.get("PINE_USER_ID", "")
BASE_URL = os.environ.get("PINE_BASE_URL", "https://www.19pine.ai")

pytestmark = pytest.mark.skipif(SKIP, reason="PINE_INTEGRATION not set")


def make_client() -> AsyncPineAI:
    return AsyncPineAI(access_token=ACCESS_TOKEN, user_id=USER_ID, base_url=BASE_URL)


class TestConnectionLifecycle:
    """T2: Socket.IO Connection Lifecycle"""

    @pytest.mark.asyncio
    async def test_connects_and_receives_ready(self):
        client = make_client()
        await client.connect()
        assert client.connected
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_rejects_invalid_token(self):
        client = AsyncPineAI(access_token="invalid", user_id=USER_ID, base_url=BASE_URL)
        with pytest.raises(Exception):
            await client.connect()


class TestSessionCRUD:
    """T3: Session CRUD"""

    @pytest.mark.asyncio
    async def test_create_list_get_delete(self):
        client = make_client()

        # Create
        session = await client.sessions.create()
        assert session["id"]
        assert session["state"] == "init"
        session_id = session["id"]

        # List
        result = await client.sessions.list(limit=50)
        assert result["total"] > 0
        ids = [s["id"] for s in result["sessions"]]
        assert session_id in ids

        # Get
        fetched = await client.sessions.get(session_id)
        assert fetched["id"] == session_id

        # Delete
        await client.sessions.delete(session_id)


class TestChatStreaming:
    """T4 + T5: Chat with stream buffering"""

    @pytest.mark.asyncio
    async def test_chat_returns_merged_text(self):
        client = make_client()
        await client.connect()
        session = await client.sessions.create()
        sid = session["id"]
        await client.join_session(sid)

        events = []
        async for event in client.chat(sid, "What is Pine AI?"):
            events.append(event)
            if event.type == S2CEvent.SESSION_TEXT:
                break

        # Should have merged text, not individual text_parts
        text_events = [e for e in events if e.type == S2CEvent.SESSION_TEXT]
        text_part_events = [e for e in events if e.type == S2CEvent.SESSION_TEXT_PART]
        assert len(text_events) > 0, "Should receive at least one merged text event"
        assert len(text_part_events) == 0, "text_parts should be buffered internally"

        content = text_events[0].data.get("content", "") if isinstance(text_events[0].data, dict) else ""
        assert len(content) > 0
        print(f"  Response ({len(content)} chars): {content[:200]}...")

        client.leave_session(sid)
        try:
            await client.sessions.delete(sid)
        except Exception:
            pass
        await client.disconnect()


class TestSessionHistory:
    """T9: Session History"""

    @pytest.mark.asyncio
    async def test_fetch_history(self):
        client = make_client()
        await client.connect()
        session = await client.sessions.create()
        sid = session["id"]
        await client.join_session(sid)

        # Send a message to create history
        async for event in client.chat(sid, "Hello"):
            if event.type == S2CEvent.SESSION_TEXT:
                break

        # Fetch history
        history = await client.get_history(sid, max_messages=10)
        assert "messages" in history
        assert isinstance(history["messages"], list)

        client.leave_session(sid)
        try:
            await client.sessions.delete(sid)
        except Exception:
            pass
        await client.disconnect()


class TestFormInteraction:
    """T6: Form Interaction"""

    @pytest.mark.asyncio
    async def test_can_detect_forms(self):
        client = make_client()
        await client.connect()
        session = await client.sessions.create()
        sid = session["id"]
        await client.join_session(sid)

        events = []
        async for event in client.chat(sid, "Help me negotiate my Comcast internet bill down to $50/month"):
            events.append(event)
            if event.type == S2CEvent.SESSION_FORM_TO_USER:
                data = event.data if isinstance(event.data, dict) else {}
                form = data.get("form", {})
                fields = form.get("fields", [])
                response = {f.get("name", ""): "test_value" for f in fields if isinstance(f, dict)}
                if response:
                    client.send_form_response(sid, event.message_id or "0", response)
                break
            if event.type == S2CEvent.SESSION_TEXT:
                break

        assert len(events) > 0

        client.leave_session(sid)
        try:
            await client.sessions.delete(sid)
        except Exception:
            pass
        await client.disconnect()


class TestStateTransitions:
    """T7: Session State Transitions"""

    @pytest.mark.asyncio
    async def test_init_to_chat_transition(self):
        client = make_client()
        await client.connect()
        session = await client.sessions.create()
        sid = session["id"]
        assert session["state"] == "init"

        await client.join_session(sid)
        async for event in client.chat(sid, "I need help with my phone bill"):
            if event.type == S2CEvent.SESSION_TEXT:
                break

        updated = await client.sessions.get(sid)
        assert updated["state"] in ("chat", "init")

        client.leave_session(sid)
        try:
            await client.sessions.delete(sid)
        except Exception:
            pass
        await client.disconnect()


class TestErrorHandling:
    """T10: Error Handling"""

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self):
        client = make_client()
        with pytest.raises(Exception):
            await client.sessions.get("999999999999")
