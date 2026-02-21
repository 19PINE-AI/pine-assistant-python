"""Unit tests for ChatEngine — immediate dispatch, stale-event filtering, state precheck."""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from pine_assistant.chat import ChatEngine, ChatEvent
from pine_assistant.models.events import S2CEvent


def _make_sio():
    sio = MagicMock()
    sio.connected = True
    sio.emit = MagicMock()
    sio.add_event_handler = MagicMock(return_value=lambda: None)
    return sio


def _ts_iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _inject_events(sio, session_id, events, delay=0.05):
    """Replace add_event_handler so it injects events after a short delay."""
    def fake_add_handler(handler):
        async def _inject():
            await asyncio.sleep(delay)
            for evt_type, raw in events:
                handler(evt_type, raw)
        asyncio.get_running_loop().create_task(_inject())
        return lambda: None
    sio.add_event_handler = fake_add_handler


def _raw(event_type, session_id, data, ts=None):
    """Build a raw server envelope."""
    meta = {}
    if ts:
        meta["timestamp"] = ts
    return (event_type, {
        "payload": {"session_id": session_id, "data": data},
        "metadata": meta,
    })


# ── _is_stale_event ──────────────────────────────────────────────────────

class TestIsStaleEvent:
    def test_old_event_is_stale(self):
        cutoff = datetime(2026, 2, 21, 10, 0, 0, tzinfo=timezone.utc)
        event = ChatEvent(
            type=S2CEvent.SESSION_WORK_LOG, session_id="s1", data={},
            metadata={"timestamp": "2026-02-20T09:00:00Z"},
        )
        assert ChatEngine._is_stale_event(event, cutoff) is True

    def test_fresh_event_passes(self):
        cutoff = datetime(2026, 2, 21, 10, 0, 0, tzinfo=timezone.utc)
        event = ChatEvent(
            type=S2CEvent.SESSION_TEXT, session_id="s1", data={},
            metadata={"timestamp": "2026-02-21T10:00:05Z"},
        )
        assert ChatEngine._is_stale_event(event, cutoff) is False

    def test_no_metadata_passes(self):
        cutoff = datetime(2026, 2, 21, 10, 0, 0, tzinfo=timezone.utc)
        event = ChatEvent(type=S2CEvent.SESSION_WORK_LOG_PART, session_id="s1", data={})
        assert ChatEngine._is_stale_event(event, cutoff) is False

    def test_missing_timestamp_passes(self):
        cutoff = datetime(2026, 2, 21, 10, 0, 0, tzinfo=timezone.utc)
        event = ChatEvent(
            type=S2CEvent.SESSION_WORK_LOG, session_id="s1", data={},
            metadata={"source": {"role": "agent"}},
        )
        assert ChatEngine._is_stale_event(event, cutoff) is False

    def test_malformed_timestamp_passes(self):
        cutoff = datetime(2026, 2, 21, 10, 0, 0, tzinfo=timezone.utc)
        event = ChatEvent(
            type=S2CEvent.SESSION_WORK_LOG, session_id="s1", data={},
            metadata={"timestamp": "not-a-date"},
        )
        assert ChatEngine._is_stale_event(event, cutoff) is False


# ── immediate dispatch (no buffering) ─────────────────────────────────────

class TestImmediateDispatch:
    @pytest.mark.asyncio
    async def test_text_part_dispatched_immediately(self):
        """text_part events are yielded as-is, not buffered."""
        sio = _make_sio()
        now = _ts_iso(datetime.now(timezone.utc))

        _inject_events(sio, "s1", [
            _raw(S2CEvent.SESSION_TEXT_PART, "s1", {"content": "Hello "}, ts=now),
            _raw(S2CEvent.SESSION_TEXT_PART, "s1", {"content": "world"}, ts=now),
            _raw(S2CEvent.SESSION_STATE, "s1", {"content": "task_finished"}, ts=now),
        ])

        engine = ChatEngine(sio, idle_timeout_s=5.0)
        events = []
        async for event in engine._listen("s1", _skip_state_precheck=True):
            events.append(event)

        types = [e.type for e in events]
        assert types.count(S2CEvent.SESSION_TEXT_PART) == 2
        assert events[0].data["content"] == "Hello "
        assert events[1].data["content"] == "world"

    @pytest.mark.asyncio
    async def test_work_log_part_dispatched_immediately(self):
        """work_log_part events are yielded immediately, not debounced."""
        sio = _make_sio()
        now = _ts_iso(datetime.now(timezone.utc))

        _inject_events(sio, "s1", [
            _raw(S2CEvent.SESSION_WORK_LOG_PART, "s1",
                 {"step_id": "1", "text_delta": "thinking..."}, ts=now),
            _raw(S2CEvent.SESSION_STATE, "s1", {"content": "task_finished"}, ts=now),
        ])

        engine = ChatEngine(sio, idle_timeout_s=5.0)
        events = []
        async for event in engine._listen("s1", _skip_state_precheck=True):
            events.append(event)

        types = [e.type for e in events]
        assert S2CEvent.SESSION_WORK_LOG_PART in types


# ── chat() stale filtering ───────────────────────────────────────────────

class TestChatStaleFiltering:
    @pytest.mark.asyncio
    async def test_chat_filters_old_events(self):
        sio = _make_sio()
        old_ts = _ts_iso(datetime.now(timezone.utc) - timedelta(hours=24))
        fresh_ts = _ts_iso(datetime.now(timezone.utc) + timedelta(seconds=1))

        _inject_events(sio, "s1", [
            _raw(S2CEvent.SESSION_WORK_LOG, "s1",
                 {"steps": [{"step_title": "old"}]}, ts=old_ts),
            _raw(S2CEvent.SESSION_TEXT_PART, "s1", {"content": "Hi!"}, ts=fresh_ts),
            _raw(S2CEvent.SESSION_STATE, "s1", {"content": "task_finished"}, ts=fresh_ts),
        ])

        engine = ChatEngine(sio, idle_timeout_s=5.0)
        events = []
        async for event in engine.chat("s1", "test"):
            events.append(event)

        types = [e.type for e in events]
        assert S2CEvent.SESSION_WORK_LOG not in types
        assert S2CEvent.SESSION_TEXT_PART in types


# ── _listen with _skip_state_precheck ─────────────────────────────────────

class TestListenSkipStatePrecheck:
    @pytest.mark.asyncio
    async def test_listen_without_skip_returns_on_terminal(self):
        sio = _make_sio()

        async def _check(_sid):
            return {"state": "task_finished"}

        engine = ChatEngine(sio, check_session_state=_check)
        events = [e async for e in engine._listen("s1")]

        assert len(events) == 1
        assert events[0].data["content"] == "task_finished"

    @pytest.mark.asyncio
    async def test_listen_with_skip_enters_event_loop(self):
        sio = _make_sio()
        now = _ts_iso(datetime.now(timezone.utc))

        async def _check(_sid):
            return {"state": "task_finished"}

        _inject_events(sio, "s1", [
            _raw(S2CEvent.SESSION_TEXT_PART, "s1", {"content": "response"}, ts=now),
            _raw(S2CEvent.SESSION_STATE, "s1", {"content": "task_finished"}, ts=now),
        ])

        engine = ChatEngine(sio, check_session_state=_check, idle_timeout_s=5.0)
        events = [e async for e in engine._listen("s1", _skip_state_precheck=True)]

        types = [e.type for e in events]
        assert S2CEvent.SESSION_TEXT_PART in types
