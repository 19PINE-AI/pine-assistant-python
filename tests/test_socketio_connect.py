"""Unit tests for SocketIOManager.connect() — connect_error surfacing."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pine_assistant.errors import ConnectionError as PineConnectionError
from pine_assistant.transport.socketio import (
    SocketIOManager,
    _format_connect_error,
)


def test_format_connect_error_string():
    assert _format_connect_error("bad token") == "bad token"


def test_format_connect_error_dict_message():
    assert _format_connect_error({"message": "invalid token"}) == "invalid token"


def test_format_connect_error_dict_error_key():
    assert _format_connect_error({"error": "unauthorized"}) == "unauthorized"


def test_format_connect_error_none():
    assert _format_connect_error(None) == "no payload"


def test_format_connect_error_unknown():
    assert _format_connect_error([1, 2]) == "[1, 2]"


class _FakeAsyncClient:
    """Minimal stand-in for socketio.AsyncClient that lets a test drive the
    connect_error path."""

    def __init__(self, connect_error_payload: Any = None) -> None:
        self._handlers: dict[str, Any] = {}
        self._connect_error_payload = connect_error_payload
        self.connected = False

    def event(self, fn):
        self._handlers[fn.__name__] = fn
        return fn

    def on(self, name):
        def deco(fn):
            self._handlers[name] = fn
            return fn
        return deco

    async def connect(self, *_args, **_kwargs):
        # Simulate the server replying with connect_error, then python-socketio
        # raising the generic ConnectionError after wait_timeout.
        handler = self._handlers.get("connect_error")
        if handler is not None:
            await handler(self._connect_error_payload)
        raise ConnectionError("One or more namespaces failed to connect")

    async def disconnect(self):
        self.connected = False


@pytest.mark.asyncio
async def test_connect_error_payload_surfaces_in_pine_connection_error():
    fake = _FakeAsyncClient(connect_error_payload={"message": "invalid token"})
    mgr = SocketIOManager(
        base_url="https://example.test",
        token="bad-token",
        user_id="u1",
        device_id="d1",
        ready_timeout=0.1,
    )
    with patch("pine_assistant.transport.socketio.socketio.AsyncClient", return_value=fake):
        with pytest.raises(PineConnectionError) as excinfo:
            await mgr.connect()
    msg = str(excinfo.value)
    assert "rejected by server" in msg
    assert "invalid token" in msg


@pytest.mark.asyncio
async def test_connect_error_without_payload_falls_back_to_generic():
    fake = _FakeAsyncClient(connect_error_payload=None)
    # Override: don't invoke the connect_error handler at all — simulate a
    # transport-level failure where the server never sent connect_error.
    async def connect_no_event(*_args, **_kwargs):
        raise ConnectionError("transport closed")
    fake.connect = connect_no_event  # type: ignore[method-assign]

    mgr = SocketIOManager(
        base_url="https://example.test",
        token="t",
        user_id="u",
        device_id="d",
        ready_timeout=0.1,
    )
    with patch("pine_assistant.transport.socketio.socketio.AsyncClient", return_value=fake):
        with pytest.raises(PineConnectionError) as excinfo:
            await mgr.connect()
    assert "transport closed" in str(excinfo.value)


class _SilentStallClient:
    """Stand-in that accepts the connect but never fires the 'ready' event —
    matches Pine backend's behavior when access_token/user_id is rejected."""

    def __init__(self) -> None:
        self._handlers: dict[str, Any] = {}
        self.connected = False

    def event(self, fn):
        self._handlers[fn.__name__] = fn
        return fn

    def on(self, name):
        def deco(fn):
            self._handlers[name] = fn
            return fn
        return deco

    async def connect(self, *_args, **_kwargs):
        self.connected = True

    async def disconnect(self):
        self.connected = False


@pytest.mark.asyncio
async def test_silent_stall_raises_diagnostic_pine_error():
    """When the WS connects but no 'ready' event arrives, surface a hint that
    points the user at re-auth, not a generic asyncio TimeoutError."""
    fake = _SilentStallClient()
    mgr = SocketIOManager(
        base_url="https://example.test",
        token="t",
        user_id="u",
        device_id="d",
        ready_timeout=0.1,
    )
    with patch("pine_assistant.transport.socketio.socketio.AsyncClient", return_value=fake):
        with pytest.raises(PineConnectionError) as excinfo:
            await mgr.connect()
    msg = str(excinfo.value)
    assert "no 'ready' event" in msg
    assert "re-run the auth flow" in msg
