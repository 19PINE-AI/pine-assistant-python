"""A scripted stand-in for socketio.AsyncClient.

Events travel the real path — envelope parsing, SocketIOManager, ChatEngine —
so what these tests exercise is the SDK, not a mock of it. Only the wire is
fake.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Callable
from typing import Any

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
SESSION_ID = "1900000000000000001"


def load_fixture(name: str) -> dict[str, Any]:
    """Load one recorded envelope by event name (`session:text` or `text`)."""
    return json.loads((FIXTURES / f"{name.replace('session:', '')}.json").read_text())


def all_fixtures() -> dict[str, dict[str, Any]]:
    """Every fixture, keyed by the event type it carries."""
    out = {}
    for path in sorted(FIXTURES.glob("*.json")):
        if path.name == "provenance.json":
            continue
        envelope = json.loads(path.read_text())
        out[envelope["type"]] = envelope
    return out


def envelope(
    event_type: str,
    data: Any = None,
    *,
    session_id: str = SESSION_ID,
    message_id: str | None = None,
    event_id: str = "00000000-0000-4000-8000-00000000dead",
    request_id: str | None = None,
    role: str = "agent",
) -> dict[str, Any]:
    """Build an arbitrary server envelope — including for events the SDK has
    never heard of."""
    return {
        "metadata": {
            "event_id": event_id,
            "request_id": request_id,
            "timestamp": "2026-08-08T00:00:00Z",
            "source": {"role": role},
            "is_volatile": False,
        },
        "type": event_type,
        "payload": {
            "session_id": session_id,
            "message_id": message_id,
            "type": event_type,
            "data": data,
        },
    }


class FakeAsyncClient:
    """Implements the surface of socketio.AsyncClient that SocketIOManager uses."""

    def __init__(self) -> None:
        self._handlers: dict[str, Any] = {}
        self.connected = False
        self.emitted: list[tuple[str, dict[str, Any]]] = []
        self.responders: dict[str, Callable[[dict[str, Any]], list[dict[str, Any]]]] = {}

    # -- socketio.AsyncClient surface -------------------------------------

    def event(self, fn: Any) -> Any:
        self._handlers[fn.__name__] = fn
        return fn

    def on(self, name: str) -> Any:
        def deco(fn: Any) -> Any:
            self._handlers[name] = fn
            return fn
        return deco

    async def connect(self, *_args: Any, **_kwargs: Any) -> None:
        self.connected = True
        await self.fire_ready()

    async def emit(self, event: str, data: Any = None) -> None:
        self.emitted.append((event, data))
        responder = self.responders.get(event)
        if responder is not None:
            for reply in responder(data):
                await self.deliver(reply)

    async def disconnect(self) -> None:
        self.connected = False

    # -- test controls ----------------------------------------------------

    async def fire_ready(self) -> None:
        handler = self._handlers.get("ready")
        if handler is not None:
            await handler()

    async def deliver(self, env: dict[str, Any]) -> None:
        """Push one server event down the same path a real one takes."""
        handler = self._handlers.get("*")
        if handler is not None:
            await handler(env["type"], env)

    def emits_of(self, event_type: str) -> list[dict[str, Any]]:
        return [payload for evt, payload in self.emitted if evt == event_type]

    def reply_to(self, event_type: str, data: Any, *, role: str = "system") -> None:
        """Answer an emitted request with an envelope echoing its request_id."""
        def responder(request: dict[str, Any]) -> list[dict[str, Any]]:
            request_id = (request.get("metadata") or {}).get("request_id")
            payload = data(request) if callable(data) else data
            return [envelope(
                event_type, payload,
                request_id=request_id,
                event_id=f"reply-{request_id}",
                role=role,
            )]
        self.responders[event_type] = responder
