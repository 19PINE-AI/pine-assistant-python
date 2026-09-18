"""Flow tests — the requirements the protocol scope states as MUST.

Events are pushed through the real transport, so these cover envelope parsing
and SocketIOManager as well as the chat engine. Each test names the requirement
it pins.
"""

import asyncio
from unittest.mock import patch

import pytest

from pine_assistant import SUPPORTED_EVENTS, AsyncPineAI
from pine_assistant.chat import CONTENT_EVENTS, FULL_REBUILD_REVISION
from pine_assistant.models.events import C2SEvent
from pine_assistant.models.form import FormData, FormField, FormShowWhen, encode_form_answers
from tests.protocol.fake import SESSION_ID, FakeAsyncClient, envelope, load_fixture

OTHER_SESSION = "1900000000000000999"


@pytest.fixture
async def client():
    fake = FakeAsyncClient()
    with patch("pine_assistant.transport.socketio.socketio.AsyncClient", return_value=fake):
        pine = AsyncPineAI(access_token="t", user_id="u", device_id="d")
        await pine.connect()
        yield pine, fake
        await pine.disconnect()


def _join_ack(_request):
    return load_fixture("join")["payload"]["data"]


# -- Recovery is by unconditional rebuild ---------------------------------


async def test_join_sends_since_revision_zero(client):
    """MUST send session:join with since_revision "0"."""
    pine, fake = client
    fake.reply_to("session:join", _join_ack)

    await pine.join_session(SESSION_ID)

    joins = fake.emits_of("session:join")
    assert len(joins) == 1
    assert joins[0]["payload"]["data"] == {"since_revision": FULL_REBUILD_REVISION}


async def test_rebuild_pages_until_the_cursor_is_exhausted(client):
    """MUST NOT infer exhaustion from an empty or short page — only the cursor
    says a range is done."""
    pine, fake = client
    pages = [
        {"messages": [{"id": "1"}, {"id": "2"}], "next_message_id": "2"},
        {"messages": [], "next_message_id": "3"},          # empty, but not the end
        {"messages": [{"id": "4"}], "next_message_id": ""},  # short, and the end
    ]
    calls = {"n": 0}

    def responder(_request):
        page = pages[calls["n"]]
        calls["n"] += 1
        return page

    fake.reply_to("session:history", responder)

    messages = await pine.rebuild(SESSION_ID, page_size=2)

    assert calls["n"] == 3, "stopped early on an empty or short page"
    assert [m["id"] for m in messages] == ["1", "2", "4"]


async def test_history_requires_the_exact_request_and_session_correlation(client):
    """A same-event reply for another session must never satisfy this request."""
    pine, fake = client
    task = asyncio.create_task(pine._sio.emit_and_wait(  # type: ignore[union-attr]
        C2SEvent.SESSION_HISTORY, {"max_messages": 1}, session_id=SESSION_ID, timeout=0.02,
    ))
    await asyncio.sleep(0)
    request = fake.emits_of("session:history")[0]
    await fake.deliver(envelope(
        "session:history", {"messages": [{"wrong": True}]}, session_id=OTHER_SESSION,
        request_id=request["metadata"]["request_id"], event_id="wrong-session",
    ))
    with pytest.raises(TimeoutError):
        await task
    assert not pine._sio._event_handlers  # type: ignore[union-attr]


async def test_concurrent_history_requests_do_not_cross_complete(client):
    """Replies may arrive in reverse order on one Socket.IO connection."""
    pine, fake = client
    first = asyncio.create_task(pine.get_history(SESSION_ID, max_messages=1))
    second = asyncio.create_task(pine.get_history(SESSION_ID, max_messages=2))
    await asyncio.sleep(0)
    requests = fake.emits_of("session:history")
    assert len(requests) == 2
    for index in (1, 0):
        request = requests[index]
        await fake.deliver(envelope(
            "session:history", {"messages": [{"request": index}]},
            request_id=request["metadata"]["request_id"], event_id=f"history-{index}",
        ))
    assert (await first)["messages"] == [{"request": 0}]
    assert (await second)["messages"] == [{"request": 1}]
    assert not pine._sio._event_handlers  # type: ignore[union-attr]


async def test_request_wait_cleans_up_after_emit_failure(client):
    pine, fake = client

    async def fail_emit(*_args, **_kwargs):
        raise RuntimeError("wire unavailable")

    fake.emit = fail_emit
    with pytest.raises(RuntimeError, match="wire unavailable"):
        await pine.get_history(SESSION_ID)
    assert not pine._sio._event_handlers  # type: ignore[union-attr]


async def test_request_wait_cleans_up_when_cancelled(client):
    pine, _fake = client
    task = asyncio.create_task(pine.get_history(SESSION_ID))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not pine._sio._event_handlers  # type: ignore[union-attr]


async def test_request_wait_cleans_up_when_socket_disconnects(client):
    pine, fake = client
    task = asyncio.create_task(pine.get_history(SESSION_ID))
    await asyncio.sleep(0)
    await fake._handlers["disconnect"]()
    with pytest.raises(RuntimeError, match="disconnected"):
        await task
    assert not pine._sio._event_handlers  # type: ignore[union-attr]


def _form_request(message_id="form-1", request_id="form-request"):
    return envelope(
        "session:form_to_user",
        {
            "form": {
                "fields": [
                    {"name": "plan", "is_required": True, "type": "radio"},
                    {"name": "extras", "show_when": {"field": "plan", "equals": ["yes"]}},
                ],
            },
        },
        message_id=message_id, request_id=request_id, role="agent",
    )


def test_form_visibility_uses_javascript_strict_primitive_equality():
    form = FormData(fields=[
        FormField(name="parent"),
        FormField(name="child", show_when=FormShowWhen(field="parent", equals=[1])),
    ])
    assert encode_form_answers(form, {"parent": True}) == {"parent": True}
    with pytest.raises(ValueError, match="not visible"):
        encode_form_answers(form, {"parent": True, "child": "value"})


async def test_form_reply_requires_persisted_echo_and_delivery_receipt(client):
    pine, fake = client
    original = _form_request()
    fake.reply_to("session:history", {"messages": [original], "next_message_id": ""})

    def responder(request):
        request_id = request["metadata"]["request_id"]
        assert request_id == "form-request"
        assert request["payload"]["message_id"] == "form-1"
        assert request["payload"]["data"] == {"content": {"plan": "yes", "extras": '["a","b"]'}}
        return [
            envelope("session:form_to_user", {"content": {"plan": "yes"}}, message_id="reply-1",
                     request_id=request_id, event_id=request["metadata"]["event_id"], role="user"),
            envelope("session:message_status", {
                "message_id": "reply-1", "request_id": request_id, "status": "delivered",
            }, request_id=request_id, event_id="delivered", role="system"),
        ]

    fake.responders["session:form_to_user"] = responder
    result = await pine.submit_form_response(SESSION_ID, "form-1", {"plan": "yes", "extras": ["a", "b"]})
    assert result.status == "delivered"
    assert result.message_id == "reply-1"
    assert result.request_id == "form-request"
    assert not pine._sio._event_handlers  # type: ignore[union-attr]


async def test_form_ack_without_persisted_reply_is_unknown_and_never_retried(client):
    pine, fake = client
    fake.reply_to("session:history", {"messages": [_form_request()], "next_message_id": ""})
    result = await pine.submit_form_response(SESSION_ID, "form-1", {"plan": "yes"}, timeout=0.02)
    assert result.status == "unknown"
    assert len(fake.emits_of("session:form_to_user")) == 1
    assert not pine._sio._event_handlers  # type: ignore[union-attr]


async def test_form_received_receipt_retains_its_actual_reason(client):
    pine, fake = client
    fake.reply_to("session:history", {"messages": [_form_request()], "next_message_id": ""})

    def responder(request):
        request_id = request["metadata"]["request_id"]
        return [
            envelope("session:form_to_user", {"content": {}}, message_id="reply-2",
                     request_id=request_id, event_id=request["metadata"]["event_id"], role="user"),
            envelope("session:message_status", {
                "message_id": "reply-2", "request_id": request_id, "status": "received",
                "reason": "Send message to agent failed.",
            }, request_id=request_id, event_id="received-2", role="system"),
        ]

    fake.responders["session:form_to_user"] = responder
    result = await pine.submit_form_response(SESSION_ID, "form-1", {"plan": "yes"}, timeout=0.02)
    assert result.status == "received"
    assert result.reason == "Send message to agent failed."


async def test_concurrent_form_replies_with_one_request_id_keep_their_own_echoes(client):
    pine, fake = client
    fake.reply_to("session:history", {"messages": [_form_request()], "next_message_id": ""})
    first = asyncio.create_task(pine.submit_form_response(SESSION_ID, "form-1", {"plan": "yes"}))
    second = asyncio.create_task(pine.submit_form_response(SESSION_ID, "form-1", {"plan": "yes"}))
    requests = []
    for _ in range(10):
        await asyncio.sleep(0)
        requests = fake.emits_of("session:form_to_user")
        if len(requests) == 2:
            break
    assert len(requests) == 2
    for index in (1, 0):
        request = requests[index]
        event_id = request["metadata"]["event_id"]
        request_id = request["metadata"]["request_id"]
        reply_id = f"concurrent-reply-{index}"
        await fake.deliver(envelope(
            "session:form_to_user", {"content": {}}, message_id=reply_id,
            request_id=request_id, event_id=event_id, role="user",
        ))
        await fake.deliver(envelope(
            "session:message_status", {"message_id": reply_id, "request_id": request_id, "status": "delivered"},
            request_id=request_id, event_id=f"receipt-{index}", role="system",
        ))
    assert (await first).message_id == "concurrent-reply-0"
    assert (await second).message_id == "concurrent-reply-1"


async def test_form_rejects_unrequested_or_hidden_answers_without_emitting(client):
    pine, fake = client
    fake.reply_to("session:history", {"messages": [_form_request()], "next_message_id": ""})
    with pytest.raises(ValueError, match="persisted form"):
        await pine.submit_form_response(SESSION_ID, "form-1", {"plan": "no", "extras": "secret"})
    assert not fake.emits_of("session:form_to_user")


async def test_form_rejects_a_persisted_request_without_fields_without_emitting(client):
    pine, fake = client
    empty = envelope(
        "session:form_to_user", {"form": {"fields": []}}, message_id="empty-form",
        request_id="empty-request", role="agent",
    )
    fake.reply_to("session:history", {"messages": [empty], "next_message_id": ""})
    with pytest.raises(ValueError, match="persisted form"):
        await pine.submit_form_response(SESSION_ID, "empty-form", {})
    assert not fake.emits_of("session:form_to_user")


async def test_reconnect_rejoins_with_a_full_rebuild(client):
    """MUST rebuild on every reconnect, not resume from a cursor."""
    pine, fake = client
    fake.reply_to("session:join", _join_ack)
    await pine.join_session(SESSION_ID)

    rebuilt = []
    pine.on_reconnect(lambda: rebuilt.append(True))
    await fake.fire_ready()  # the server re-announces readiness after a drop
    await asyncio.sleep(0)  # the re-join is scheduled, not awaited

    rejoins = fake.emits_of("session:join")
    assert len(rejoins) == 2
    assert rejoins[1]["payload"]["data"] == {"since_revision": FULL_REBUILD_REVISION}
    assert rebuilt == [True]


# -- Unrecognized events are ignored --------------------------------------


async def test_unknown_events_pass_through_unchanged(client):
    """MUST ignore unrecognized events without erroring, dropping them, or
    disturbing ordering. The SDK delivers them verbatim instead."""
    pine, fake = client
    unknown_payload = {"anything": [1, 2, {"deep": True}]}

    def responder(_request):
        return [
            load_fixture("text_part"),
            envelope("session:an_event_from_the_future", unknown_payload,
                     event_id="unknown-1"),
            load_fixture("text"),
            envelope("session:state", {"content": "task_finished"},
                     event_id="final-1", role="system"),
        ]

    fake.responders["session:message"] = responder
    events = [e async for e in pine.chat(SESSION_ID, "hello")]

    types = [e.type for e in events]
    assert types[:3] == ["session:text_part", "session:an_event_from_the_future", "session:text"]
    unknown = events[1]
    assert unknown.data == unknown_payload, "payload was altered on the way out"
    assert unknown.event_id == "unknown-1"


async def test_events_for_another_session_are_not_delivered(client):
    """A client tracks one session."""
    pine, fake = client

    def responder(_request):
        return [
            envelope("session:text", {"content": "elsewhere"}, session_id=OTHER_SESSION,
                     event_id="other-1"),
            envelope("session:text", {"content": "here"}, event_id="here-1"),
            envelope("session:state", {"content": "task_finished"},
                     event_id="final-2", role="system"),
        ]

    fake.responders["session:message"] = responder
    events = [e async for e in pine.chat(SESSION_ID, "hello")]

    contents = [e.data.get("content") for e in events if e.type == "session:text"]
    assert contents == ["here"]


# -- Deduplication --------------------------------------------------------


async def test_duplicate_events_are_suppressed(client):
    """A rebuild re-delivers what was already seen; the same event must not
    surface twice."""
    pine, fake = client
    duplicate = envelope("session:text", {"content": "once"}, event_id="dup-1")

    def responder(_request):
        return [duplicate, duplicate,
                envelope("session:state", {"content": "task_finished"},
                     event_id="final-3", role="system")]

    fake.responders["session:message"] = responder
    events = [e async for e in pine.chat(SESSION_ID, "hello")]

    assert sum(1 for e in events if e.type == "session:text") == 1


async def test_dedup_keys_on_id_and_type_together(client):
    """MUST NOT key on the event identifier alone — identifiers collide across
    message types, and keying on one alone silently drops real events."""
    pine, fake = client
    shared_id = "shared-event-id"

    def responder(_request):
        return [
            envelope("session:text", {"content": "a text"}, event_id=shared_id),
            envelope("session:update_title", {"content": "a title"}, event_id=shared_id,
                     role="system"),
            envelope("session:state", {"content": "task_finished"},
                     event_id="final-4", role="system"),
        ]

    fake.responders["session:message"] = responder
    events = [e async for e in pine.chat(SESSION_ID, "hello")]

    types = [e.type for e in events]
    assert "session:text" in types
    assert "session:update_title" in types


# -- Turn termination -----------------------------------------------------


async def test_turn_ends_on_a_settled_session_state(client):
    pine, fake = client

    def responder(_request):
        return [
            load_fixture("text"),
            envelope("session:state", {"content": "task_finished"},
                     event_id="final-5", role="system"),
        ]

    fake.responders["session:message"] = responder
    events = [e async for e in pine.chat(SESSION_ID, "hello")]

    assert events[-1].type == "session:state"


async def test_turn_ends_on_a_terminal_session_state(client):
    pine, fake = client

    def responder(_request):
        # Constructed: a recorded session:state holds whichever state the
        # recording ended on, which need not be a terminal one.
        return [load_fixture("text"),
                envelope("session:state", {"content": "task_finished"},
                         event_id="terminal-1", role="system")]

    fake.responders["session:message"] = responder
    events = [e async for e in pine.chat(SESSION_ID, "hello")]

    assert events[-1].type == "session:state"
    assert events[-1].data["content"] == "task_finished"


async def test_out_of_scope_events_do_not_end_a_turn(client):
    """Control flow must not hinge on an event outside the supported surface."""
    pine, fake = client

    def responder(_request):
        return [
            envelope("session:payment", {"status": "pending"}, event_id="oos-1"),
            envelope("session:reward", {"charge_type": "percentage"}, event_id="oos-2"),
            load_fixture("text"),
            envelope("session:state", {"content": "task_finished"},
                     event_id="final-6", role="system"),
        ]

    fake.responders["session:message"] = responder
    events = [e async for e in pine.chat(SESSION_ID, "hello")]

    types = [e.type for e in events]
    assert types.index("session:text") < types.index("session:state")
    assert "session:payment" in types


# -- Escape hatch ---------------------------------------------------------


async def test_emit_event_sends_anything_enveloped(client):
    """The unsupported surface stays reachable, just unmodelled."""
    pine, fake = client

    pine.emit_event("session:location_selection", {"list": [{"id": "p1"}]}, SESSION_ID, "m1")
    await asyncio.sleep(0)  # emit is scheduled, not awaited

    sent = fake.emits_of("session:location_selection")
    assert len(sent) == 1
    assert sent[0]["payload"]["data"] == {"list": [{"id": "p1"}]}
    assert sent[0]["payload"]["message_id"] == "m1"


# -- When a turn ends -----------------------------------------------------


@pytest.fixture
async def impatient(client):
    """A client whose waits are measured in fractions of a second."""
    pine, fake = client
    pine._chat._idle_timeout_s = 1.0
    pine._chat._response_idle_timeout_s = 0.1

    async def _in_progress(_sid):
        return {"state": "chat"}
    pine._chat._check_session_state = _in_progress
    return pine, fake


async def _collect(pine, budget=4.0, **kwargs):
    async def drain():
        return [e async for e in pine.chat(SESSION_ID, "hello", **kwargs)]
    try:
        return await asyncio.wait_for(drain(), timeout=budget)
    except asyncio.TimeoutError:
        return None


async def test_a_turn_ends_when_the_agent_stops_speaking(impatient):
    pine, fake = impatient
    fake.responders["session:message"] = lambda _r: [
        envelope("session:text_part", {"content": "half a "}, event_id="t1"),
        envelope("session:text_part", {"content": "sentence"}, event_id="t2"),
    ]

    events = await _collect(pine)

    assert events is not None, "the turn never ended"
    assert [e.type for e in events] == ["session:text_part", "session:text_part"]


async def test_a_running_tool_is_not_mistaken_for_a_finished_turn(impatient):
    """An agent that says what it is about to do, then does it, goes quiet
    while the work runs. Reading that silence as the end of the turn cuts the
    caller off from the result."""
    pine, fake = impatient
    fake.responders["session:message"] = lambda _r: [
        envelope("session:text", {"content": "Placing the call now."}, event_id="s1"),
        envelope("session:tool_status",
                 {"tool_name": "phone_call", "status": "in_progress"}, event_id="w1"),
    ]

    events = await _collect(pine, budget=1.5)

    assert events is None, "the turn ended while the tool was still running"


async def test_a_turn_not_ending_on_content_waits(impatient):
    """The boundary of the rule, stated rather than left to be discovered.

    A turn is over when the agent has spoken and gone quiet. When the last
    thing to arrive is not the agent speaking, the silence that follows is
    read as work still running, and the turn waits. Every plain turn observed
    against a live server ended on content, but nothing guarantees it —
    `turn_timeout` is what bounds the case where it does not.
    """
    pine, fake = impatient
    trailing_reasoning = [
        envelope("session:text", {"content": "Done."}, event_id="s2"),
        envelope("session:llm_thinking", {"type": "turn_end", "final": True}, event_id="k1"),
    ]

    fake.responders["session:message"] = lambda _r: trailing_reasoning
    assert await _collect(pine, budget=2.0) is None

    fake.responders["session:message"] = lambda _r: trailing_reasoning
    events = await _collect(pine, budget=2.0, turn_timeout=0.4)
    assert events is not None
    assert [e.type for e in events] == ["session:text", "session:llm_thinking"]


async def test_turn_timeout_bounds_a_turn_that_would_not_end(impatient):
    pine, fake = impatient
    fake.responders["session:message"] = lambda _r: [
        envelope("session:llm_thinking", {"type": "placeholder"}, event_id="p1"),
    ]

    events = await _collect(pine, budget=4.0, turn_timeout=0.5)

    assert events is not None, "turn_timeout did not bound the turn"
    assert [e.type for e in events] == ["session:llm_thinking"], "events seen before the deadline are kept"


async def test_content_events_are_all_supported():
    """No timing may hinge on an event outside the supported surface."""
    assert {e.value for e in CONTENT_EVENTS} <= SUPPORTED_EVENTS


async def test_a_session_stopped_on_its_balance_ends_the_turn(impatient):
    """Credits run out mid-task. Nothing further arrives until the balance is
    restored, which is not something the session can be told."""
    pine, fake = impatient
    fake.responders["session:message"] = lambda _r: [
        envelope("session:text", {"content": "Working on it."}, event_id="b1"),
        envelope("session:state", {"content": "credits_exhausted"},
                 event_id="b2", role="system"),
    ]

    events = await _collect(pine, budget=4.0)

    assert events is not None, "the turn never ended"
    assert events[-1].data["content"] == "credits_exhausted"
