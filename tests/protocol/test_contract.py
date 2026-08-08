"""Contract tests — one envelope at a time, no flow.

Each supported event has a fixture. These assert the SDK reads it: the envelope
parses, the type survives, and the fields the protocol scope names are
reachable. A failure here means a payload shape moved.

A fixture records a shape, not a scenario. Whichever instance a recording
happened to catch is the one checked in — `session:state` may hold "chat" rather
than a terminal state, `session:input_state` an open composer rather than a
blocked one. Assertions here stay on what every instance carries; a specific
condition is constructed by the test that needs it.
"""

import json
import pathlib

import pytest

from pine_assistant.chat import event_from_envelope
from pine_assistant.models.envelope import MessageEnvelope
from pine_assistant.models.events import SUPPORTED_EVENTS, S2CEvent
from pine_assistant.models.form import FormToUserData
from pine_assistant.models.session import InputState, InputStateCode
from pine_assistant.models.task import (
    LLMThinkingData,
    MessageStatusData,
    RequiredActionData,
    RestrictionData,
    RichContentData,
    TaskFinishedData,
    TaskReadyData,
    ToolStatusData,
)
from pine_assistant.transport.envelope import parse_envelope
from tests.protocol.fake import FIXTURES, all_fixtures, load_fixture

FIXTURE_EVENTS = sorted(all_fixtures())


def test_every_supported_event_has_a_fixture():
    """`ready` carries no envelope; everything else in scope needs a sample."""
    expected = {e.value for e in S2CEvent} - {S2CEvent.READY.value}
    assert set(FIXTURE_EVENTS) == expected


def test_fixtures_carry_only_supported_events():
    assert set(FIXTURE_EVENTS) <= SUPPORTED_EVENTS


def test_every_fixture_declares_its_provenance():
    """A derived sample and a recorded one are not equally trustworthy, so the
    difference is written down rather than remembered."""
    provenance = json.loads((FIXTURES / "provenance.json").read_text())
    assert set(provenance) == set(FIXTURE_EVENTS)
    for event, entry in provenance.items():
        assert entry["source"] in ("recorded", "derived"), event


@pytest.mark.parametrize("event_type", FIXTURE_EVENTS)
def test_envelope_parses(event_type):
    envelope = all_fixtures()[event_type]
    parsed = parse_envelope(envelope)
    assert isinstance(parsed, MessageEnvelope)
    assert parsed.type == event_type
    assert parsed.payload.session_id


@pytest.mark.parametrize("event_type", FIXTURE_EVENTS)
def test_event_reaches_the_caller_intact(event_type):
    envelope = all_fixtures()[event_type]
    event = event_from_envelope(event_type, envelope, envelope["payload"]["session_id"])
    assert event.type == event_type
    assert event.event_id == envelope["metadata"]["event_id"]
    assert event.data == envelope["payload"]["data"]


# -- the fields the scope names ------------------------------------------


def test_task_finished_carries_the_textual_conclusion():
    """A finished task always states an outcome. `result_description` and
    `summary.brief` are not always written — a task that failed carries the
    narrative and nothing else."""
    data = TaskFinishedData.model_validate(load_fixture("task_finished")["payload"]["data"])
    assert data.status
    assert data.completion is not None
    assert data.completion.result_title
    assert data.completion.outcome_narrative


def test_task_finished_on_success_quantifies_the_outcome():
    """Constructed: a task that reaches a result is not ours to arrange."""
    data = TaskFinishedData.model_validate({
        "status": "completed",
        "completion": {
            "result_title": "Bill reduced by $35/month",
            "result_description": "A 12-month promotional rate was applied.",
            "outcome_narrative": "Pine called, held, and secured a promotional rate.",
            "summary": {"brief": "Saved $420 over 12 months.", "money_saved": 420.0,
                        "calls_made": 1, "credits_invested": 500},
        },
    })
    assert data.completion.result_description
    assert data.completion.summary.brief
    assert data.completion.summary.credits_invested


def test_tool_status_carries_the_outbound_call_record():
    """The number called and the textual outcome. The quantified fields are
    written when there is something to count — a call that never connected
    reports its outcome and no duration."""
    data = ToolStatusData.model_validate(load_fixture("tool_status")["payload"]["data"])
    assert data.operation_id
    assert data.tool_name
    assert data.target
    assert data.status
    assert data.summary is not None
    assert data.summary.text


def test_tool_status_on_a_completed_call_quantifies_it():
    """Constructed: recording a connected call means calling someone."""
    data = ToolStatusData.model_validate({
        "operation_id": "op-1", "tool_name": "phone_call", "target": "+15555550100",
        "status": "completed",
        "summary": {"text": "Reached billing.", "duration": 1320,
                    "actions_count": 3, "credits_consumed": 500},
    })
    assert data.summary.duration
    assert data.summary.credits_consumed


def test_input_state_reports_the_composer_and_its_reason():
    """Blocked or not, the reason is on the event — never inferred from which
    other events arrived."""
    data = InputState.model_validate(load_fixture("input_state")["payload"]["data"])
    assert data.content
    assert data.accepting_input is not data.blocked
    if data.blocked:
        assert data.code or data.detail


def test_input_state_codes_name_the_two_conditions_worth_handling():
    """Constructed, not recorded: an ordinary session reaches neither."""
    awaiting = InputState(content="input_disabled", code=InputStateCode.TASK_READY)
    assert awaiting.awaiting_credits and not awaiting.needs_phone_verification

    unverified = InputState(
        content="input_disabled", code=InputStateCode.PHONE_VERIFICATION_REQUIRED,
    )
    assert unverified.needs_phone_verification and not unverified.awaiting_credits


def test_llm_thinking_is_typed():
    """A reasoning step always declares its type. Search has no event of its
    own — it appears here as a `tool_call` step, whose fields are pinned below."""
    data = LLMThinkingData.model_validate(load_fixture("llm_thinking")["payload"]["data"])
    assert data.type


def test_llm_thinking_carries_tool_calls():
    """Constructed: whether a turn produces a tool_call step is not ours to
    arrange."""
    data = LLMThinkingData.model_validate({
        "type": "tool_call", "tool_name": "web_search", "status": "completed",
        "history": [{"tool_name": "web_search", "status": "completed"}],
    })
    assert data.type == "tool_call"
    assert data.tool_name
    assert data.history


def test_rich_content_carries_its_own_body():
    """Not repeated in session:text — ignoring it loses the content."""
    data = RichContentData.model_validate(load_fixture("rich_content")["payload"]["data"])
    assert data.title
    assert data.content


def test_message_status_carries_a_status_and_its_reason():
    """The only way to tell a rejected or rate-limited message from one still
    being processed."""
    data = MessageStatusData.model_validate(load_fixture("message_status")["payload"]["data"])
    assert data.status
    assert data.request_id


def test_restriction_states_the_task_will_not_complete():
    data = RestrictionData.model_validate(load_fixture("restriction")["payload"]["data"])
    assert data.level
    assert data.message


def test_required_action_reports_whether_a_response_is_awaited():
    data = RequiredActionData.model_validate(load_fixture("required_action")["payload"]["data"])
    assert data.is_required_action is True


def test_task_ready_carries_the_credit_cost():
    """`confirmed` is the authorization state, not a request for one: when the
    balance covers `required` the server starts the task itself and this event
    is informational. It waits only when the balance does not."""
    data = TaskReadyData.model_validate(load_fixture("task_ready")["payload"]["data"])
    assert data.required > 0
    assert isinstance(data.confirmed, bool)


def test_form_to_user_carries_its_fields():
    data = FormToUserData.model_validate(load_fixture("form_to_user")["payload"]["data"])
    assert data.message_to_user
    assert data.form.fields
    assert data.form.fields[0].name


def test_history_page_is_a_message_list():
    """Exhaustion is read from the cursor. An absent cursor is the end — and an
    empty page is not, which is why `messages` may be null while more remain."""
    data = load_fixture("history")["payload"]["data"]
    assert "messages" in data


def test_unknown_payload_fields_are_tolerated():
    """Payloads may gain fields at any time."""
    envelope = load_fixture("text")
    envelope["payload"]["data"]["a_field_added_next_week"] = {"nested": True}
    envelope["metadata"]["some_new_metadata"] = 1
    assert parse_envelope(envelope) is not None


def test_fixture_files_are_stable_json():
    for path in sorted(pathlib.Path(FIXTURES).glob("*.json")):
        json.loads(path.read_text())


def test_recording_leaves_identifiers_intact():
    """Redaction rewrites values that identify a person. An identifier is an
    opaque number, and rewriting one destroys the shape being recorded."""
    for event_type, envelope in sorted(all_fixtures().items()):
        event_id = envelope["metadata"]["event_id"]
        assert not event_id.startswith("+"), f"{event_type}: event_id was redacted"
        message_id = envelope["payload"].get("message_id")
        if message_id:
            assert not str(message_id).startswith("+"), f"{event_type}: message_id was redacted"


def test_recorded_forms_carry_no_answers():
    """A form is where the user's own data lives — the details it asks for are
    prefilled from their profile, the placeholders are built from those values,
    and the server writes choice labels in their voice ("I (Name) prefer ...").
    Redaction has to reach all three; grepping for what leaked last time does
    not generalise, so this asserts the shape instead.
    """
    provenance = json.loads((FIXTURES / "provenance.json").read_text())
    for event_type, envelope in sorted(all_fixtures().items()):
        if provenance[event_type]["source"] != "recorded":
            continue
        fields = (((envelope["payload"].get("data") or {}).get("form") or {})
                  .get("fields") or [])
        for field in fields:
            for key in ("prefilled", "placeholder"):
                if field.get(key):
                    assert field[key] == "[redacted]", f"{event_type}.{field.get('name')}.{key}"
            for option in field.get("options") or []:
                assert option == "[redacted]", f"{event_type}.{field.get('name')}.options"
