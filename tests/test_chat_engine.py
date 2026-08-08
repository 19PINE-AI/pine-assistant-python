"""Unit tests for the chat engine's pure parts.

Turn behaviour is covered in tests/protocol/test_flow.py, where events travel
the real transport instead of being handed straight to a handler.
"""

from pine_assistant.chat import (
    SUBSTANTIVE_EVENTS,
    ChatEvent,
    Deduplicator,
    event_from_envelope,
)
from pine_assistant.models.events import SUPPORTED_EVENTS


def _event(event_id, event_type="session:text"):
    return ChatEvent(type=event_type, session_id="s1", data={}, event_id=event_id)


class TestDeduplicator:
    def test_second_sighting_is_a_duplicate(self):
        dedup = Deduplicator()
        assert dedup.is_duplicate(_event("e1")) is False
        assert dedup.is_duplicate(_event("e1")) is True

    def test_same_id_different_type_is_not_a_duplicate(self):
        """Keying on the identifier alone would drop the second event."""
        dedup = Deduplicator()
        assert dedup.is_duplicate(_event("e1", "session:text")) is False
        assert dedup.is_duplicate(_event("e1", "session:update_title")) is False

    def test_events_without_an_identifier_are_never_suppressed(self):
        dedup = Deduplicator()
        assert dedup.is_duplicate(_event(None)) is False
        assert dedup.is_duplicate(_event(None)) is False


class TestEventFromEnvelope:
    def test_carries_the_payload_through_untouched(self):
        raw = {
            "metadata": {"event_id": "e1", "timestamp": "2026-08-08T00:00:00Z"},
            "type": "session:whatever",
            "payload": {"session_id": "s1", "message_id": "m1", "data": {"deep": [1, 2]}},
        }
        event = event_from_envelope("session:whatever", raw, "s1")
        assert event.type == "session:whatever"
        assert event.event_id == "e1"
        assert event.message_id == "m1"
        assert event.data == {"deep": [1, 2]}
        assert event.metadata == raw["metadata"]

    def test_tolerates_a_missing_payload_and_metadata(self):
        event = event_from_envelope("session:text", {}, "s1")
        assert event.data is None
        assert event.event_id is None


def test_turn_control_uses_only_supported_events():
    """A turn must not begin or end on an event we do not maintain."""
    assert {e.value for e in SUBSTANTIVE_EVENTS} <= SUPPORTED_EVENTS
