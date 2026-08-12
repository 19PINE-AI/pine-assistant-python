"""Basic unit tests for pine-assistant package."""

from pine_assistant import (
    SUPPORTED_EVENTS,
    AsyncPineAI,
    AuthError,
    C2SEvent,
    ConnectionError,
    PineAI,
    PineAIError,
    S2CEvent,
    SessionError,
    is_supported_event,
)


def test_public_exports():
    assert PineAI is not None
    assert AsyncPineAI is not None


def test_error_hierarchy():
    assert issubclass(AuthError, PineAIError)
    assert issubclass(SessionError, PineAIError)
    assert issubclass(ConnectionError, PineAIError)


def test_error_attributes():
    err = PineAIError(code="test_code", message="something broke")
    assert err.code == "test_code"
    assert str(err) == "something broke"
    assert err.details is None

    err_with_details = SessionError("bad session", details={"id": "123"})
    assert err_with_details.code == "session_error"
    assert err_with_details.details == {"id": "123"}


def test_event_constants():
    assert C2SEvent.SESSION_MESSAGE == "session:message"
    assert S2CEvent.SESSION_TEXT == "session:text"
    assert S2CEvent.SESSION_LLM_THINKING == "session:llm_thinking"
    assert S2CEvent.SESSION_TOOL_STATUS == "session:tool_status"


def test_only_the_supported_surface_is_modelled():
    """The event constants are the protocol scope, not an inventory of what the
    server emits."""
    assert len(list(S2CEvent)) == 16
    assert set(C2SEvent) <= set(SUPPORTED_EVENTS)


def test_is_supported_event_separates_the_two_surfaces():
    assert is_supported_event("session:text")
    assert is_supported_event("session:llm_thinking")
    # Emitted by the server, and reachable — but not maintained.
    assert not is_supported_event("session:work_log")
    assert not is_supported_event("session:payment")
    assert not is_supported_event("session:an_event_from_the_future")
