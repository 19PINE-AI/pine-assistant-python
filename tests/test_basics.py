"""Basic unit tests for pine-assistant package."""

from pine_assistant import (
    AsyncPineAI,
    PineAI,
    PineAIError,
    AuthError,
    SessionError,
    ConnectionError,
    C2SEvent,
    S2CEvent,
    __version__,
)
from pine_assistant.models.events import NotificationEvent


def test_version():
    assert __version__ == "0.1.1"


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
    assert NotificationEvent.NEW_MESSAGE == "notification:new_message"
