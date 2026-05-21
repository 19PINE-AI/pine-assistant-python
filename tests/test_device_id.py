"""Unit tests for device_id resolution — env var, explicit arg, file persistence."""

from pathlib import Path
from unittest.mock import patch

import pine_assistant.client as client_module
from pine_assistant.client import _get_or_create_device_id


def test_explicit_arg_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("PINE_DEVICE_ID", "from-env")
    monkeypatch.setattr(client_module, "DEVICE_ID_FILE", tmp_path / "device_id")
    assert _get_or_create_device_id("explicit") == "explicit"


def test_env_var_wins_over_file(monkeypatch, tmp_path):
    device_file = tmp_path / "device_id"
    device_file.write_text("from-file")
    monkeypatch.setenv("PINE_DEVICE_ID", "from-env")
    monkeypatch.setattr(client_module, "DEVICE_ID_FILE", device_file)
    assert _get_or_create_device_id() == "from-env"


def test_env_var_is_stripped(monkeypatch, tmp_path):
    monkeypatch.setenv("PINE_DEVICE_ID", "  trimmed  \n")
    monkeypatch.setattr(client_module, "DEVICE_ID_FILE", tmp_path / "device_id")
    assert _get_or_create_device_id() == "trimmed"


def test_falls_back_to_file(monkeypatch, tmp_path):
    monkeypatch.delenv("PINE_DEVICE_ID", raising=False)
    device_file = tmp_path / "device_id"
    device_file.write_text("from-file")
    monkeypatch.setattr(client_module, "DEVICE_ID_FILE", device_file)
    assert _get_or_create_device_id() == "from-file"


def test_creates_and_persists_when_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("PINE_DEVICE_ID", raising=False)
    device_file = tmp_path / "nested" / "device_id"
    monkeypatch.setattr(client_module, "DEVICE_ID_FILE", device_file)
    first = _get_or_create_device_id()
    assert device_file.read_text() == first
    # A second call returns the persisted value, not a new UUID.
    assert _get_or_create_device_id() == first


def test_unwritable_dir_warns_and_returns_random(monkeypatch, tmp_path, caplog):
    monkeypatch.delenv("PINE_DEVICE_ID", raising=False)
    # Point at an unwritable location: a file used as a "directory".
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    monkeypatch.setattr(client_module, "DEVICE_ID_FILE", blocker / "device_id")
    with caplog.at_level("WARNING", logger="pine_assistant.client"):
        device_id = _get_or_create_device_id()
    assert device_id  # got *some* id back
    assert any("PINE_DEVICE_ID" in rec.message for rec in caplog.records)
