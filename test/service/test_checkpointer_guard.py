"""
Tests for the global checkpointer accessor.

Guards the fix that turns an opaque KeyError (when a request arrives before the
app lifespan initialized the shared saver) into an explicit RuntimeError.
"""
import pytest

import service.checkpointer as cp
from service.checkpointer import get_global_checkpointer


def test_raises_clear_error_when_not_initialized(monkeypatch):
    monkeypatch.setattr(cp, "checkpointers", {}, raising=True)
    with pytest.raises(RuntimeError):
        get_global_checkpointer()


def test_returns_saver_when_initialized(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(cp, "checkpointers", {"aiosql": sentinel}, raising=True)
    assert get_global_checkpointer() is sentinel
