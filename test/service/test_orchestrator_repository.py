"""
Concurrency and robustness tests for OrchestratorRepository.

These guard the fix that put both internal maps behind a lock and made the
periodic cleanup use defensive pops, so background cleanup racing live request
handling can neither diverge the two maps nor raise a KeyError that kills the
cleanup loop.
"""
import threading
from types import SimpleNamespace

import pytest

import service.streaming.orchestrator_repository as orch_mod
from service.streaming.orchestrator_repository import OrchestratorRepository


class _FakeOrchestrator:
    """Lightweight stand-in so the tests exercise the repository logic, not the
    real StreamEventOrchestrator (which pulls in the whole agent stack)."""

    def __init__(self, stream_input):
        self.stream_input = stream_input
        self.processors = []
        self.resets = 0

    def reset_for_new_request(self):
        self.resets += 1


def _stream_input(approval=None):
    return SimpleNamespace(tool_call_approval=approval)


@pytest.fixture(autouse=True)
def _patch_orchestrator(monkeypatch):
    monkeypatch.setattr(orch_mod, "StreamEventOrchestrator", _FakeOrchestrator)


def test_creates_then_reuses_and_resets_on_new_user_message():
    repo = OrchestratorRepository()
    first = repo.get_or_create("t1", _stream_input())
    second = repo.get_or_create("t1", _stream_input())
    assert first is second  # same instance reused for the same thread
    assert first.resets == 1  # reset on the second (new user message)


def test_approval_response_does_not_reset_state():
    repo = OrchestratorRepository()
    repo.get_or_create("t1", _stream_input())
    orch = repo.get_or_create("t1", _stream_input(approval=object()))
    assert orch.resets == 0  # approval continuation must keep existing state


def test_cleanup_is_defensive_and_keeps_maps_consistent():
    repo = OrchestratorRepository()
    repo.get_or_create("old", _stream_input())
    repo.get_or_create("fresh", _stream_input())

    # Force "old" to look stale.
    repo._last_access["old"] = 0.0
    removed = repo.cleanup_inactive(max_age_seconds=1)

    assert removed == 1
    assert "old" not in repo._orchestrators and "old" not in repo._last_access
    assert "fresh" in repo._orchestrators
    # Running again removes nothing and must not raise.
    assert repo.cleanup_inactive(max_age_seconds=1) == 0


def test_concurrent_get_or_create_and_cleanup_never_raises():
    repo = OrchestratorRepository()
    errors: list[Exception] = []

    def worker(worker_id: int):
        try:
            for i in range(200):
                thread_id = f"t{(worker_id * 200 + i) % 25}"
                repo.get_or_create(thread_id, _stream_input())
                if i % 10 == 0:
                    # Age everything and clean while other workers are creating.
                    for key in list(repo._last_access):
                        repo._last_access[key] = 0.0
                    repo.cleanup_inactive(max_age_seconds=0)
        except Exception as exc:  # noqa: BLE001 - surface any race failure
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent access raised: {errors[:3]}"
    # The lock guarantees both maps are mutated together, so their key sets match.
    assert set(repo._orchestrators) == set(repo._last_access)
