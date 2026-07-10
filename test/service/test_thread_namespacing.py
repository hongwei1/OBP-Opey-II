"""
Regression tests for session-private thread namespacing.

These lock in the IDOR fix: every client-supplied thread_id is confined to the
caller's own session namespace, so one session can neither read, resume, stop
nor regenerate another session's thread by supplying its id.
"""
from service.opey_session import compute_effective_thread_id, compute_thread_namespace

# Two distinct session ids (attacker A, victim B).
A = "11111111-1111-1111-1111-111111111111"
B = "22222222-2222-2222-2222-222222222222"


def test_none_maps_to_stable_default():
    assert compute_effective_thread_id(A, None) == compute_effective_thread_id(A, None)
    assert compute_effective_thread_id(A, None).endswith("::default")


def test_namespacing_is_idempotent():
    once = compute_effective_thread_id(A, "conv-1")
    twice = compute_effective_thread_id(A, once)
    assert once == twice  # an already-namespaced id is not double-prefixed


def test_cross_session_isolation():
    victim_key = compute_effective_thread_id(B, "conv-1")
    # Attacker A replays the victim's handle; it must not resolve to B's key.
    assert compute_effective_thread_id(A, victim_key) != victim_key


def test_raw_session_id_replay_is_blocked():
    # Even if B's raw session id leaks, A cannot use it to reach B's threads.
    assert compute_effective_thread_id(A, B) != compute_effective_thread_id(B, None)
    assert compute_effective_thread_id(A, B) != compute_effective_thread_id(B, B)


def test_distinct_client_threads_get_distinct_keys():
    assert compute_effective_thread_id(A, "x") != compute_effective_thread_id(A, "y")


def test_handle_does_not_leak_raw_session_id():
    assert A not in compute_effective_thread_id(A, "x")
    assert compute_thread_namespace(A) != compute_thread_namespace(B)
