"""
Integration tests for the Redis-backed session store.

Unlike test/session/backends/test_redis_backend.py (which mocks the client and
also imports a stale module path), these exercise RedisBackend against a REAL
Redis instance and round-trip an actual SessionData model through it.

Requires a reachable Redis (REDIS_URL, default redis://localhost:6379). If none
is available the tests skip rather than fail, so the file is safe to collect
anywhere; CI runs them against a Redis service container.
"""
import os
import uuid

import pytest
import pytest_asyncio
import redis.asyncio as aioredis
from fastapi_sessions.backends.session_backend import BackendError

# Importing auth.session triggers service.__init__ -> service.service, which in
# turn imports auth.session; importing service.service first resolves that
# circular import deterministically (the app's own startup path does the same).
import service.service  # noqa: F401,E402

from auth.session import SessionData  # noqa: E402
from auth.session.backends.redis_backend import RedisBackend  # noqa: E402

pytestmark = pytest.mark.asyncio

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


@pytest_asyncio.fixture
async def backend():
    client = aioredis.from_url(REDIS_URL, decode_responses=True)
    try:
        await client.ping()
    except Exception as exc:  # noqa: BLE001 - any connection failure => skip
        pytest.skip(f"Redis not available at {REDIS_URL}: {exc}")
    try:
        yield RedisBackend[uuid.UUID, SessionData](client, SessionData)
    finally:
        close = getattr(client, "aclose", None) or client.close
        await close()


async def _cleanup(backend, session_id):
    try:
        await backend.delete(session_id)
    except BackendError:
        pass


async def test_create_read_roundtrip(backend):
    session_id = uuid.uuid4()
    data = SessionData(
        consent_id="consent-1",
        is_anonymous=False,
        token_usage=5,
        request_count=2,
        user_id="user-1",
        bearer_token="tok-1",
    )
    try:
        await backend.create(session_id, data)
        got = await backend.read(session_id)
        assert got.user_id == "user-1"
        assert got.is_anonymous is False
        assert got.token_usage == 5
        assert got.request_count == 2
        assert got.consent_id == "consent-1"
        assert got.bearer_token == "tok-1"
    finally:
        await _cleanup(backend, session_id)


async def test_read_missing_raises(backend):
    with pytest.raises(BackendError):
        await backend.read(uuid.uuid4())


async def test_update_roundtrip(backend):
    session_id = uuid.uuid4()
    await backend.create(
        session_id,
        SessionData(is_anonymous=True, token_usage=0, request_count=0),
    )
    try:
        await backend.update(
            session_id,
            SessionData(
                is_anonymous=False,
                token_usage=10,
                request_count=3,
                user_id="user-2",
                bearer_token="tok-2",
                consent_id="consent-2",
            ),
        )
        got = await backend.read(session_id)
        assert got.token_usage == 10
        assert got.request_count == 3
        assert got.user_id == "user-2"
        assert got.is_anonymous is False
    finally:
        await _cleanup(backend, session_id)


async def test_update_missing_raises(backend):
    with pytest.raises(BackendError):
        await backend.update(uuid.uuid4(), SessionData(is_anonymous=True))


async def test_delete_then_read_missing(backend):
    session_id = uuid.uuid4()
    await backend.create(session_id, SessionData(is_anonymous=True))
    await backend.delete(session_id)
    with pytest.raises(BackendError):
        await backend.read(session_id)


async def test_delete_missing_raises(backend):
    with pytest.raises(BackendError):
        await backend.delete(uuid.uuid4())
