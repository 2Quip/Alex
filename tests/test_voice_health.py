"""Tests for app/voice_health.py."""

import json

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, TestClient, TestServer

from app.voice_health import VoiceAgentHealth, _health_handler


@pytest.fixture
def tracker():
    """Fresh VoiceAgentHealth instance per test."""
    return VoiceAgentHealth()


# --- VoiceAgentHealth state tests ---


def test_initial_state(tracker):
    snap = tracker.snapshot()
    assert snap["status"] == "starting"
    assert snap["active_sessions"] == 0
    assert snap["error"] is None
    assert snap["uptime_seconds"] >= 0


def test_mark_running(tracker):
    tracker.mark_running()
    snap = tracker.snapshot()
    assert snap["status"] == "running"
    assert snap["error"] is None


def test_mark_error(tracker):
    tracker.mark_error("connection lost")
    snap = tracker.snapshot()
    assert snap["status"] == "error"
    assert snap["error"] == "connection lost"


def test_mark_running_clears_error(tracker):
    tracker.mark_error("bad thing")
    tracker.mark_running()
    snap = tracker.snapshot()
    assert snap["status"] == "running"
    assert snap["error"] is None


def test_session_counting(tracker):
    tracker.session_started()
    tracker.session_started()
    assert tracker.snapshot()["active_sessions"] == 2
    tracker.session_ended()
    assert tracker.snapshot()["active_sessions"] == 1
    tracker.session_ended()
    assert tracker.snapshot()["active_sessions"] == 0


def test_session_ended_no_negative(tracker):
    tracker.session_ended()
    tracker.session_ended()
    assert tracker.snapshot()["active_sessions"] == 0


# --- HTTP handler tests ---


@pytest.fixture
def health_app():
    app = web.Application()
    app.router.add_get("/health", _health_handler)
    return app


@pytest_asyncio.fixture
async def health_client(health_app, aiohttp_client):
    return await aiohttp_client(health_app)


@pytest.mark.asyncio
async def test_health_endpoint_503_when_starting(health_client, monkeypatch):
    """Before mark_running(), the handler should return 503."""
    # The module-level `health` singleton starts in "starting" state.
    # We import it to ensure it hasn't been mutated by other tests.
    from app.voice_health import health

    # Reset to starting state for deterministic test
    with health._lock:
        health._status = "starting"
        health._error = None

    resp = await health_client.get("/health")
    assert resp.status == 503
    body = await resp.json()
    assert body["status"] == "starting"


@pytest.mark.asyncio
async def test_health_endpoint_200_when_running(health_client):
    from app.voice_health import health

    health.mark_running()
    resp = await health_client.get("/health")
    assert resp.status == 200
    body = await resp.json()
    assert body["status"] == "running"
    assert "uptime_seconds" in body
    assert "active_sessions" in body
