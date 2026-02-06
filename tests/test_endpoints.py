"""Tests for FastAPI endpoints in app/main.py."""

import json
from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_root(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["service"] == "Agno Agent API"


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy"}


@pytest.mark.asyncio
async def test_chat(client):
    mock_result = {"response": "Hello there", "session_id": "sess-123"}
    with patch(
        "app.main.agno_service.chat", new_callable=AsyncMock, return_value=mock_result
    ):
        resp = await client.post(
            "/chat", json={"message": "Hi", "user_id": "user1"}
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["response"] == "Hello there"
    assert data["session_id"] == "sess-123"


@pytest.mark.asyncio
async def test_chat_missing_message(client):
    resp = await client.post("/chat", json={})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_chat_default_user_id(client):
    mock_result = {"response": "ok", "session_id": "s1"}
    with patch(
        "app.main.agno_service.chat", new_callable=AsyncMock, return_value=mock_result
    ) as mock_chat:
        resp = await client.post("/chat", json={"message": "test"})
    assert resp.status_code == 200
    call_kwargs = mock_chat.call_args.kwargs
    assert call_kwargs["user_id"] == "default"


@pytest.mark.asyncio
async def test_chat_service_error(client):
    with patch(
        "app.main.agno_service.chat",
        new_callable=AsyncMock,
        side_effect=RuntimeError("LLM down"),
    ):
        resp = await client.post(
            "/chat", json={"message": "Hi", "user_id": "u1"}
        )
    assert resp.status_code == 500


@pytest.mark.asyncio
async def test_chat_stream(client):
    async def fake_stream(message, session_id, user_id):
        yield f"data: {json.dumps({'type': 'session', 'session_id': 'sess-1'})}\n\n"
        yield f"data: {json.dumps({'type': 'content', 'content': 'Hello'})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'execution_time': 0.5})}\n\n"

    with patch("app.main.agno_service.chat_stream", side_effect=fake_stream):
        resp = await client.post(
            "/chat/stream", json={"message": "Hi", "user_id": "u1"}
        )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    assert '"type": "session"' in body
    assert '"type": "content"' in body
    assert '"type": "done"' in body


@pytest.mark.asyncio
async def test_diagnostics(client):
    mock_result = {
        "diagnostics": ["Bad battery", "Faulty starter"],
        "listing_id": "EQP-1",
        "session_id": "ds-1",
        "execution_time": 1.2,
    }
    with patch(
        "app.main.diagnostics_service.diagnose",
        new_callable=AsyncMock,
        return_value=mock_result,
    ):
        resp = await client.post(
            "/diagnostics",
            json={"message": "won't start", "listing_id": "EQP-1", "user_id": "u1"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["diagnostics"] == ["Bad battery", "Faulty starter"]
    assert data["listing_id"] == "EQP-1"
    assert data["execution_time"] == 1.2


@pytest.mark.asyncio
async def test_diagnostics_missing_listing_id(client):
    resp = await client.post(
        "/diagnostics", json={"message": "won't start"}
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_diagnostics_service_error(client):
    with patch(
        "app.main.diagnostics_service.diagnose",
        new_callable=AsyncMock,
        side_effect=RuntimeError("DB gone"),
    ):
        resp = await client.post(
            "/diagnostics",
            json={"message": "broken", "listing_id": "EQP-1", "user_id": "u1"},
        )
    assert resp.status_code == 500
