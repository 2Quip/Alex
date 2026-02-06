"""Tests for AgnoService in app/services/agno_service.py."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agno.agent import (
    RunCompletedEvent,
    RunContentEvent,
    RunErrorEvent,
    RunStartedEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)
from agno.run.agent import RunOutput

from app.services.agno_service import AgnoService


# --- Initialization ---


@pytest.mark.asyncio
async def test_initialize():
    service = AgnoService()
    with (
        patch("app.services.agno_service.DuckDuckGoTools") as MockDDG,
        patch("app.services.agno_service.SQLTools") as MockSQL,
        patch("app.services.agno_service.Agent") as MockAgent,
    ):
        await service.initialize()

    assert service._initialized is True
    assert service.agent is not None
    MockDDG.assert_called_once()
    MockAgent.assert_called_once()


@pytest.mark.asyncio
async def test_initialize_idempotent():
    service = AgnoService()
    with (
        patch("app.services.agno_service.DuckDuckGoTools"),
        patch("app.services.agno_service.SQLTools"),
        patch("app.services.agno_service.Agent") as MockAgent,
    ):
        await service.initialize()
        await service.initialize()

    MockAgent.assert_called_once()


@pytest.mark.asyncio
async def test_initialize_error():
    service = AgnoService()
    with (
        patch(
            "app.services.agno_service.DuckDuckGoTools",
            side_effect=RuntimeError("fail"),
        ),
        pytest.raises(RuntimeError, match="fail"),
    ):
        await service.initialize()

    assert service._initialized is False


@pytest.mark.asyncio
async def test_cleanup(agno_service_instance):
    await agno_service_instance.cleanup()
    assert agno_service_instance._initialized is False


@pytest.mark.asyncio
async def test_ensure_initialized_calls_init():
    service = AgnoService()
    with (
        patch("app.services.agno_service.DuckDuckGoTools"),
        patch("app.services.agno_service.SQLTools"),
        patch("app.services.agno_service.Agent"),
    ):
        await service.ensure_initialized()

    assert service._initialized is True


# --- Chat ---


@pytest.mark.asyncio
async def test_chat_returns_response(agno_service_instance):
    result = await agno_service_instance.chat(message="Hello", session_id="s1")
    assert result["response"] == "Test response"
    assert result["session_id"] == "s1"


@pytest.mark.asyncio
async def test_chat_generates_session_id(agno_service_instance):
    result = await agno_service_instance.chat(message="Hi")
    assert result["session_id"]  # non-empty UUID string
    assert len(result["session_id"]) == 36  # UUID format


@pytest.mark.asyncio
async def test_chat_preserves_session_id(agno_service_instance):
    result = await agno_service_instance.chat(message="Hi", session_id="my-session")
    assert result["session_id"] == "my-session"


@pytest.mark.asyncio
async def test_chat_creates_fresh_sql_tools(agno_service_instance, mock_sql_tools):
    await agno_service_instance.chat(message="Hi", session_id="s1")
    agno_service_instance._create_sql_tools.assert_called_once()
    assert mock_sql_tools in agno_service_instance.agent.tools


@pytest.mark.asyncio
async def test_chat_empty_content(agno_service_instance):
    agno_service_instance.agent.arun = AsyncMock(
        return_value=RunOutput(content=None)
    )
    result = await agno_service_instance.chat(message="Hi", session_id="s1")
    assert result["response"] == ""


@pytest.mark.asyncio
async def test_chat_error_propagates(agno_service_instance):
    agno_service_instance.agent.arun = AsyncMock(
        side_effect=RuntimeError("API error")
    )
    with pytest.raises(RuntimeError, match="API error"):
        await agno_service_instance.chat(message="Hi", session_id="s1")


# --- Chat Stream ---


@pytest.mark.asyncio
async def test_chat_stream_yields_events(agno_service_instance):
    async def fake_stream(*args, **kwargs):
        yield RunStartedEvent()
        yield RunContentEvent(content="Hello world")
        yield RunCompletedEvent()

    agno_service_instance.agent.arun = MagicMock(return_value=fake_stream())

    chunks = []
    async for chunk in agno_service_instance.chat_stream(message="Hi", session_id="s1"):
        chunks.append(chunk)

    # First chunk: session event
    session_data = json.loads(chunks[0].removeprefix("data: ").strip())
    assert session_data["type"] == "session"
    assert session_data["session_id"] == "s1"

    # Find content chunk
    content_chunks = [
        c for c in chunks if '"type": "content"' in c
    ]
    assert len(content_chunks) == 1
    content_data = json.loads(content_chunks[0].removeprefix("data: ").strip())
    assert content_data["content"] == "Hello world"

    # Last chunk: done event
    done_data = json.loads(chunks[-1].removeprefix("data: ").strip())
    assert done_data["type"] == "done"
    assert "execution_time" in done_data


@pytest.mark.asyncio
async def test_chat_stream_tool_events(agno_service_instance):
    tool_started = MagicMock(spec=ToolCallStartedEvent)
    tool_started.tool = MagicMock()
    tool_started.tool.tool_name = "duckduckgo_search"
    tool_started.tool.tool_args = "test query"

    tool_completed = MagicMock(spec=ToolCallCompletedEvent)
    tool_completed.tool = MagicMock()
    tool_completed.tool.tool_name = "duckduckgo_search"
    tool_completed.tool.result = "search results"

    async def fake_stream(*args, **kwargs):
        yield tool_started
        yield tool_completed

    agno_service_instance.agent.arun = MagicMock(return_value=fake_stream())

    chunks = []
    async for chunk in agno_service_instance.chat_stream(message="search", session_id="s1"):
        chunks.append(chunk)

    tool_start_chunks = [c for c in chunks if '"type": "tool_start"' in c]
    assert len(tool_start_chunks) == 1

    tool_complete_chunks = [c for c in chunks if '"type": "tool_complete"' in c]
    assert len(tool_complete_chunks) == 1


@pytest.mark.asyncio
async def test_chat_stream_error_event(agno_service_instance):
    error_event = MagicMock(spec=RunErrorEvent)
    error_event.error = "Something went wrong"

    async def fake_stream(*args, **kwargs):
        yield error_event

    agno_service_instance.agent.arun = MagicMock(return_value=fake_stream())

    chunks = []
    async for chunk in agno_service_instance.chat_stream(message="Hi", session_id="s1"):
        chunks.append(chunk)

    error_chunks = [c for c in chunks if '"type": "error"' in c]
    assert len(error_chunks) == 1


@pytest.mark.asyncio
async def test_chat_stream_exception_yields_error(agno_service_instance):
    async def failing_stream(*args, **kwargs):
        raise RuntimeError("stream broke")
        yield  # noqa: unreachable — makes this an async generator

    agno_service_instance.agent.arun = MagicMock(return_value=failing_stream())

    chunks = []
    async for chunk in agno_service_instance.chat_stream(message="Hi", session_id="s1"):
        chunks.append(chunk)

    error_chunks = [c for c in chunks if '"type": "error"' in c]
    assert len(error_chunks) == 1
    error_data = json.loads(error_chunks[0].removeprefix("data: ").strip())
    assert "stream broke" in error_data["error"]
