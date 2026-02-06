"""Tests for DiagnosticsService in app/services/diagnostics_service.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agno.run.agent import RunOutput

from app.services.diagnostics_service import DiagnosticsOutput, DiagnosticsService


# --- Initialization ---


@pytest.mark.asyncio
async def test_initialize():
    service = DiagnosticsService()
    with (
        patch("app.services.diagnostics_service.DuckDuckGoTools") as MockDDG,
        patch("app.services.diagnostics_service.SQLTools"),
        patch("app.services.diagnostics_service.Agent") as MockAgent,
    ):
        await service.initialize()

    assert service._initialized is True
    MockDDG.assert_called_once()
    MockAgent.assert_called_once()


@pytest.mark.asyncio
async def test_initialize_idempotent():
    service = DiagnosticsService()
    with (
        patch("app.services.diagnostics_service.DuckDuckGoTools"),
        patch("app.services.diagnostics_service.SQLTools"),
        patch("app.services.diagnostics_service.Agent") as MockAgent,
    ):
        await service.initialize()
        await service.initialize()

    MockAgent.assert_called_once()


@pytest.mark.asyncio
async def test_cleanup(diagnostics_service_instance):
    await diagnostics_service_instance.cleanup()
    assert diagnostics_service_instance._initialized is False


# --- Diagnose ---


@pytest.mark.asyncio
async def test_diagnose_with_pydantic_output(diagnostics_service_instance):
    output = DiagnosticsOutput(diagnostics=["Bad battery", "Faulty starter"])
    diagnostics_service_instance.agent.arun = AsyncMock(
        return_value=RunOutput(content=output)
    )

    result = await diagnostics_service_instance.diagnose(
        message="won't start", listing_id="EQP-1", session_id="s1"
    )

    assert result["diagnostics"] == ["Bad battery", "Faulty starter"]
    assert result["listing_id"] == "EQP-1"
    assert result["session_id"] == "s1"
    assert isinstance(result["execution_time"], float)


@pytest.mark.asyncio
async def test_diagnose_with_json_string(diagnostics_service_instance):
    diagnostics_service_instance.agent.arun = AsyncMock(
        return_value=RunOutput(
            content='{"diagnostics": ["Low oil", "Filter clogged"]}'
        )
    )

    result = await diagnostics_service_instance.diagnose(
        message="overheating", listing_id="EQP-2", session_id="s2"
    )

    assert result["diagnostics"] == ["Low oil", "Filter clogged"]


@pytest.mark.asyncio
async def test_diagnose_with_unparseable_content(diagnostics_service_instance):
    diagnostics_service_instance.agent.arun = AsyncMock(
        return_value=RunOutput(content="not json at all")
    )

    result = await diagnostics_service_instance.diagnose(
        message="noise", listing_id="EQP-3", session_id="s3"
    )

    assert result["diagnostics"] == []


@pytest.mark.asyncio
async def test_diagnose_with_no_content(diagnostics_service_instance):
    diagnostics_service_instance.agent.arun = AsyncMock(
        return_value=MagicMock(spec=[], content=None)  # no 'content' attr via spec=[]
    )
    # Patch hasattr behavior — RunOutput without content
    mock_response = MagicMock()
    del mock_response.content  # hasattr(response, 'content') → False
    diagnostics_service_instance.agent.arun = AsyncMock(return_value=mock_response)

    result = await diagnostics_service_instance.diagnose(
        message="noise", listing_id="EQP-3", session_id="s3"
    )

    assert result["diagnostics"] == []


@pytest.mark.asyncio
async def test_diagnose_generates_session_id(diagnostics_service_instance):
    output = DiagnosticsOutput(diagnostics=["Issue 1"])
    diagnostics_service_instance.agent.arun = AsyncMock(
        return_value=RunOutput(content=output)
    )

    result = await diagnostics_service_instance.diagnose(
        message="problem", listing_id="EQP-1"
    )

    assert result["session_id"]
    assert len(result["session_id"]) == 36


@pytest.mark.asyncio
async def test_diagnose_includes_listing_id_in_agent_input(
    diagnostics_service_instance,
):
    output = DiagnosticsOutput(diagnostics=["Issue 1"])
    diagnostics_service_instance.agent.arun = AsyncMock(
        return_value=RunOutput(content=output)
    )

    await diagnostics_service_instance.diagnose(
        message="problem", listing_id="EQP-42", session_id="s1"
    )

    call_kwargs = diagnostics_service_instance.agent.arun.call_args.kwargs
    assert "EQP-42" in call_kwargs["input"]


@pytest.mark.asyncio
async def test_diagnose_creates_fresh_sql_tools(
    diagnostics_service_instance, mock_sql_tools
):
    output = DiagnosticsOutput(diagnostics=["Issue 1"])
    diagnostics_service_instance.agent.arun = AsyncMock(
        return_value=RunOutput(content=output)
    )

    await diagnostics_service_instance.diagnose(
        message="problem", listing_id="EQP-1", session_id="s1"
    )

    diagnostics_service_instance._create_sql_tools.assert_called_once()


@pytest.mark.asyncio
async def test_diagnose_error_propagates(diagnostics_service_instance):
    diagnostics_service_instance.agent.arun = AsyncMock(
        side_effect=RuntimeError("LLM error")
    )

    with pytest.raises(RuntimeError, match="LLM error"):
        await diagnostics_service_instance.diagnose(
            message="broken", listing_id="EQP-1", session_id="s1"
        )
