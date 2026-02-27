"""Tests for DiagnosticsService in app/services/diagnostics_service.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agno.run.agent import RunOutput

from app.services.diagnostics_service import DiagnosticsService, _parse_diagnostics


# --- _parse_diagnostics ---


def test_parse_diagnostics_blank_line_split():
    text = "Most likely: Bad battery.\n\nPossible: Faulty starter."
    result = _parse_diagnostics(text)
    assert result == ["Most likely: Bad battery.", "Possible: Faulty starter."]


def test_parse_diagnostics_numbered_split():
    text = "1. Bad battery cause and fix.\n2. Faulty starter cause and fix."
    result = _parse_diagnostics(text)
    assert result == ["Bad battery cause and fix.", "Faulty starter cause and fix."]


def test_parse_diagnostics_max_five():
    text = "\n\n".join(f"Diag {i}" for i in range(10))
    result = _parse_diagnostics(text)
    assert len(result) == 5


def test_parse_diagnostics_filters_json():
    text = '{"error": "something"}\n\nActual diagnostic here.'
    result = _parse_diagnostics(text)
    assert result == ["Actual diagnostic here."]


def test_parse_diagnostics_filters_errors():
    text = "Error running tool\n\nMost likely: Bad battery."
    result = _parse_diagnostics(text)
    assert result == ["Most likely: Bad battery."]


def test_parse_diagnostics_filters_tool_use_failed():
    text = "tool_use_failed: timeout\n\nPossible: Overheating."
    result = _parse_diagnostics(text)
    assert result == ["Possible: Overheating."]


def test_parse_diagnostics_empty():
    assert _parse_diagnostics("") == []
    assert _parse_diagnostics("   ") == []


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
async def test_diagnose_with_plain_text(diagnostics_service_instance):
    diagnostics_service_instance.agent.arun = AsyncMock(
        return_value=RunOutput(
            content="Most likely: Bad battery.\n\nPossible: Faulty starter."
        )
    )

    result = await diagnostics_service_instance.diagnose(
        message="won't start", listing_id="EQP-1", session_id="s1"
    )

    assert result["diagnostics"] == [
        "Most likely: Bad battery.",
        "Possible: Faulty starter.",
    ]
    assert result["listing_id"] == "EQP-1"
    assert result["session_id"] == "s1"
    assert isinstance(result["execution_time"], float)


@pytest.mark.asyncio
async def test_diagnose_with_no_content(diagnostics_service_instance):
    mock_response = MagicMock()
    mock_response.content = None
    diagnostics_service_instance.agent.arun = AsyncMock(return_value=mock_response)

    result = await diagnostics_service_instance.diagnose(
        message="noise", listing_id="EQP-3", session_id="s3"
    )

    assert result["diagnostics"] == []


@pytest.mark.asyncio
async def test_diagnose_generates_session_id(diagnostics_service_instance):
    diagnostics_service_instance.agent.arun = AsyncMock(
        return_value=RunOutput(content="Most likely: Issue 1.")
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
    diagnostics_service_instance.agent.arun = AsyncMock(
        return_value=RunOutput(content="Most likely: Issue 1.")
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
    diagnostics_service_instance.agent.arun = AsyncMock(
        return_value=RunOutput(content="Most likely: Issue 1.")
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

    with patch("app.core.retry.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(RuntimeError, match="LLM error"):
            await diagnostics_service_instance.diagnose(
                message="broken", listing_id="EQP-1", session_id="s1"
            )
