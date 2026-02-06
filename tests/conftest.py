"""Test configuration — mocks external dependencies before any app imports."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

# Set fake env vars BEFORE any app code is imported.
# settings.py runs Settings() at module level which validates these.
os.environ.setdefault("DATABASE_URL", "libsql://test.turso.io")
os.environ.setdefault("DATABASE_AUTH_TOKEN", "test-token")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("OPENROUTER_API_KEY", "test-openrouter-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("LOG_FILE", "logs/test.log")

# Patch create_engine and SqliteDb before any service module imports them at module level.
_mock_engine = MagicMock()
_create_engine_patcher = patch("sqlalchemy.create_engine", return_value=_mock_engine)
_sqlite_db_patcher = patch("agno.db.sqlite.SqliteDb", return_value=MagicMock())
_create_engine_patcher.start()
_sqlite_db_patcher.start()

# Now safe to import app code
import pytest
import pytest_asyncio
import httpx

from agno.agent import Agent
from agno.run.agent import RunOutput
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.sql import SQLTools

from app.services.agno_service import AgnoService
from app.services.diagnostics_service import DiagnosticsService


@pytest.fixture
def mock_agent():
    """Mocked Agno Agent with a canned response."""
    agent = AsyncMock(spec=Agent)
    agent.arun = AsyncMock(return_value=RunOutput(content="Test response"))
    agent.tools = []
    return agent


@pytest.fixture
def mock_ddg_tools():
    return MagicMock(spec=DuckDuckGoTools)


@pytest.fixture
def mock_sql_tools():
    return MagicMock(spec=SQLTools)


@pytest.fixture
def agno_service_instance(mock_agent, mock_ddg_tools, mock_sql_tools):
    """Fresh AgnoService with mocked dependencies — not the global singleton."""
    service = AgnoService()
    service.agent = mock_agent
    service.ddg_tools = mock_ddg_tools
    service._initialized = True
    service._create_sql_tools = MagicMock(return_value=mock_sql_tools)
    return service


@pytest.fixture
def diagnostics_service_instance(mock_agent, mock_ddg_tools, mock_sql_tools):
    """Fresh DiagnosticsService with mocked dependencies."""
    service = DiagnosticsService()
    service.agent = mock_agent
    service.ddg_tools = mock_ddg_tools
    service._initialized = True
    service._create_sql_tools = MagicMock(return_value=mock_sql_tools)
    return service


@pytest_asyncio.fixture
async def client():
    """Async HTTP client with service initialization/cleanup mocked out."""
    from app.services.agno_service import agno_service
    from app.services.diagnostics_service import diagnostics_service
    from app.main import app

    with (
        patch.object(agno_service, "initialize", new_callable=AsyncMock),
        patch.object(agno_service, "cleanup", new_callable=AsyncMock),
        patch.object(diagnostics_service, "initialize", new_callable=AsyncMock),
        patch.object(diagnostics_service, "cleanup", new_callable=AsyncMock),
    ):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as ac:
            yield ac
