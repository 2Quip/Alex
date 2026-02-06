# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

2Quip Agent is an AI service agent ("Alex") built with FastAPI and the Agno framework. It assists equipment technicians with troubleshooting/repair and supports managers with operational analysis. The agent has access to DuckDuckGo web search and read-only SQL database queries, with an optional LiveKit-based voice interface.

## Commands

```bash
# Install dependencies
uv sync

# Run API server (development with auto-reload)
uv run uvicorn app.main:app --reload

# Run voice agent (LiveKit)
uv run python -m app.livekit_agent dev

# Run directly (no reload)
uv run python -m app.main

# Docker
docker build -t 2quip-agent .
docker compose up

# Production deploy (uses port 8090, requires .venv)
./deploy.sh start|stop|restart

# Run tests
uv run pytest

# Run a single test file
uv run pytest tests/test_endpoints.py

# Run a single test
uv run pytest tests/test_endpoints.py::test_root -v
```

No linter is currently configured.

## Architecture

### Entry Point & API Layer
- `app/main.py` — FastAPI app with lifespan management (startup/shutdown). Defines endpoints, Pydantic request/response models, and CORS middleware.
- Endpoints: `GET /`, `GET /health`, `POST /chat`, `POST /chat/stream` (SSE), `POST /diagnostics`

### Services (singleton pattern with lazy init)
- `app/services/agno_service.py` — Main chat service. Creates an Agno `Agent` with OpenAI GPT-5-mini, DuckDuckGo tools, and SQLTools. Supports both regular and SSE streaming responses. Contains the 120-line system prompt that defines "Alex" persona and auto-detects TECHNICIAN vs MANAGEMENT mode. Fresh SQLTools are created per request to avoid Turso connection expiration.
- `app/services/diagnostics_service.py` — Equipment diagnostics with structured Pydantic output (max 5 diagnostics). Similar agent setup but with shorter history (3 runs vs 5).
- `app/services/livekit_agno_plugin.py` — `LLMAdapter` wraps the Agno agent as a LiveKit-compatible LLM; `AgnoStream` converts Agno events to LiveKit chat chunks.

### Tools
- `app/tools/send_document.py` — `SendDocumentTool` (Agno `Toolkit`) that POSTs document URLs to an external webhook. Used by all three agents (chat, diagnostics, voice) when `DOCUMENT_WEBHOOK_URL` is configured. Sends JSON payload with `title`, `url`, `recipient`, and `timestamp`. Handles timeout, HTTP errors, and connection failures gracefully.

### Voice Agent
- `app/livekit_agent.py` — LiveKit voice pipeline: Silero VAD → AssemblyAI STT → Agno Agent (via LLMAdapter) → Cartesia TTS. Uses separate SQLite DB (`tmp/livekit_sessions.db`). Session persistence is disabled (causes pickle errors with coroutines).

### Configuration
- `app/config/settings.py` — Pydantic `BaseSettings` loading from `.env`. Singleton at `settings`.
- Required env vars: `DATABASE_URL`, `DATABASE_AUTH_TOKEN`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY`
- Optional env vars (voice): `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `ASSEMBLYAI_API_KEY`, `CARTESIA_API_KEY`, `DEEPGRAM_API_KEY`
- Optional env vars (document sending): `DOCUMENT_WEBHOOK_URL` (when set, enables the `SendDocumentTool` on all agents)
- Optional env vars (logging): `LOG_LEVEL` (default `INFO`), `LOG_FILE` (default `logs/agno_agent_api.log`)
- See `.env.example` and `.env.livekit.example` for templates.

### Logging
- `app/core/logging.py` — Centralized logging config called from both entry points (`app/main.py` and `app/livekit_agent.py`). Provides `setup_logging()` (console + rotating file handler, 10MB/5 backups) and a shared `logger_hook()` used as an Agno tool hook for timing tool calls. All modules use `logging.getLogger(__name__)`.

### Data Flow
1. Request arrives at FastAPI endpoint
2. Service calls `ensure_initialized()`, creates fresh SQL tools, generates session ID if needed
3. Agno agent runs with tools (DuckDuckGo, SQLTools) and session history from local SQLite (`tmp/data.db`)
4. For streaming: events are mapped to SSE types (`session`, `tool_start`, `tool_complete`, `content`, `done`, `error`)

### Key Patterns
- Services are singletons instantiated at module level, initialized in FastAPI lifespan
- DuckDuckGo tools are shared (stateless); SQL tools are recreated per request (connection expiry)
- Model can be swapped between OpenAI, Groq, and OpenRouter by changing the `model=` parameter in agent initialization
- Database is read-only (SELECT queries only, enforced in system prompt and SQLTools config)
- Logging is centralized in `app/core/logging.py`; outputs to both console and `logs/agno_agent_api.log` with rotation

## Dependencies

Managed via `uv` with `pyproject.toml`. Python 3.12 required. Key packages: `agno`, `fastapi[all]`, `ddgs`, `sqlalchemy-libsql`, `groq`, `pydantic-settings`. Voice deps: `livekit`, `livekit-agents`, and various livekit plugins.
