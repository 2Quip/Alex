import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config.settings import settings
from app.core.logging import setup_logging

# Configure logging before importing services (so their module-level code inherits config)
setup_logging(log_level=settings.LOG_LEVEL, log_file=settings.LOG_FILE)

from app.services.agno_service import agno_service  # noqa: E402
from app.services.diagnostics_service import diagnostics_service  # noqa: E402

logger = logging.getLogger(__name__)


# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: str = "default"


class ChatResponse(BaseModel):
    response: str
    session_id: str


class DiagnosticsRequest(BaseModel):
    message: str
    listing_id: str
    session_id: Optional[str] = None
    user_id: str = "default"


class DiagnosticsResponse(BaseModel):
    diagnostics: list[str]
    listing_id: str
    session_id: str
    execution_time: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    logger.info("Starting up Agno Agent API...")
    await agno_service.initialize()
    await diagnostics_service.initialize()
    logger.info("Agno Agent and Diagnostics services initialized successfully")

    yield

    # Shutdown
    logger.info("Shutting down Agno Agent API...")
    await agno_service.cleanup()
    await diagnostics_service.cleanup()
    logger.info("Cleanup complete")


app = FastAPI(
    title="Agno Agent API",
    description="AI Agent with Web Search and MySQL Database capabilities",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Agno Agent API"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with the AI agent.

    The agent has access to:
    - Web search (DuckDuckGo) for current information
    - MySQL database for data queries and operations
    - Conversation history stored in MySQL
    """
    try:
        result = await agno_service.chat(
            message=request.message,
            session_id=request.session_id,
            user_id=request.user_id,
        )
        return ChatResponse(
            response=result["response"],
            session_id=result["session_id"],
        )
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream chat responses from the AI agent.

    The agent has access to:
    - Web search (DuckDuckGo) for current information
    - MySQL database for data queries and operations
    - Conversation history stored in MySQL

    Returns a streaming response with Server-Sent Events (SSE) format.
    """
    try:
        return StreamingResponse(
            agno_service.chat_stream(
                message=request.message,
                session_id=request.session_id,
                user_id=request.user_id,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as e:
        logger.error(f"Chat stream error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/diagnostics", response_model=DiagnosticsResponse)
async def diagnostics(request: DiagnosticsRequest):
    """
    Get equipment diagnostics based on issue description and listing ID.

    The diagnostics agent analyzes the issue and provides up to 5 potential
    diagnoses by:
    - Querying the listing table for equipment information
    - Using web search for similar issues and common failure modes
    - Analyzing historical data if available

    Returns structured diagnostics with session tracking.
    """
    try:
        result = await diagnostics_service.diagnose(
            message=request.message,
            listing_id=request.listing_id,
            session_id=request.session_id,
            user_id=request.user_id,
        )
        return DiagnosticsResponse(
            diagnostics=result["diagnostics"],
            listing_id=result["listing_id"],
            session_id=result["session_id"],
            execution_time=result["execution_time"],
        )
    except Exception as e:
        logger.error(f"Diagnostics error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
