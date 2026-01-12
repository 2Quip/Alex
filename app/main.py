import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.services.agno_service import agno_service

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: str = "default"


class ChatResponse(BaseModel):
    response: str
    session_id: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    logger.info("Starting up Agno Agent API...")
    await agno_service.initialize()
    logger.info("Agno Agent initialized successfully")

    yield

    # Shutdown
    logger.info("Shutting down Agno Agent API...")
    await agno_service.cleanup()
    logger.info("Cleanup complete")


app = FastAPI(
    title="Agno Agent API",
    description="AI Agent with Web Search and MySQL Database capabilities",
    version="1.0.0",
    lifespan=lifespan,
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
