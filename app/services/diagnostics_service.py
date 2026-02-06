import logging
import time
import uuid
from typing import List, Optional

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.groq import Groq
from agno.models.openai import OpenAIChat
from agno.models.openrouter import OpenRouter
from agno.run.agent import RunOutput
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.sql import SQLTools
from pydantic import BaseModel, Field

from app.config.settings import settings
from app.core.logging import logger_hook
from app.tools.send_document import SendDocumentTool

logger = logging.getLogger(__name__)

# Database configuration
ENGINE = settings.db_engine
GROQ_API_KEY = settings.GROQ_API_KEY

# System prompt for the diagnostics agent
DIAGNOSTICS_SYSTEM_PROMPT = """
You are Alex, an AI diagnostic specialist for equipment troubleshooting.

Query the `listing` table for the id to get equipment information. If no data found, use web search as fallback.
Analyze the reported issue/symptoms and provide up to 5 potential diagnostics.
Keep diagnostics clear, actionable, and prioritized by likelihood.
Do not add any special markdown formatting, just plain text.

If the user asks you to send or share a document (PDF, repair guide, manual), use the send_document tool with the title and URL instead of just describing the content.
"""

# Turso Database for chat history storage
turso_db = SqliteDb(db_file="tmp/data.db")


class DiagnosticsOutput(BaseModel):
    """Structured output for diagnostics"""
    diagnostics: List[str] = Field(..., max_length=5, description="List of potential diagnoses (max 5)")

class DiagnosticsService:
    """Service for handling equipment diagnostics with structured output"""

    def __init__(self):
        self.agent: Optional[Agent] = None
        self._initialized = False
        self.ddg_tools = None
        self._extra_tools: list = []

    def _create_sql_tools(self):
        """Create a fresh SQLTools instance to avoid connection expiration"""
        logger.debug("Creating fresh SQLTools instance for diagnostics")
        return SQLTools(db_engine=ENGINE)

    async def initialize(self):
        """Initialize the diagnostics agent with DuckDuckGo search and SQL tools"""
        if self._initialized:
            return

        try:
            # Initialize DuckDuckGo tools
            self.ddg_tools = DuckDuckGoTools(
                timeout=20,
                fixed_max_results=10,
            )
            
            # Create fresh SQL tools instance
            sql_tools = self._create_sql_tools()

            # Build tools list
            if settings.DOCUMENT_WEBHOOK_URL:
                self._extra_tools.append(SendDocumentTool(webhook_url=settings.DOCUMENT_WEBHOOK_URL, webhook_secret=settings.DOCUMENT_WEBHOOK_SECRET))
            tools = [self.ddg_tools, sql_tools] + self._extra_tools

            # Create the agent with structured output
            self.agent = Agent(
                # model=Groq(id="openai/gpt-oss-120b", api_key=GROQ_API_KEY),
                # model=OpenRouter(id="google/gemini-2.5-flash", api_key=settings.OPENROUTER_API_KEY),
                model=OpenAIChat(id="gpt-5-mini-2025-08-07", api_key=settings.OPENAI_API_KEY),
                markdown=False,
                tools=tools,
                system_message=DIAGNOSTICS_SYSTEM_PROMPT,
                db=turso_db,
                add_history_to_context=True,
                num_history_runs=3,  # Keep fewer history for focused diagnostics
                output_schema=DiagnosticsOutput,  # Enable structured output
                tool_hooks=[logger_hook],
            )
            
            self._initialized = True
            logger.info("Diagnostics agent initialized successfully with structured output")
            
        except Exception as e:
            logger.error(f"Failed to initialize diagnostics agent: {str(e)}")
            raise

    async def ensure_initialized(self):
        """Ensure the agent is initialized before use"""
        if not self._initialized:
            await self.initialize()

    async def cleanup(self):
        """Cleanup resources"""
        self._initialized = False
        logger.info("Diagnostics service cleaned up")

    async def diagnose(
        self,
        message: str,
        listing_id: str,
        session_id: Optional[str] = None,
        user_id: str = "default",
    ) -> dict:
        """
        Process a diagnostics request using the Agno agent with structured output.

        Args:
            message: Description of the issue/symptoms
            listing_id: The listing identifier to analyze
            session_id: Optional session ID for conversation continuity
            user_id: User ID for the session

        Returns:
            Dict containing structured diagnostics and session_id
        """
        await self.ensure_initialized()

        # Create fresh SQL tools for this request
        logger.debug("Creating fresh SQL tools for diagnostics request")
        fresh_sql_tools = self._create_sql_tools()
        self.agent.tools = [self.ddg_tools, fresh_sql_tools] + self._extra_tools

        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())

        # Construct the diagnostic request message
        diagnostic_message = f"""
Listing ID: {listing_id}
Issue Description: {message}

Please analyze this equipment issue and provide up to 5 potential diagnostics.
Include historical data from the database for this listing_id if available.
Research similar issues and common failure modes using web search if needed.
Return the response as structured JSON with the diagnostics array.
"""

        try:
            start_time = time.time()
            
            # Run the agent with structured output
            response: RunOutput = await self.agent.arun(
                input=diagnostic_message,
                session_id=session_id,
                user_id=user_id,
            )
            
            execution_time = time.time() - start_time

            # The response should be a DiagnosticsOutput object due to response_model
            if hasattr(response, 'content'):
                # If content is already structured (Pydantic model)
                if isinstance(response.content, DiagnosticsOutput):
                    diagnostics_data = response.content.model_dump()
                else:
                    # Parse if it's still a string
                    import json
                    try:
                        diagnostics_data = json.loads(response.content)
                    except:
                        diagnostics_data = {"diagnostics": []}
            else:
                diagnostics_data = {"diagnostics": []}

            logger.info(
                f"Diagnostics generated for listing {listing_id}, session {session_id} "
                f"in {round(execution_time, 3)}s"
            )

            return {
                "diagnostics": diagnostics_data.get("diagnostics", []),
                "listing_id": listing_id,
                "session_id": session_id,
                "execution_time": round(execution_time, 3),
            }

        except Exception as e:
            logger.error(f"Diagnostics error: {str(e)}")
            raise


# Singleton instance
diagnostics_service = DiagnosticsService()
