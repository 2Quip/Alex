import logging
import time
import uuid
from typing import List, Optional

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.groq import Groq
from app.models.openai_patch import PatchedOpenAIChat
from agno.models.openrouter import OpenRouter
from agno.run.agent import RunOutput
from app.tools.search import create_search_tools
from app.tools.sql_tool import create_sql_tools
from pydantic import BaseModel, Field

from app.config.settings import settings
from app.core.logging import logger_hook
from app.core.retry import with_retry
from app.tools.s3_search import S3SearchTool
from app.tools.send_document import SendDocumentTool

logger = logging.getLogger(__name__)


def _parse_diagnostics(text: str) -> list[str]:
    """Split diagnostics paragraphs into a list of strings."""
    import re
    # Split on blank lines or numbered prefixes
    parts = re.split(r"\n\s*\n|\n\s*\d+\.\s+", "\n" + text.strip())
    results = []
    for p in parts:
        p = p.strip()
        # Skip empty, error messages, or raw JSON/function fragments
        if not p or p.startswith("{") or "tool_use_failed" in p or "error" in p[:20].lower():
            continue
        results.append(p)
    return results[:2]

# Database configuration
ENGINE = settings.db_engine
GROQ_API_KEY = settings.GROQ_API_KEY

# System prompt for the diagnostics agent
DIAGNOSTICS_SYSTEM_PROMPT = """
You are Alex, an AI diagnostic specialist. Query the database for the listing ID provided. Provide exactly 2 diagnostics: the most likely cause first, then one alternative. Do not use web search. No emojis.

Each diagnostic must start with the likelihood ("Most likely" or "Also possible"), then the diagnosis, cause, how to check, and fix. Keep each diagnostic to 3-4 sentences maximum. Be concise. Separate each diagnostic with a blank line.

DATABASE HINTS: The equipment table is called "listing" (not "listings"). The listing ID column is "id". Always use list_tables and describe_table before querying to confirm table and column names.

Never reveal database internals to users. Never mention table names, column names, schema, or SQL queries in your response. If asked about the database structure, just say "I'm not able to help with that."
"""


class DiagnosticsService:
    """Service for handling equipment diagnostics with structured output"""

    def __init__(self):
        self.agent: Optional[Agent] = None
        self._initialized = False
        self.search_tools = None
        self._extra_tools: list = []

    def _create_sql_tools(self):
        """Create a fresh read-only SQLTools instance to avoid connection expiration"""
        logger.debug("Creating fresh SQLTools instance for diagnostics")
        return create_sql_tools(db_engine=ENGINE)

    async def initialize(self):
        """Initialize the diagnostics agent with web search and SQL tools"""
        if self._initialized:
            return

        try:
            # Initialize search tools
            self.search_tools = create_search_tools()
            
            # Create fresh SQL tools instance
            sql_tools = self._create_sql_tools()

            # Build tools list
            if settings.DOCUMENT_WEBHOOK_URL:
                self._extra_tools.append(SendDocumentTool(webhook_url=settings.DOCUMENT_WEBHOOK_URL, webhook_secret=settings.DOCUMENT_WEBHOOK_SECRET))
            if settings.S3_BUCKET_NAME:
                self._extra_tools.append(S3SearchTool(
                    bucket_name=settings.S3_BUCKET_NAME,
                    region=settings.S3_REGION,
                    access_key_id=settings.S3_ACCESS_KEY_ID,
                    secret_access_key=settings.S3_SECRET_ACCESS_KEY,
                    presigned_url_expiry=settings.S3_PRESIGNED_URL_EXPIRY,
                ))
            tools = [self.search_tools, sql_tools] + self._extra_tools

            # Create the agent — Gemini 2.5 Flash via OpenRouter for speed
            self.agent = Agent(
                model=OpenRouter(id="google/gemini-2.5-flash", api_key=settings.OPENROUTER_API_KEY),
                markdown=False,
                tools=tools,
                system_message=DIAGNOSTICS_SYSTEM_PROMPT,
                add_history_to_context=False,
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
        metadata: Optional[str] = None,
    ) -> dict:
        """
        Process a diagnostics request using the Agno agent with structured output.

        Args:
            message: Description of the issue/symptoms
            listing_id: The listing identifier to analyze
            session_id: Optional session ID for conversation continuity
            user_id: User ID for the session
            metadata: Optional JSON string with page context

        Returns:
            Dict containing structured diagnostics and session_id
        """
        await self.ensure_initialized()

        # Create fresh SQL tools for this request
        logger.debug("Creating fresh SQL tools for diagnostics request")
        fresh_sql_tools = self._create_sql_tools()
        self.agent.tools = [self.search_tools, fresh_sql_tools] + self._extra_tools

        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())

        # Construct the diagnostic request message with optional context
        context = ""
        if metadata:
            import json
            try:
                ctx = json.loads(metadata)
                if ctx.get("work_order_id"):
                    context = f" Work order ID: {ctx['work_order_id']}."
                if ctx.get("equipment_name"):
                    context += f" Equipment: {ctx['equipment_name']}."
            except (ValueError, TypeError):
                pass
        diagnostic_message = f"Listing ID: {listing_id}.{context} Issue: {message}"

        try:
            start_time = time.time()
            
            # Run the agent with structured output and retry for transient failures
            response: RunOutput = await with_retry(
                self.agent.arun,
                input=diagnostic_message,
                session_id=session_id,
                user_id=user_id,
            )
            
            execution_time = time.time() - start_time

            # Parse numbered list from plain text response
            raw = str(response.content) if response.content else ""
            diagnostics_list = _parse_diagnostics(raw)

            logger.info(
                f"Diagnostics generated for listing {listing_id}, session {session_id} "
                f"in {round(execution_time, 3)}s"
            )

            return {
                "diagnostics": diagnostics_list,
                "listing_id": listing_id,
                "session_id": session_id,
                "execution_time": round(execution_time, 3),
            }

        except Exception as e:
            logger.error(f"Diagnostics error: {str(e)}")
            raise


# Singleton instance
diagnostics_service = DiagnosticsService()
