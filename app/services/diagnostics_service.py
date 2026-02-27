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
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.sql import SQLTools
from pydantic import BaseModel, Field

from app.config.settings import settings
from app.core.logging import logger_hook
from app.core.retry import with_retry
from app.tools.s3_search import S3SearchTool
from app.tools.send_document import SendDocumentTool

logger = logging.getLogger(__name__)


def _parse_diagnostics(text: str) -> list[dict]:
    """Parse structured diagnostics into dicts with likelihood, diagnosis, cause, fix.

    Expected format per line:
        1. [High] Diagnosis title | Cause: brief cause | Fix: brief fix

    Falls back to plain string if a line doesn't match the pattern.
    """
    import re
    items = re.split(r"\n\s*\d+\.\s+", "\n" + text.strip())
    results = []
    for item in items:
        item = item.strip()
        if not item:
            continue

        # Try to parse structured format: [Likelihood] Title | Cause: ... | Fix: ...
        m = re.match(
            r"\[(?P<likelihood>High|Medium|Low)\]\s*(?P<diagnosis>.+?)(?:\s*\|\s*Cause:\s*(?P<cause>.+?))?(?:\s*\|\s*Fix:\s*(?P<fix>.+?))?$",
            item,
            re.IGNORECASE | re.DOTALL,
        )
        if m:
            results.append({
                "likelihood": m.group("likelihood").capitalize(),
                "diagnosis": m.group("diagnosis").strip(),
                "cause": (m.group("cause") or "").strip(),
                "fix": (m.group("fix") or "").strip(),
            })
        else:
            # Fallback: return as plain diagnosis
            results.append({
                "likelihood": "",
                "diagnosis": item,
                "cause": "",
                "fix": "",
            })

        if len(results) >= 5:
            break
    return results

# Database configuration
ENGINE = settings.db_engine
GROQ_API_KEY = settings.GROQ_API_KEY

# System prompt for the diagnostics agent
DIAGNOSTICS_SYSTEM_PROMPT = """
You are Alex, an AI diagnostic specialist. Query the listing table for the id. Provide up to 5 diagnostics prioritized by likelihood. Do not use web search unless the database has zero relevant data. No emojis.

Use this exact format for each diagnostic, one per line:
1. [High] Diagnosis title | Cause: brief cause | Fix: brief fix
2. [Medium] Diagnosis title | Cause: brief cause | Fix: brief fix

Likelihood must be High, Medium, or Low. Keep each line to one sentence per field. No intro, no summary.
"""


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
                timeout=10,
                fixed_max_results=5,
            )
            
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
            tools = [self.ddg_tools, sql_tools] + self._extra_tools

            # Create the agent — Groq for fast inference
            self.agent = Agent(
                model=Groq(id="llama-3.3-70b-versatile", api_key=GROQ_API_KEY),
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
        diagnostic_message = f"Listing ID: {listing_id}. Issue: {message}"

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
