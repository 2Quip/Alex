import logging
import time
import uuid
from typing import Optional

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.groq import Groq
from agno.run.agent import RunOutput
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.sql import SQLTools

from app.config.settings import settings

logger = logging.getLogger(__name__)

# Turso Database configuration (Turso is built on libSQL/SQLite)
# System prompt for the agent
ENGINE = settings.db_url
GROQ_API_KEY = settings.GROQ_API_KEY
SYSTEM_PROMPT = """
You are a helpful AI assistant named Alex with access to web search and database capabilities.

## 🛠️ AVAILABLE TOOLS

### 1. Web Search (DuckDuckGo)
- Use this to search the web for current information, news, or any topic the user asks about.
- Great for finding up-to-date information that may not be in the database.

### 2. Database Tools (Turso/SQLite) - READ ONLY
- You can ONLY READ from the Turso database (built on libSQL/SQLite).
- ONLY use SQL SELECT queries to retrieve information.
- You are NOT permitted to perform INSERT, UPDATE, DELETE, or any data modification operations.
- If the user requests data modifications, politely inform them that you only have read access to the database.

## 💬 INTERACTION STYLE
- Always check the database for information before using web search.
- Be helpful, concise, and accurate.
- When searching the web, summarize findings clearly.
- When querying the database, present results in a readable format (tables when appropriate).
- If you're unsure about something, say so and ask for clarification.

## 🚨 SAFETY RULES
- Database access is READ ONLY - no modifications allowed.
- Protect sensitive information.
"""


# Turso Database for chat history storage (Turso uses SQLite-compatible syntax)
# turso_db = SqliteDb(
#     db_url=DB_URL,
#     session_table="agent_sessions",
# )

turso_db = SqliteDb(db_file="tmp/data.db")


class AgnoService:
    """Service for handling chat with Agno agent with web search and database tools"""

    def __init__(self):
        self.agent: Optional[Agent] = None
        self._initialized = False
        self.ddg_tools = None

    def _create_sql_tools(self):
        """Create a fresh SQLTools instance to avoid connection expiration"""
        logger.debug("Creating fresh SQLTools instance")
        return SQLTools(db_engine=ENGINE)

    async def initialize(self):
        """Initialize the Agno agent with DuckDuckGo search and SQL tools"""
        if self._initialized:
            return

        try:
            # Initialize DuckDuckGo tools once (these don't expire)
            self.ddg_tools = DuckDuckGoTools(
                backend='auto',  # Automatically select best backend (bing, yahoo, or duckduckgo)
                timeout=20,  # Increased timeout for reliability
                fixed_max_results=10,  # Limit results per query to avoid rate limiting
            )
            # Create fresh SQL tools instance for initial setup
            sql_tools = self._create_sql_tools()

            # Create the agent with web search and database tools
            self.agent = Agent(
                model=Groq(id="openai/gpt-oss-120b", api_key=GROQ_API_KEY),
                markdown=False,
                tools=[self.ddg_tools, sql_tools],
                system_message=SYSTEM_PROMPT,
                db=turso_db,  # Use Turso (SQLite-compatible) for chat history
                add_history_to_context=True,
                num_history_runs=5,
            )
            self._initialized = True
            logger.info(
                "Agno agent initialized successfully with DuckDuckGo and Turso database tools"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Agno agent: {str(e)}")
            raise

    async def ensure_initialized(self):
        """Ensure the agent is initialized before use"""
        if not self._initialized:
            await self.initialize()

    async def cleanup(self):
        """Cleanup resources"""
        self._initialized = False
        logger.info("Agno service cleaned up")

    async def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        user_id: str = "default",
    ) -> dict:
        """
        Process a chat message using the Agno agent.

        Args:
            message: The user's message
            session_id: Optional session ID for conversation continuity
            user_id: User ID for the session

        Returns:
            Dict containing response and session_id
        """
        await self.ensure_initialized()
        
        # Create fresh SQL tools for EVERY request to avoid Turso connection expiration
        logger.debug("Creating fresh SQL tools for this request")
        fresh_sql_tools = self._create_sql_tools()
        self.agent.tools = [self.ddg_tools, fresh_sql_tools]

        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())

        try:
            # Run the agent asynchronously
            start_time = time.time()
            response: RunOutput = await self.agent.arun(
                input=message,
                session_id=session_id,
                user_id=user_id,
            )
            execution_time = time.time() - start_time

            # Extract response text
            response_text = response.content if response.content else ""

            logger.info(
                f"Response generated for session {session_id} "
                f"in {round(execution_time, 3)}s"
            )

            return {
                "response": response_text,
                "session_id": session_id,
            }

        except Exception as e:
            logger.error(f"Agno chat error: {str(e)}")
            raise


# Singleton instance
agno_service = AgnoService()
