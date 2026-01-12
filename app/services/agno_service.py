import logging
import os
import time
import uuid
from typing import Optional

from agno.agent import Agent
from agno.db.mysql import MySQLDb
from agno.models.groq import Groq
from agno.run.agent import RunOutput
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.sql import SQLTools
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Database configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "baas")
DB_PASSWORD = os.getenv("DB_PASSWORD", "baas")
DB_NAME = os.getenv("DB_NAME", "baas")
DB_URL = f"mysql+mysqldb://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# System prompt for the agent
SYSTEM_PROMPT = """
You are a helpful AI assistant with access to web search and database capabilities.

## 🛠️ AVAILABLE TOOLS

### 1. Web Search (DuckDuckGo)
- Use this to search the web for current information, news, or any topic the user asks about.
- Great for finding up-to-date information that may not be in the database.

### 2. Database Tools (MySQL)
- You can query and interact with the MySQL database.
- Use SQL SELECT queries to retrieve information.
- Use SQL INSERT/UPDATE for data modifications (when appropriate).
- Always be careful with data modifications and confirm with the user when needed.

## 💬 INTERACTION STYLE
- Be helpful, concise, and accurate.
- When searching the web, summarize findings clearly.
- When querying the database, present results in a readable format (tables when appropriate).
- If you're unsure about something, say so and ask for clarification.

## 🚨 SAFETY RULES
- Never delete data without explicit user confirmation.
- Be cautious with UPDATE and DELETE operations.
- Protect sensitive information.
"""


# MySQL Database for chat history storage
mysql_db = MySQLDb(
    db_url=DB_URL,
    session_table="agent_sessions",
)


class AgnoService:
    """Service for handling chat with Agno agent with web search and database tools"""

    def __init__(self):
        self.agent: Optional[Agent] = None
        self._initialized = False

    async def initialize(self):
        """Initialize the Agno agent with DuckDuckGo search and SQL tools"""
        if self._initialized:
            return

        try:
            # Initialize tools
            ddg_tools = DuckDuckGoTools()
            sql_tools = SQLTools(db_url=DB_URL)

            # Create the agent with web search and database tools
            self.agent = Agent(
                model=Groq(id="openai/gpt-oss-120b"),
                markdown=True,
                tools=[ddg_tools, sql_tools],
                system_message=SYSTEM_PROMPT,
                db=mysql_db,  # Use MySQL for chat history
                add_history_to_messages=True,
                num_history_responses=5,
            )
            self._initialized = True
            logger.info(
                "Agno agent initialized successfully with DuckDuckGo and MySQL tools"
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
