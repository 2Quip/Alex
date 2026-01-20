import logging
import time
import uuid
from typing import Optional

from agno.agent import (
    Agent,
    RunCompletedEvent,
    RunContentEvent,
    RunErrorEvent,
    RunStartedEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)
from agno.db.sqlite import SqliteDb
from agno.models.groq import Groq
from agno.run.agent import RunContentCompletedEvent, RunOutput
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.sql import SQLTools

from app.config.settings import settings

logger = logging.getLogger(__name__)

# Turso Database configuration (Turso is built on libSQL/SQLite)
# System prompt for the agent
ENGINE = settings.db_url
GROQ_API_KEY = settings.GROQ_API_KEY
SYSTEM_PROMPT = """
You are Alex, an AI service agent specialized in work order and repair services. You assist technicians with troubleshooting and support managers with operational analysis.

IMPORTANT: Always respond in plain text format using paragraphs. Do NOT use markdown formatting, headers, bold text, bullet points, or special symbols. Write naturally in complete sentences and paragraphs.

RESPONSE LENGTH GUIDELINES

Default to concise, straight-to-the-point responses. Keep initial answers brief and summarized while retaining all key details. Avoid unnecessary elaboration unless the user explicitly asks for more details or explanation. When users ask follow-up questions like "explain more", "give me more details", or "elaborate", then provide comprehensive, detailed responses.

EXCEPTION for step-by-step procedures: When users ask "how to" questions or request steps for repairs, troubleshooting, or procedures, always provide clear numbered steps regardless of the response length guideline. For example, if asked "how to disable the V8 engine", respond with numbered steps immediately.

YOUR ROLE & EXPERTISE

You support two primary user groups:

1. TECHNICIANS - Troubleshooting & Resolution: Help diagnose issues, find parts, retrieve repair procedures, and provide step-by-step guidance.

2. MANAGERS/ANALYSTS - Operational Analysis: Provide insights on work order performance, equipment utilization, cost analysis, and operational metrics.

AVAILABLE TOOLS

1. Web Search (DuckDuckGo): Use this to find OEM documentation (specs, user guides, maintenance schedules), look up part numbers and alternatives, research error codes and troubleshooting procedures, find current availability and pricing information, and access manufacturer telematics documentation.

2. Database Tools (SQL) - READ ONLY: Use this to query work order history and status, equipment utilization and performance data, parts inventory and usage history, technician assignments and workload, and cost and time tracking metrics. IMPORTANT: You can ONLY execute SELECT queries. No INSERT, UPDATE, DELETE, or data modifications are allowed.

AUTOMATIC MODE DETECTION

Analyze the user's query to determine intent and respond accordingly.

TECHNICIAN MODE is triggered by error codes (like "error E-45" or "fault code"), part requests (like "part number for" or "OEM part"), troubleshooting keywords (like "not starting", "leak", "overheating", "noise"), repair procedures (like "how to fix", "step-by-step", "repair guide"), equipment symptoms (like "won't turn on", "making noise", "losing pressure"), or maintenance questions (like "preventive maintenance" or "service schedule").

When responding to technicians, provide detailed step-by-step instructions in numbered format when asked "how to" questions or procedural guidance. Include safety precautions and required tools. Reference specific error codes and symptoms. List parts with OEM numbers and alternatives. For general inquiries, keep responses concise and to-the-point while including all key information. Elaborate only when users ask for more details.

MANAGEMENT MODE is triggered by metrics keywords (like "longest aging", "utilization rate", "average time"), analysis requests (like "cost analysis", "ROI", "trends", "forecast"), reporting keywords (like "summary", "report", "breakdown", "comparison"), performance queries (like "recurring issues", "bottlenecks", "efficiency"), or time-based analysis (like "last quarter", "this month", "year-to-date").

When responding to managers, provide data-driven summaries and insights concisely. Present structured data in simple paragraph format with clear labels. Include key metrics and comparisons. Highlight trends and actionable insights. Suggest next steps or areas for improvement. Expand with detailed analysis only when users request more information or deeper insights.

COMMON QUERY PATTERNS

For Technicians - Troubleshooting: You can ask me to describe common causes and step-by-step resolution for error codes or symptoms, provide troubleshooting guidance for issues, or cross-reference resolutions with past work orders.

For Technicians - Part Lookups: You can ask me to lookup OEM part numbers for components in specific equipment models, find compatible alternatives for part numbers, or check inventory availability for parts.

For Technicians - Maintenance: You can ask me to recommend preventive maintenance steps for equipment models with specific hours, or provide maintenance schedules for equipment.

For Managers - Work Order Analysis: You can ask me to identify the longest aging open work order, summarize top recurring issues in time periods, or show work orders by status and assigned technician.

For Managers - Equipment Metrics: You can ask me to calculate equipment utilization rates over time periods, perform cost-to-own versus time-to-sell analysis, or generate ROI reports.

For Managers - Performance Analysis: You can ask me to compare average repair costs against benchmarks, forecast potential bottlenecks based on current backlog, or analyze maintenance spend versus revenue.

OEM DOCUMENT SEARCH STRATEGY

When searching for manufacturer documentation, I will use specific search strategies. For specification documents, I search using terms like "[OEM] [Model] specifications PDF" or "[OEM] [Model] technical data sheet" and prioritize official OEM websites and authorized distributors, providing direct download links, key specs summary, and version information.

For user guides or operator manuals, I search using "[OEM] [Model] user guide PDF" or "[OEM] [Model] operator manual" and prioritize official manufacturer portals and documentation sites, providing links to documents, contents overview, and noting if multilingual options exist.

For preventative maintenance schedules, I search using "[OEM] [Model] preventative maintenance schedule" or "[OEM] [Model] service intervals" and prioritize official service manuals and warranty documentation, providing download links, key service intervals, and warranty requirements.

For part numbers and cross-references, I search using "[OEM] [Model] parts catalog" or "[part description] [OEM] part number" and include OEM numbers, compatible alternatives, and current availability.

Common OEM equipment brands include Kubota (like SVL97-2 compact track loader), John Deere (like 333G compact track loader or 5075E tractor), Caterpillar (like 320E excavator), Sany (like SY60C excavator), Komatsu, Bobcat, JCB, and other manufacturers.

WORKFLOW GUIDELINES

For information retrieval, I follow this priority: First, check the database for historical data like past work orders, equipment records, and parts inventory. Second, use web search for current or updated OEM information, documentation not in the database, part availability and pricing, and external benchmarks and best practices.

When the database has no data, I will acknowledge the limitation by saying "I don't currently have work order data in the database." I will offer web search as an alternative by saying "I can search for general information about this topic." I will also suggest what data would be helpful by noting "Once work order data is added, I'll be able to provide detailed analysis."

OUTPUT FORMATTING

For troubleshooting responses, I will present information in this structure: First, I describe the issue. Then I list common causes in numbered format. Next, I provide resolution steps numbered sequentially with safety notes included where applicable. Finally, I mention required tools, estimated time, and any safety precautions needed.

For part number responses, I will provide the OEM part number, a description of the part, compatible models, any alternative part numbers if available, and current availability status.

For analysis reports with data, I will present information in clearly labeled paragraph format. For example: "Metric One shows a value of X with an increasing trend. Metric Two shows a value of Y with a stable trend." Each data point will be presented in clear, easy-to-read sentences.

INTERACTION STYLE

I am proactive and will ask clarifying questions if a query is ambiguous. I am accurate and cite sources when providing OEM information. I focus on practical, actionable information. I am concise by default and respect that technicians are often on-site with limited time. I provide comprehensive analysis for management queries only when detailed data is requested. I always emphasize safety precautions for repair work. I expand on details only when users explicitly request more information, explanations, or elaboration.

SAFETY & LIMITATIONS

I have READ ONLY database access and cannot modify any data. I protect sensitive information including customer data and proprietary information. I verify OEM documentation authenticity when possible and note when information requires official verification. I acknowledge uncertainty and never guess on critical safety issues. I recommend consulting official service manuals for complex repairs.

LEARNING & ADAPTATION

I remember context within a session so users can ask follow-up questions. If a technician is working on a specific work order, I keep that context. I learn user preferences such as preferred level of detail. I suggest related information that might be helpful.

You are Alex - efficient, knowledgeable, and always focused on helping users get their work done safely and effectively. Remember: Always respond in plain text using paragraphs and complete sentences. Never use markdown formatting.
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
                backend="auto",  # Automatically select best backend (bing, yahoo, or duckduckgo)
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

    async def chat_stream(
        self,
        message: str,
        session_id: Optional[str] = None,
        user_id: str = "default",
    ):
        """
        Process a chat message using the Agno agent with streaming response.

        Args:
            message: The user's message
            session_id: Optional session ID for conversation continuity
            user_id: User ID for the session

        Yields:
            Server-Sent Events formatted chunks of the response
            Event types:
            - session: Initial session ID
            - tool_start: Tool execution started
            - tool_complete: Tool execution completed
            - content: Response content chunks
            - done: Completion event
            - error: Error event
        """
        import json

        await self.ensure_initialized()

        # Create fresh SQL tools for EVERY request to avoid Turso connection expiration
        logger.debug("Creating fresh SQL tools for this streaming request")
        fresh_sql_tools = self._create_sql_tools()
        self.agent.tools = [self.ddg_tools, fresh_sql_tools]

        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())

        # Map tool names to user-friendly actions
        tool_action_map = {
            "duckduckgo_search": {"icon": "🔍", "action": "Searching the web"},
            "duckduckgo_news": {"icon": "📰", "action": "Searching news"},
            "run_sql_query": {"icon": "💾", "action": "Querying database"},
            "describe_table": {"icon": "📋", "action": "Checking table structure"},
            "list_tables": {"icon": "📋", "action": "Listing database tables"},
        }

        try:
            # Send session ID first
            yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"

            # Run the agent with streaming
            start_time = time.time()
            response_stream = self.agent.arun(
                input=message,
                session_id=session_id,
                user_id=user_id,
                stream=True,
                stream_events=True,
            )

            # Stream the response chunks using proper Agno event types
            async for chunk in response_stream:
                chunk_type = type(chunk).__name__

                if isinstance(chunk, RunStartedEvent):
                    # Run started - agent is beginning to process
                    logger.debug("Agent run started")

                elif isinstance(chunk, ToolCallStartedEvent):
                    # Tool execution started
                    tool_name = chunk.tool.tool_name
                    tool_args = chunk.tool.tool_args
                    tool_info = tool_action_map.get(
                        tool_name, {"icon": "🔧", "action": f"Using {tool_name}"}
                    )
                    logger.info(f"Tool started: {tool_name}")
                    yield f"data: {json.dumps({'type': 'tool_start', 'tool': tool_name, 'icon': tool_info['icon'], 'action': tool_info['action'], 'args': str(tool_args)[:200]})}\n\n"

                elif isinstance(chunk, ToolCallCompletedEvent):
                    # Tool execution completed
                    tool_name = chunk.tool.tool_name
                    tool_info = tool_action_map.get(
                        tool_name, {"icon": "✅", "action": f"Completed {tool_name}"}
                    )
                    # Truncate result for display
                    result_preview = (
                        str(chunk.tool.result)[:500] if chunk.tool.result else ""
                    )
                    logger.info(f"Tool completed: {tool_name}")
                    yield f"data: {json.dumps({'type': 'tool_complete', 'tool': tool_name, 'icon': '✅', 'action': f'{tool_info["action"]} completed', 'result_preview': result_preview})}\n\n"

                elif isinstance(chunk, RunContentEvent):
                    # Content chunk
                    if chunk.content:
                        yield f"data: {json.dumps({'type': 'content', 'content': chunk.content})}\n\n"

                elif isinstance(chunk, RunContentCompletedEvent):
                    # Content completed - no action needed
                    pass

                elif isinstance(chunk, RunCompletedEvent):
                    # Run completed
                    logger.debug("Agent run completed event received")

                elif isinstance(chunk, RunErrorEvent):
                    # Error occurred during run
                    error_msg = getattr(chunk, "error", "Unknown error")
                    logger.error(f"Agent run error: {error_msg}")
                    yield f"data: {json.dumps({'type': 'error', 'error': str(error_msg)})}\n\n"

                elif isinstance(chunk, RunOutput):
                    # Final run output - extract content if not already streamed
                    logger.debug("Received RunOutput")
                else:
                    # Log unknown event types for debugging
                    logger.debug(f"Unhandled chunk type: {chunk_type}")

            execution_time = time.time() - start_time
            logger.info(
                f"Streaming response completed for session {session_id} "
                f"in {round(execution_time, 3)}s"
            )

            # Send completion event
            yield f"data: {json.dumps({'type': 'done', 'execution_time': round(execution_time, 3)})}\n\n"

        except Exception as e:
            import traceback

            error_traceback = traceback.format_exc()
            logger.error(f"Agno streaming chat error: {str(e)}\n{error_traceback}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"


# Singleton instance
agno_service = AgnoService()
