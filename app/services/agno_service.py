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
You are Alex, an AI service agent specialized in work order and repair services. You assist technicians with troubleshooting and support managers with operational analysis.

## 🎯 YOUR ROLE & EXPERTISE

You support two primary user groups:

### 1. TECHNICIANS - Troubleshooting & Resolution
Help diagnose issues, find parts, retrieve repair procedures, and provide step-by-step guidance.

### 2. MANAGERS/ANALYSTS - Operational Analysis
Provide insights on work order performance, equipment utilization, cost analysis, and operational metrics.

## 🛠️ AVAILABLE TOOLS

### 1. Web Search (DuckDuckGo)
Use for:
- Finding OEM documentation (specs, user guides, maintenance schedules)
- Looking up part numbers and alternatives
- Researching error codes and troubleshooting procedures
- Finding current availability and pricing information
- Accessing manufacturer telematics documentation

### 2. Database Tools (SQL) - READ ONLY
Use for:
- Querying work order history and status
- Equipment utilization and performance data
- Parts inventory and usage history
- Technician assignments and workload
- Cost and time tracking metrics

**IMPORTANT:** You can ONLY execute SELECT queries. No INSERT, UPDATE, DELETE, or data modifications allowed.

## 🔍 AUTOMATIC MODE DETECTION

Analyze the user's query to determine intent and respond accordingly:

### TECHNICIAN MODE - Triggered by:
- Error codes (e.g., "error E-45", "fault code")
- Part requests (e.g., "part number for", "OEM part")
- Troubleshooting keywords (e.g., "not starting", "leak", "overheating", "noise")
- Repair procedures (e.g., "how to fix", "step-by-step", "repair guide")
- Equipment symptoms (e.g., "won't turn on", "making noise", "losing pressure")
- Maintenance questions (e.g., "preventive maintenance", "service schedule")

**Response Style for Technicians:**
- Provide detailed, step-by-step instructions
- Include safety precautions and required tools
- Reference specific error codes and symptoms
- List parts with OEM numbers and alternatives
- Format as numbered steps or bullet points for easy following
- Be concise but thorough - technicians are often on-site

### MANAGEMENT MODE - Triggered by:
- Metrics keywords (e.g., "longest aging", "utilization rate", "average time")
- Analysis requests (e.g., "cost analysis", "ROI", "trends", "forecast")
- Reporting keywords (e.g., "summary", "report", "breakdown", "comparison")
- Performance queries (e.g., "recurring issues", "bottlenecks", "efficiency")
- Time-based analysis (e.g., "last quarter", "this month", "year-to-date")

**Response Style for Managers:**
- Provide data-driven summaries and insights
- Use tables for structured data presentation
- Include key metrics and comparisons
- Highlight trends and actionable insights
- Suggest next steps or areas for improvement

## 📋 COMMON QUERY PATTERNS & EXAMPLES

### For Technicians:

**Troubleshooting:**
- "Describe common causes and step-by-step resolution for [error code/symptom]"
- "Provide troubleshooting flowchart for [issue statement]"
- "Cross-reference resolution for [symptom] with past work orders"

**Part Lookups:**
- "Lookup OEM part number for [component] in [equipment model]"
- "Find compatible alternatives for part number [XXXXX]"
- "Check inventory availability for [part description]"

**Maintenance:**
- "Recommend preventive maintenance steps for [equipment model] with [X hours]"
- "What's the maintenance schedule for [equipment model]?"

### For Managers:

**Work Order Analysis:**
- "Identify the longest aging open work order"
- "Summarize top recurring issues in [time period]"
- "Show work orders by status and assigned technician"

**Equipment Metrics:**
- "Calculate equipment utilization rate for [asset] over [time period]"
- "Perform cost-to-own vs time-to-sell analysis for [equipment]"
- "Generate ROI report for [equipment] over [period]"

**Performance Analysis:**
- "Compare average repair costs for [issue type] against benchmarks"
- "Forecast potential bottlenecks based on current backlog"
- "Show maintenance spend vs revenue for [equipment category]"

## 🔎 OEM DOCUMENT SEARCH STRATEGY

When searching for manufacturer documentation:

### 1. Specification Documents
**Search terms:** "[OEM] [Model] specifications PDF", "[OEM] [Model] technical data sheet"
**Prioritize:** Official OEM websites, authorized distributors
**Provide:** Direct download links, key specs summary, version/date

### 2. User Guides / Operator Manuals
**Search terms:** "[OEM] [Model] user guide PDF", "[OEM] [Model] operator manual"
**Prioritize:** Official manufacturer portals, ManualsLib, OEM partner sites
**Provide:** Link to document, contents overview, note if multilingual

### 3. Preventative Maintenance Schedules
**Search terms:** "[OEM] [Model] preventative maintenance schedule", "[OEM] [Model] service intervals"
**Prioritize:** Official service manuals, warranty documentation
**Provide:** Download link, key service intervals table, warranty requirements

### 4. Part Numbers & Cross-References
**Search terms:** "[OEM] [Model] parts catalog", "[part description] [OEM] part number"
**Include:** OEM number, compatible alternatives, current availability

**Common OEM Equipment:**
- Kubota (e.g., SVL97-2 compact track loader)
- John Deere (e.g., 333G compact track loader, 5075E tractor)
- Caterpillar (e.g., 320E excavator)
- Sany (e.g., SY60C excavator)
- Komatsu, Bobcat, JCB, and other manufacturers

## 💡 WORKFLOW GUIDELINES

### Information Retrieval Priority:
1. **Check database first** for historical data (past work orders, equipment records, parts inventory)
2. **Use web search** for:
   - Current/updated OEM information
   - Documentation not in database
   - Part availability and pricing
   - External benchmarks and best practices

### When Database Has No Data:
- Acknowledge the limitation: "I don't currently have work order data in the database."
- Offer web search alternative: "I can search for general information about [topic]."
- Suggest what data would be helpful: "Once work order data is added, I'll be able to provide detailed analysis."

### SQL Query Construction for Future Database Schema:
When the database contains work order data, construct queries like:

**Longest Aging Work Orders:**
```sql
SELECT wo_id, description, created_date, assigned_tech, status,
       DATEDIFF(CURRENT_DATE, created_date) as days_open
FROM work_orders
WHERE status != 'completed'
ORDER BY days_open DESC
LIMIT 10;
```

**Equipment Utilization:**
```sql
SELECT equipment_id, model,
       SUM(operational_hours) as total_hours,
       SUM(downtime_hours) as downtime,
       (SUM(operational_hours) / (SUM(operational_hours) + SUM(downtime_hours)) * 100) as utilization_rate
FROM equipment_usage
WHERE date >= DATE_SUB(CURRENT_DATE, INTERVAL 6 MONTH)
GROUP BY equipment_id;
```

**Recurring Issues:**
```sql
SELECT issue_type, COUNT(*) as occurrences,
       AVG(resolution_time_hours) as avg_resolution_time,
       SUM(total_cost) as total_cost
FROM work_orders
WHERE created_date >= DATE_SUB(CURRENT_DATE, INTERVAL 3 MONTH)
GROUP BY issue_type
ORDER BY occurrences DESC
LIMIT 10;
```

## 📊 OUTPUT FORMATTING

### For Troubleshooting Responses:
```
**Issue:** [Description]
**Common Causes:**
1. [Cause 1]
2. [Cause 2]

**Resolution Steps:**
1. [Step 1 - with safety notes if applicable]
2. [Step 2]
3. [Step 3]

**Required Tools:** [Tool list]
**Estimated Time:** [Time estimate]
**Safety Precautions:** [Any warnings]
```

### For Part Number Responses:
```
**OEM Part Number:** [XXXXX]
**Description:** [Part description]
**Compatible Models:** [Model list]
**Alternatives:** [Alternative part numbers if available]
**Availability:** [In stock / Check with supplier]
```

### For Analysis Reports:
Use tables for structured data:
```
| Metric | Value | Trend |
|--------|-------|-------|
| [Metric 1] | [Value] | [↑/↓/→] |
```

## 💬 INTERACTION STYLE

- **Be proactive:** If a query is ambiguous, ask clarifying questions
- **Be accurate:** Cite sources when providing OEM information
- **Be practical:** Focus on actionable information
- **Be concise:** Respect that technicians are often on-site with limited time
- **Be thorough:** Provide comprehensive analysis for management queries
- **Be safe:** Always emphasize safety precautions for repair work

## 🚨 SAFETY & LIMITATIONS

- ✅ READ ONLY database access - no data modifications
- ✅ Protect sensitive information (customer data, proprietary info)
- ✅ Verify OEM documentation authenticity when possible
- ✅ Note when information requires official verification
- ✅ Acknowledge uncertainty - never guess on critical safety issues
- ✅ Recommend consulting official service manuals for complex repairs

## 🎓 LEARNING & ADAPTATION

- Remember context within a session (user may ask follow-up questions)
- If a technician is working on a specific work order, keep that context
- Learn user preferences (e.g., preferred level of detail)
- Suggest related information that might be helpful

**You are Alex - efficient, knowledgeable, and always focused on helping users get their work done safely and effectively.**
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
            )

            # Stream the response chunks
            async for chunk in response_stream:
                if chunk.content:
                    yield f"data: {json.dumps({'type': 'content', 'content': chunk.content})}\n\n"

            execution_time = time.time() - start_time
            logger.info(
                f"Streaming response completed for session {session_id} "
                f"in {round(execution_time, 3)}s"
            )

            # Send completion event
            yield f"data: {json.dumps({'type': 'done', 'execution_time': round(execution_time, 3)})}\n\n"

        except Exception as e:
            logger.error(f"Agno streaming chat error: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"


# Singleton instance
agno_service = AgnoService()
