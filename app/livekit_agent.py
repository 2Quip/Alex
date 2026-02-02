"""
LiveKit Voice Agent with Agno Integration

This module implements a voice agent using LiveKit's VoicePipelineAgent
combined with Agno's powerful agentic capabilities including tool calling,
knowledge bases, and memory.

Usage:
    python -m app.livekit_agent dev

Environment Variables Required:
    - LIVEKIT_URL: Your LiveKit server URL (wss://your-server.livekit.cloud)
    - LIVEKIT_API_KEY: Your LiveKit API key
    - LIVEKIT_API_SECRET: Your LiveKit API secret
    - OPENAI_API_KEY: Your OpenAI API key (for the LLM)
    - ASSEMBLYAI_API_KEY: Your AssemblyAI API key (for STT) - or use Deepgram
    - CARTESIA_API_KEY: Your Cartesia API key (for TTS) - optional
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path so absolute imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from agno.agent import Agent as AgnoAgent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIChat
from agno.models.openrouter import OpenRouter
from agno.db.sqlite import SqliteDb
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.sql import SQLTools
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    inference,
    room_io,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from app.services.livekit_agno_plugin import LLMAdapter
from app.config.settings import settings

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Voice-optimized System Prompt
# =============================================================================

VOICE_SYSTEM_PROMPT = """
You are Alex, a helpful voice AI assistant specialized in work order and repair services. You assist technicians with troubleshooting and support managers with operational analysis.

IMPORTANT VOICE INTERACTION GUIDELINES:
- Keep responses CONCISE and conversational - you're speaking, not writing
- Use natural speech patterns and contractions (I'm, you'll, let's, etc.)
- Avoid long lists, complex formatting, or technical jargon
- When using tools, briefly explain what you're doing
- If you don't know something, say so honestly
- Be friendly, helpful, and direct

YOUR CAPABILITIES:
1. Web Search: You can search the web for OEM documentation, part numbers, troubleshooting guides, and current information.
2. Database Queries: You can query work order history, equipment data, parts inventory, and operational metrics (READ ONLY).

RESPONSE STYLE:
- For simple questions: Give direct, brief answers
- For troubleshooting: Explain the issue, then provide 2-3 key steps
- For data queries: Summarize the key findings conversationally
- Always acknowledge when you're searching or querying

Remember: Your responses will be spoken aloud, so be natural and conversational!
"""


# =============================================================================
# Database and Tool Setup
# =============================================================================

ENGINE = settings.db_engine
OPENROUTER_API_KEY = settings.OPENROUTER_API_KEY
# SQLite DB for session persistence - disabled due to pickle errors with LiveKit
turso_db = SqliteDb(db_file="tmp/livekit_sessions.db")


def create_sql_tools():
    """Create a fresh SQLTools instance to avoid connection expiration."""
    return SQLTools(db_engine=ENGINE)


def create_agno_agent(session_id: str | None = None) -> AgnoAgent:
    """Create and configure the Agno agent with tools for voice interaction."""
    
    # Initialize tools
    ddg_tools = DuckDuckGoTools()
    sql_tools = create_sql_tools()
    
    # Create agent without db persistence to avoid pickle errors
    # LiveKit's AgentSession maintains the conversation context instead
    agent = AgnoAgent(
        model=OpenRouter(id="openai/gpt-oss-120b", api_key=OPENROUTER_API_KEY),
        tools=[ddg_tools, sql_tools],
        instructions=VOICE_SYSTEM_PROMPT,
        markdown=False,  # No markdown for voice
        add_datetime_to_context=True,
        # Session persistence disabled - causes pickle errors with coroutines
        db=turso_db,
        add_history_to_context=True,
        num_history_runs=3,
    )
    
    return agent


# =============================================================================
# LiveKit Agent Setup
# =============================================================================

server = AgentServer()


def prewarm(proc: JobProcess):
    """Prewarm function - loads VAD model ahead of time."""
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("VAD model prewarmed successfully")


server.setup_fnc = prewarm


@server.rtc_session()
async def voice_agent(ctx: JobContext):
    """Main voice agent session handler."""
    
    # Add logging context
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    
    logger.info(f"Voice agent starting for room: {ctx.room.name}")
    
    # Get the room SID (it's a coroutine property, so we need to await it)
    
    # Create the Agno agent with room-specific session
    agno_agent = create_agno_agent()
    
    # Wrap the Agno agent for LiveKit
    livekit_llm = LLMAdapter(
        agent=agno_agent,
        session_id=ctx.room.name,
    )
    
    # Create the voice pipeline session
    session = AgentSession(
        # Speech-to-text (STT) - your agent's ears
        stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
        
        # LLM - your agent's brain (Agno wrapped for LiveKit)
        llm=livekit_llm,
        
        # Text-to-speech (TTS) - your agent's voice
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"  # A natural-sounding voice
        ),
        
        # Turn detection for natural conversation flow
        turn_detection=MultilingualModel(),
        
        # Voice Activity Detection (prewarmed)
        vad=ctx.proc.userdata["vad"],
        
        # Allow LLM to start generating while user is finishing speaking
        preemptive_generation=True,
    )
    
    # Start the voice session
    await session.start(
        agent=Agent(instructions="You're Alex, a helpful voice assistant."),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )
    
    logger.info(f"Voice session started for room: {ctx.room.name}")
    
    # Connect to the room
    await ctx.connect()
    
    logger.info(f"Connected to room: {ctx.room.name}")


if __name__ == "__main__":
    cli.run_app(server)
