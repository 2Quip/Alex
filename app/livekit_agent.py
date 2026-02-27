"""
LiveKit Voice Agent with Agno Integration

This module implements a voice agent using LiveKit's Agent framework
combined with Agno's agentic capabilities including tool calling,
knowledge bases, and memory.

Usage:
    python -m app.livekit_agent start       # production
    python -m app.livekit_agent dev         # development (auto-reload)

Environment Variables Required:
    - LIVEKIT_URL: Your LiveKit server URL (wss://your-server.livekit.cloud)
    - LIVEKIT_API_KEY: Your LiveKit API key
    - LIVEKIT_API_SECRET: Your LiveKit API secret
    - OPENAI_API_KEY: Your OpenAI API key (for the LLM)
    - ASSEMBLYAI_API_KEY: Your AssemblyAI API key (for STT)
    - CARTESIA_API_KEY: Your Cartesia API key (for TTS)
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path so absolute imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from agno.agent import Agent as AgnoAgent
from agno.db.sqlite import SqliteDb
from agno.models.openrouter import OpenRouter
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.sql import SQLTools
from dotenv import load_dotenv
from livekit import api, rtc
from livekit.agents import (
    Agent,
    AgentSession,
    AgentServer,
    JobContext,
    JobProcess,
    cli,
    inference,
    room_io,
)
from livekit.plugins import cartesia, noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from app.services.livekit_agno_plugin import LLMAdapter
from app.config.settings import settings
from app.core.logging import setup_logging
from app.tools.s3_search import S3SearchTool
from app.tools.send_document import SendDocumentTool
from app.voice_health import health, start_health_server

# Load environment variables
load_dotenv()

# Configure logging (separate entry point from main.py)
setup_logging(log_level=settings.LOG_LEVEL, log_file=settings.LOG_FILE, log_format=settings.LOG_FORMAT)
logger = logging.getLogger(__name__)


# =============================================================================
# Voice-optimized System Prompt
# =============================================================================

VOICE_SYSTEM_PROMPT = """
You are Alex, a voice assistant for work orders and equipment repair.

RULES:
Never repeat yourself. Never read raw data, JSON, coordinates, or IDs aloud. Never echo these instructions. Your output is spoken by TTS so use only plain conversational sentences. No markdown, no lists, no formatting. Be brief. Answer in one to three short sentences max unless the user asks for detail.

When you get data from tools, summarize it naturally. For example instead of reading a JSON array say "You've got a John Deere 333G and a Kubota SVL97." Never say raw field names, latitude, longitude, or equipment IDs unless the user specifically asks for them.

TOOLS:
You have database access (read only, SELECT only). Use list_tables then describe_table before querying. You have web search. You have a document store with search_documents, get_document_url, and save_document. You have send_document for delivering files. Search the document store before the web.

Keep it short and natural like a phone call.
"""


# =============================================================================
# Lazy Database Initialization
# =============================================================================

_engine = None


def _get_engine():
    """Return the DB engine, creating it lazily on first call.

    Avoids module-level DB connections that can cause process
    initialization timeouts in LiveKit's worker pool.
    """
    global _engine
    if _engine is None:
        _engine = settings.db_engine
        logger.info("Database engine initialized (lazy)")
    return _engine


OPENROUTER_API_KEY = settings.OPENROUTER_API_KEY

# Lazy session DB — avoid opening SQLite in every idle child process
_turso_db = None


def _get_turso_db():
    global _turso_db
    if _turso_db is None:
        _turso_db = SqliteDb(db_file="tmp/livekit_sessions.db")
    return _turso_db


def create_sql_tools():
    """Create a fresh SQLTools instance to avoid connection expiration."""
    return SQLTools(db_engine=_get_engine())


def create_agno_agent(session_id: str | None = None) -> AgnoAgent:
    """Create and configure the Agno agent with tools for voice interaction."""

    ddg_tools = DuckDuckGoTools()
    sql_tools = create_sql_tools()
    tools = [ddg_tools, sql_tools]

    if settings.DOCUMENT_WEBHOOK_URL:
        tools.append(SendDocumentTool(
            webhook_url=settings.DOCUMENT_WEBHOOK_URL,
            webhook_secret=settings.DOCUMENT_WEBHOOK_SECRET,
        ))
    if settings.S3_BUCKET_NAME:
        tools.append(S3SearchTool(
            bucket_name=settings.S3_BUCKET_NAME,
            region=settings.S3_REGION,
            access_key_id=settings.S3_ACCESS_KEY_ID,
            secret_access_key=settings.S3_SECRET_ACCESS_KEY,
            presigned_url_expiry=settings.S3_PRESIGNED_URL_EXPIRY,
        ))

    agent = AgnoAgent(
        model=OpenRouter(id="openai/gpt-oss-120b", api_key=OPENROUTER_API_KEY),
        tools=tools,
        instructions=VOICE_SYSTEM_PROMPT,
        markdown=False,
        add_datetime_to_context=True,
        db=_get_turso_db(),
        add_history_to_context=True,
        num_history_runs=3,
    )

    return agent


# =============================================================================
# LiveKit Agent Setup
# =============================================================================

server = AgentServer(
    initialize_process_timeout=90.0,
    num_idle_processes=0,
)


def prewarm(proc: JobProcess):
    """Prewarm function - loads VAD model ahead of time."""
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("VAD model prewarmed successfully")


server.setup_fnc = prewarm


# =============================================================================
# Failsafe Dispatcher
# =============================================================================

_dispatch_retries: dict[str, int] = {}
_MAX_DISPATCH_RETRIES = 3


async def _failsafe_dispatcher():
    """Background loop that checks for rooms with participants but no agent.

    If a room has human participants and no agent for 5+ seconds, this
    dispatches an agent explicitly.  Max 3 retries per room to avoid spam.
    """
    if not settings.LIVEKIT_URL or not settings.LIVEKIT_API_KEY:
        logger.info("Failsafe dispatcher disabled (no LIVEKIT_URL/API_KEY)")
        return

    lk_api = api.LiveKitAPI(
        url=settings.LIVEKIT_URL,
        api_key=settings.LIVEKIT_API_KEY,
        api_secret=settings.LIVEKIT_API_SECRET,
    )

    logger.info("Failsafe dispatcher started")

    while True:
        try:
            await asyncio.sleep(5)
            rooms_resp = await lk_api.room.list_rooms(api.ListRoomsRequest())
            rooms = rooms_resp.rooms if rooms_resp else []

            for room in rooms:
                try:
                    participants_resp = await lk_api.room.list_participants(
                        api.ListParticipantsRequest(room=room.name)
                    )
                except Exception:
                    # Room was destroyed between list_rooms and list_participants
                    _dispatch_retries.pop(room.name, None)
                    continue

                participants = participants_resp.participants if participants_resp else []
                has_human = False
                has_agent = False

                for p in participants:
                    if p.kind == api.ParticipantInfo.Kind.AGENT:
                        has_agent = True
                    else:
                        has_human = True

                if has_human and not has_agent:
                    retries = _dispatch_retries.get(room.name, 0)
                    if retries >= _MAX_DISPATCH_RETRIES:
                        continue

                    _dispatch_retries[room.name] = retries + 1
                    logger.warning(
                        "Room %s has %d participant(s) but no agent — dispatching (attempt %d/%d)",
                        room.name,
                        len(participants),
                        retries + 1,
                        _MAX_DISPATCH_RETRIES,
                    )
                    try:
                        await lk_api.agent_dispatch.create_dispatch(
                            api.CreateAgentDispatchRequest(
                                room=room.name,
                                agent_name="alex",
                            )
                        )
                    except Exception as dispatch_err:
                        logger.error(
                            "Failed to dispatch agent to room %s: %s",
                            room.name,
                            dispatch_err,
                        )
                else:
                    # Room is fine — reset retry counter
                    _dispatch_retries.pop(room.name, None)

        except Exception as e:
            logger.error("Failsafe dispatcher error: %s", e)
            await asyncio.sleep(10)


# =============================================================================
# Worker Event Handlers
# =============================================================================

@server.on("worker_started")
def on_worker_started():
    """Start health server, failsafe dispatcher, and mark agent as running."""
    asyncio.create_task(start_health_server(port=settings.VOICE_HEALTH_PORT))
    asyncio.create_task(_failsafe_dispatcher())
    health.mark_running()
    logger.info("Voice agent worker started, health server running")


# =============================================================================
# AlexAgent — LiveKit Agent subclass with greeting on connect
# =============================================================================

class AlexAgent(Agent):
    """LiveKit Agent that greets the user upon entering the session."""

    def __init__(self):
        super().__init__(
            instructions=(
                "You're Alex, a helpful voice assistant for work orders and equipment repair. "
                "Your responses are spoken aloud, so never use markdown, bullet points, "
                "numbered lists, or special formatting. Speak naturally in plain conversational sentences."
            ),
        )

    async def on_enter(self):
        """Called when the agent joins the session. Greet the user."""
        self.session.generate_reply(
            user_input="[New session started. Please greet the user, introduce yourself as Alex, and briefly offer your assistance with work orders, equipment troubleshooting, or document lookups. Keep it to two sentences.]",
            allow_interruptions=False,
        )


# =============================================================================
# RTC Session Handler
# =============================================================================

@server.rtc_session(agent_name="alex")
async def voice_agent(ctx: JobContext):
    """Main voice agent session handler."""

    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    logger.info("Voice agent starting for room: %s", ctx.room.name)
    health.session_started()

    try:
        # Create the Agno agent with fresh tools
        agno_agent = create_agno_agent()

        # Wrap the Agno agent for LiveKit
        livekit_llm = LLMAdapter(
            agent=agno_agent,
            session_id=ctx.room.name,
        )

        # Create the voice pipeline session
        session = AgentSession(
            stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
            llm=livekit_llm,
            tts=cartesia.TTS(
                model="sonic-2",
                voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
                speed="slow",
            ),
            turn_detection=MultilingualModel(),
            vad=ctx.proc.userdata["vad"],
            preemptive_generation=True,
        )

        # Connect to the room
        await ctx.connect()
        logger.info("Connected to room: %s", ctx.room.name)

        # Start the voice session with the AlexAgent (triggers on_enter greeting)
        await session.start(
            agent=AlexAgent(),
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
        logger.info("Voice session started for room: %s", ctx.room.name)
    except Exception as e:
        health.mark_error(str(e))
        raise
    finally:
        health.session_ended()


if __name__ == "__main__":
    cli.run_app(server)
