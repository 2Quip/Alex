# 2Quip Voice Agent - LiveKit + Agno Integration

This module enables real-time voice calling capabilities for the 2Quip AI Agent using LiveKit's voice infrastructure combined with Agno's powerful agentic capabilities.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    LiveKit Room                          │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐              │
│  │  User   │───▶│   VAD   │───▶│   STT   │              │
│  │  Audio  │    │(Silero) │    │(AssemblyAI)            │
│  └─────────┘    └─────────┘    └────┬────┘              │
│                                      │                   │
│                                      ▼                   │
│  ┌─────────────────────────────────────────────────┐    │
│  │             LLMAdapter (Agno Plugin)             │    │
│  │  ┌─────────────────────────────────────────┐    │    │
│  │  │              Agno Agent (Alex)           │    │    │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐   │    │    │
│  │  │  │Web Search│ │Database │ │ Memory  │   │    │    │
│  │  │  │(DuckDuckGo)│ │(SQL)  │ │(SQLite) │   │    │    │
│  │  │  └─────────┘ └─────────┘ └─────────┘   │    │    │
│  │  └─────────────────────────────────────────┘    │    │
│  └──────────────────────┬──────────────────────────┘    │
│                         │                                │
│                         ▼                                │
│  ┌─────────┐    ┌─────────┐                             │
│  │   TTS   │◀───│AgnoStream│                             │
│  │(Cartesia)│    └─────────┘                             │
│  └────┬────┘                                             │
│       │                                                  │
│       ▼                                                  │
│  ┌─────────┐                                             │
│  │  Audio  │                                             │
│  │ Output  │                                             │
│  └─────────┘                                             │
└─────────────────────────────────────────────────────────┘
```

## Features

- **Real-time Voice Interaction**: Talk to Alex using natural speech
- **Tool Calling**: Web search (DuckDuckGo) and database queries (SQL) via voice
- **Session Persistence**: Conversation history maintained across calls
- **Noise Cancellation**: Built-in noise reduction for clearer audio
- **Turn Detection**: Smart detection of when user stops speaking
- **Streaming Responses**: Low-latency streaming for natural conversation flow

## Installation

### 1. Install Dependencies

```bash
# Install base dependencies
pip install -e .

# Install LiveKit voice dependencies (choose your providers)
pip install livekit-plugins-assemblyai  # For STT
pip install livekit-plugins-cartesia    # For TTS

# Or install all voice providers at once
pip install -e ".[voice-all]"
```

### 2. Set Up Environment Variables

Copy the example environment file and fill in your API keys:

```bash
cp .env.livekit.example .env
```

Required environment variables:

```env
# LiveKit Configuration
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your-api-key
LIVEKIT_API_SECRET=your-api-secret

# AI/ML API Keys
OPENAI_API_KEY=your-openai-api-key
ASSEMBLYAI_API_KEY=your-assemblyai-api-key
CARTESIA_API_KEY=your-cartesia-api-key

# Database (existing setup)
DATABASE_URL=libsql://your-database.turso.io
DATABASE_AUTH_TOKEN=your-auth-token
```

### 3. Get API Keys

- **LiveKit**: Sign up at [LiveKit Cloud](https://cloud.livekit.io) or [self-host](https://docs.livekit.io/home/self-hosting/)
- **AssemblyAI**: Get a key at [AssemblyAI](https://www.assemblyai.com/)
- **Cartesia**: Get a key at [Cartesia](https://cartesia.ai/)
- **OpenAI**: Get a key at [OpenAI](https://platform.openai.com/)

## Usage

### Running the Voice Agent

```bash
# Development mode (with hot reload)
python -m app.livekit_agent dev

# Production mode
python -m app.livekit_agent start
```

### Connecting to the Agent

#### Option 1: LiveKit Playground
1. Go to your LiveKit Cloud dashboard
2. Navigate to your project
3. Use the built-in playground to connect

#### Option 2: Custom Web Client
1. Open `livekit_test.html` in a browser
2. Enter your LiveKit URL and a room token
3. Click the call button to connect

#### Option 3: Programmatic Connection
```python
from livekit import api

# Generate a room token
token = api.AccessToken(
    api_key="your-api-key",
    api_secret="your-api-secret"
).with_identity("user-123") \
 .with_name("John Doe") \
 .with_grants(api.VideoGrants(room_join=True, room="my-room")) \
 .to_jwt()
```

## Project Structure

```
2Quip_agent/
├── livekit_agent.py            # Main LiveKit voice agent
├── livekit_test.html           # Web client for testing
├── .env.livekit.example        # Environment variables template
├── livekit_plugins_agno/       # Agno plugin for LiveKit
│   ├── __init__.py             # Plugin registration
│   ├── agno.py                 # LLMAdapter + AgnoStream
│   └── version.py              # Version info
└── app/
    └── services/
        └── agno_service.py     # Existing Agno service
```

## Customization

### Change the Voice

Edit `livekit_agent.py` to use a different TTS voice:

```python
tts=inference.TTS(
    model="cartesia/sonic-3",
    voice="your-preferred-voice-id"
)
```

### Change STT/TTS Providers

You can use different providers:

```python
# Deepgram instead of AssemblyAI
from livekit.plugins import deepgram
stt=deepgram.STT()
tts=deepgram.TTS()

# OpenAI Realtime (for ultra-low latency)
from livekit.plugins import openai
llm=openai.realtime.RealtimeModel(voice="marin")
```

### Add Custom Tools

Add tools to the Agno agent in `livekit_agent.py`:

```python
from agno.tools import tool

@tool
def get_work_order_status(order_id: str) -> str:
    """Get the status of a work order."""
    # Your implementation
    return f"Work order {order_id} is in progress"

# Add to agent
agent = AgnoAgent(
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[ddg_tools, sql_tools, get_work_order_status],
    ...
)
```

## Troubleshooting

### Common Issues

1. **"Connection refused" error**
   - Make sure the agent is running (`python livekit_agent.py dev`)
   - Check your LiveKit URL and credentials

2. **No audio from agent**
   - Verify your TTS API key (CARTESIA_API_KEY)
   - Check browser console for errors

3. **Agent not responding**
   - Verify your OPENAI_API_KEY is valid
   - Check agent logs for errors

4. **Poor audio quality**
   - Ensure noise cancellation is enabled
   - Use a good microphone

### Logs

View agent logs:
```bash
# The agent logs to stdout by default
python livekit_agent.py dev

# For more verbose logging
LOGLEVEL=DEBUG python livekit_agent.py dev
```

## API Reference

### LLMAdapter

Wraps an Agno Agent for use with LiveKit:

```python
from livekit_plugins_agno import LLMAdapter

livekit_llm = LLMAdapter(
    agent=agno_agent,           # The Agno Agent to wrap
    session_id="session-123",   # Optional session ID for state
    user_id="user-456",         # Optional user ID for memory
)
```

### AgnoStream

Internal class that handles streaming responses from Agno to LiveKit. You typically don't interact with this directly.

## Resources

- [LiveKit Documentation](https://docs.livekit.io/)
- [Agno Documentation](https://docs.agno.com/)
- [LiveKit Agents Guide](https://docs.livekit.io/agents/)
- [AssemblyAI Documentation](https://www.assemblyai.com/docs/)
- [Cartesia Documentation](https://docs.cartesia.ai/)

## License

MIT
