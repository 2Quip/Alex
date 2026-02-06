from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    # Database Configuration
    DATABASE_URL: str
    DATABASE_AUTH_TOKEN: str

    # LLM API Keys
    GROQ_API_KEY: str
    OPENROUTER_API_KEY: str
    OPENAI_API_KEY: str

    # LiveKit Configuration (optional - only needed for voice agent)
    LIVEKIT_URL: Optional[str] = None
    LIVEKIT_API_KEY: Optional[str] = None
    LIVEKIT_API_SECRET: Optional[str] = None

    # STT/TTS API Keys (optional - for LiveKit voice agent)
    ASSEMBLYAI_API_KEY: Optional[str] = None
    CARTESIA_API_KEY: Optional[str] = None
    DEEPGRAM_API_KEY: Optional[str] = None

    # Document webhook (optional - for sending documents to users during calls)
    DOCUMENT_WEBHOOK_URL: Optional[str] = None
    DOCUMENT_WEBHOOK_SECRET: Optional[str] = None

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/agno_agent_api.log"

    @property
    def db_engine(self) -> str:
        engine = create_engine(
            f"sqlite+{self.DATABASE_URL}?secure=true",
            connect_args={
                "auth_token": self.DATABASE_AUTH_TOKEN,
            },
            pool_recycle=120,
            pool_pre_ping=True
        )
        return engine



settings = Settings()
