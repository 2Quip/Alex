from pydantic_settings import BaseSettings
from sqlalchemy import create_engine

class Settings(BaseSettings):
    # Database Configuration

    DATABASE_URL: str
    DATABASE_AUTH_TOKEN: str

    GROQ_API_KEY: str
    OPENROUTER_API_KEY: str
    OPENAI_API_KEY: str

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

    class Config:
        env_file = ".env"


settings = Settings()
