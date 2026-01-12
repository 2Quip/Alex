from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database Configuration
    db_host: str = "localhost"
    db_port: int = 3306
    db_user: str = "baas"
    db_password: str = "baas"
    db_name: str = "baas"

    # API Keys
    groq_api_key: str = ""

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000

    @property
    def db_url(self) -> str:
        return f"mysql+mysqldb://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    class Config:
        env_file = ".env"


settings = Settings()
