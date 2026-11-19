from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent
ENV_FILE = BACKEND_DIR / ".env"

load_dotenv(ENV_FILE)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    default_model: str = "gpt-4o-mini"
    orchestrator_model: str | None = None
    risk_model: str | None = None
    recommendation_model: str | None = None

    data_path: Path = BACKEND_DIR / "data" / "archive" / "diabetes.csv"
    db_path: Path = BACKEND_DIR / "data" / "app.db"

    session_history_limit: int = 20
    max_turns: int = 8
    max_query_chars: int = 2000

    cors_origins: list[str] = ["http://localhost:5173"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
