from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Hafawah WhatsApp Agent"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    evolution_base_url: str = Field(default="https://your-evolution-api.example.com")
    evolution_api_key: str = Field(default="")
    evolution_instance: str = Field(default="os11")

    openrouter_api_key: str = Field(default="")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")
    openrouter_text_model: str = Field(default="google/gemini-2.5-flash")
    openrouter_vision_model: str = Field(default="google/gemini-2.5-flash")
    openrouter_audio_model: str = Field(default="google/gemini-2.5-flash")

    hotel_name: str = Field(default="حفاوة")
    hotel_city: str = Field(default="الرياض")
    hotel_timezone: str = Field(default="Asia/Riyadh")
    hotel_phone: str = Field(default="+966500000000")
    reception_whatsapp: str = Field(default="+966500000001")

    database_url: str = Field(default=f"sqlite:///{(DATA_DIR / 'agent.db').as_posix()}")
    knowledge_file: str = Field(default=(DATA_DIR / "knowledge" / "hafawah.md").as_posix())
    bookings_export_file: str = Field(default=(DATA_DIR / "exports" / "booking_requests.csv").as_posix())

    langfuse_public_key: str = Field(default="")
    langfuse_secret_key: str = Field(default="")
    langfuse_host: str = Field(
        default="",
        validation_alias=AliasChoices("LANGFUSE_HOST", "LANGFUSE_BASE_URL"),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "knowledge").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "exports").mkdir(parents=True, exist_ok=True)
    return Settings()
