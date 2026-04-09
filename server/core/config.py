"""Application configuration loaded from environment / .env file."""

from __future__ import annotations

from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    database_url: str = "postgresql+asyncpg://ccr:ccr@localhost:5432/ccr"

    # JWT
    jwt_secret: str = "change_me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 480

    # CORS — comma-separated origins string, parsed into a list
    cors_origins: str = "http://localhost:5173,http://localhost:8000"

    # Market data
    fred_api_key: str = ""

    # SMTP (optional)
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = "ccr-alerts@example.com"

    # Scheduler killswitch — set SCHEDULER_ENABLED=false to disable all
    # background jobs without redeploying (useful during incidents).
    scheduler_enabled: bool = True

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
