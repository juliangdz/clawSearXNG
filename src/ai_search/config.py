"""Application configuration via pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration, loaded from environment / .env file."""

    anthropic_api_key: str
    redis_url: str = "redis://localhost:6379/0"
    searxng_url: str = "http://localhost:8888"
    cache_ttl_hours: int = 24
    max_results: int = 8
    port: int = 7777
    log_level: str = "INFO"
    environment: str = "development"
    enable_cross_encoder: bool = False

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# Module-level singleton; imported everywhere.
settings = Settings()
