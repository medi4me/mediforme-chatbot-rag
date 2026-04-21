"""애플리케이션 설정 — 환경변수 로딩."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "local"
    log_level: str = "INFO"

    openai_api_key: str = ""
    database_url: str = "postgresql://mediforme:mediforme@localhost:5432/mediforme_rag"

    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072

    default_top_k: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
