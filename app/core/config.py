from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "deep-research-agent"
    planner_llm_provider: str = "anthropic"
    planner_llm_model: str = "claude-sonnet-5-5"
    researcher_llm_provider: str = "gemini"
    researcher_llm_model: str = "gemini-3.5-flash-lite"
    writer_llm_provider: str = "anthropic"
    writer_llm_model: str = "claude-sonnet-5-5"
    reviewer_llm_provider: str = "anthropic"
    reviewer_llm_model: str = "claude-sonnet-5-5"
    anthropic_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "anthropic_api_key", "llm_api_key"),
    )
    gemini_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("GEMINI_API_KEY", "gemini_api_key"),
    )
    tavily_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("TAVILY_API_KEY", "tavily_api_key"),
    )
    vector_db_path: str = "data/chroma_db"
    vector_db_collection_name: str = "genai_banking_research"
    checkpoint_db_path: str = "data/checkpoints.sqlite"
    output_dir: str = "outputs"
    quality_threshold: float = 0.8
    max_revision_attempts: int = 3
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
