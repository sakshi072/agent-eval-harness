from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Optional
from pydantic import Field

class AgentSettings(BaseSettings):
    base_url: str = "http://localhost:8003"

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_prefix="AGENT_", 
        case_sensitive=False,
        extra="ignore"
    )

class RagasSettings(BaseSettings):
    generator_model: str = "gpt-4o-mini"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_prefix="RAGAS_", 
        case_sensitive=False,
        extra="ignore"
    )

class LangFuseSettings(BaseSettings):
    public_key: Optional[str] = None
    secret_key: Optional[str] = None
    base_url:str = "https://us.cloud.langfuse.com"

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_prefix="LANGFUSE_", 
        case_sensitive=False,
        extra="ignore"
    )

class Settings(BaseSettings):
    ENV: str = "development"
    PORT: int = 8004
    OPENAI_API_KEY: Optional[str] = None

    agent: AgentSettings = Field(default_factory=AgentSettings)
    ragas: RagasSettings = Field(default_factory=RagasSettings)
    langfuse: LangFuseSettings = Field(default_factory=LangFuseSettings)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance
    
    - Settings only loaded once (performance)
    - Same instance reused everywhere (consistency)
    - Can be easily mocked in tests
    """
    return Settings()

settings = get_settings()