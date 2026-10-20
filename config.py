from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_provider: Literal["anthropic", "openai", "huggingface"] = "anthropic"
    anthropic_api_key: str | None = None
    model_name: str = "claude-sonnet-5-5"
    openai_api_key: str | None = None
    openai_model_name: str = "gpt-4o-mini"
    huggingface_api_key: str | None = None
    huggingface_model_name: str = "Qwen/Qwen2.5-72B-Instruct:deepinfra"
    model_base_url: str | None = None
    http_timeout: float = 20.0


settings = Settings()