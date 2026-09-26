"""Application settings, read from the environment or `.env`."""

import logging
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Every knob the application has. Defaults match `docker-compose.yml`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # A blank `OPENAI_API_KEY=` in .env.example must read as unset, not as "".
        env_ignore_empty=True,
    )

    llm_provider: str = "openai"  # key of providers.PROVIDERS
    # Defaulted, not required: pyright rejects `Settings()` when a field has no default.
    openai_api_key: SecretStr = SecretStr("")
    openai_chat_model: str = "gpt-4o"
    # The intent gate classifies and lists ingredients, so it does not need the answer model.
    openai_intent_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimension: int = Field(default=1536, gt=0)

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "challenge"
    postgres_user: str = "pipeline"
    postgres_password: SecretStr = SecretStr("pipeline-pass")

    recipes_table_prefix: str = "recipes"
    recipes_dir: Path = Path("data/recipes")
    # Off = skip the gate entirely (no LLM call): every request goes straight to retrieval and
    # is answered with the closest cookbook match, however off-topic the text is.
    intent_gate_enabled: bool = True
    retriever_top_k: int = Field(default=3, ge=1, le=10)
    retriever_candidates: int = Field(default=8, ge=1, le=50)
    max_image_bytes: int = 10 * 1024 * 1024
    log_level: str = "INFO"

    @property
    def postgres_dsn(self) -> str:
        """Connection string for the pgvector document store."""
        password = self.postgres_password.get_secret_value()
        return (
            f"postgresql://{self.postgres_user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


def configure_logging(level: str) -> None:
    """Set up stdlib logging. Values belong in the message, never only in `extra=`."""
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
