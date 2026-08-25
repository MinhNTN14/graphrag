"""Application configuration loaded from environment variables.

Uses pydantic-settings ``BaseSettings`` so values can be provided via a ``.env``
file or the process environment. A cached singleton is exposed via
``get_settings()``.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly typed application settings.

    All fields map to the environment variables documented in ``.env.example``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Gemini ---------------------------------------------------------
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")

    # --- Neo4j ----------------------------------------------------------
    neo4j_uri: str = Field(default="bolt://neo4j:7687", alias="NEO4J_URI")
    neo4j_username: str = Field(default="neo4j", alias="NEO4J_USERNAME")
    neo4j_password: str = Field(default="password123", alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")

    # --- Ingestion / retrieval -----------------------------------------
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP")
    top_k_results: int = Field(default=5, alias="TOP_K_RESULTS")

    # --- API ------------------------------------------------------------
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @property
    def neo4j_auth(self) -> tuple[str, str]:
        """Return the ``(username, password)`` tuple used by the Neo4j driver."""
        return (self.neo4j_username, self.neo4j_password)


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance."""
    return Settings()
