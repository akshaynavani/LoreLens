"""Typed application settings loaded from environment variables / `.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Known embedding dimensions so the Pinecone index can be created without a probe call.
EMBEDDING_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="LORELENS_",
        extra="ignore",
    )

    # --- OpenAI ---
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    chat_model: str = "gpt-4o"
    fast_model: str = "gpt-4o-mini"

    # --- Embeddings ---
    embedding_provider: Literal["openai", "huggingface"] = "openai"
    openai_embedding_model: str = "text-embedding-3-small"
    hf_embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_cache_dir: Path = Path(".cache/embeddings")
    embedding_batch_size: int = 128

    # --- Pinecone ---
    pinecone_api_key: SecretStr | None = Field(default=None, validation_alias="PINECONE_API_KEY")
    pinecone_index: str = "lorelens-docs"
    pinecone_namespace: str = "default"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    upsert_batch_size: int = 100

    # --- Chunking ---
    chunk_strategy: Literal["markdown", "code_aware", "token"] = "markdown"
    chunk_size: int = 512
    chunk_overlap: int = 64

    # --- Ingestion ---
    manifest_path: Path = Path(".cache/manifest.json")
    docs_base_url: str | None = None

    # --- Retrieval ---
    top_k: int = 8
    candidate_k: int = 24
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    retrieval_cache_ttl: int = 300
    retrieval_cache_size: int = 1024
    min_score: float = 0.2

    # --- Agent ---
    tool_mode: Literal["local", "mcp"] = "local"
    max_rewrites: int = 2
    max_tool_calls: int = 6
    mcp_server_command: str = "python"
    mcp_server_args: list[str] = Field(default_factory=lambda: ["-m", "lorelens.mcp_server"])

    # --- LangFuse ---
    langfuse_public_key: str | None = Field(default=None, validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: SecretStr | None = Field(
        default=None, validation_alias="LANGFUSE_SECRET_KEY"
    )
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com", validation_alias="LANGFUSE_HOST"
    )
    prompt_label: str = "production"

    # --- API ---
    api_key: SecretStr | None = None
    log_level: str = "INFO"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    @property
    def embedding_model(self) -> str:
        if self.embedding_provider == "openai":
            return self.openai_embedding_model
        return self.hf_embedding_model

    @property
    def embedding_dimension(self) -> int:
        try:
            return EMBEDDING_DIMENSIONS[self.embedding_model]
        except KeyError as exc:  # pragma: no cover - configuration error
            raise ValueError(
                f"Unknown dimension for embedding model {self.embedding_model!r}; "
                "add it to EMBEDDING_DIMENSIONS"
            ) from exc

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
