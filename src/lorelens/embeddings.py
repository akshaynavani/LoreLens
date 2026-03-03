"""Embedding provider factory (OpenAI or HuggingFace) with an on-disk cache."""

from __future__ import annotations

from functools import lru_cache

from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_core.embeddings import Embeddings

from lorelens.config import Settings, get_settings


def _base_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        if settings.openai_api_key is None:
            raise RuntimeError("OPENAI_API_KEY is required for the OpenAI embedding provider")
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
            chunk_size=settings.embedding_batch_size,
            max_retries=6,
        )

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the `huggingface` extra: pip install 'lorelens[huggingface]'") from exc

    return HuggingFaceEmbeddings(
        model_name=settings.hf_embedding_model,
        encode_kwargs={"normalize_embeddings": True, "batch_size": settings.embedding_batch_size},
    )


def build_embeddings(settings: Settings | None = None, *, cache: bool = True) -> Embeddings:
    """Return the configured embedder; document embeddings are cached by content hash so
    re-ingesting an unchanged corpus or a different chunking experiment costs nothing."""
    settings = settings or get_settings()
    base = _base_embeddings(settings)
    if not cache:
        return base
    settings.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
    store = LocalFileStore(str(settings.embedding_cache_dir))
    return CacheBackedEmbeddings.from_bytes_store(
        base, store, namespace=settings.embedding_model.replace("/", "_")
    )


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    return build_embeddings()
