"""HuggingFace cross-encoder reranker (optional, loaded lazily)."""

from __future__ import annotations

import threading
from typing import Protocol

from lorelens.log import get_logger
from lorelens.models import RetrievedChunk

logger = get_logger(__name__)


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]: ...


class CrossEncoderReranker:
    """Scores (query, chunk) pairs jointly; far more precise than bi-encoder similarity
    for the small candidate set returned by Pinecone."""

    def __init__(self, model_name: str, max_length: int = 512) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self._model = None
        self._lock = threading.Lock()

    def _load(self):  # noqa: ANN202
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    logger.info("Loading reranker %s", self.model_name)
                    self._model = CrossEncoder(self.model_name, max_length=self.max_length)
        return self._model

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        model = self._load()
        pairs = [(query, f"{c.title or ''} {c.section or ''}\n{c.text}") for c in chunks]
        scores = model.predict(pairs)
        rescored = [
            c.model_copy(update={"rerank_score": float(s)}) for c, s in zip(chunks, scores, strict=True)
        ]
        rescored.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)
        return rescored[:top_k]


class NoopReranker:
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        return sorted(chunks, key=lambda c: c.score, reverse=True)[:top_k]
