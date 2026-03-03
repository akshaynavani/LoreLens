"""Thin Pinecone wrapper: index lifecycle, batched upserts, filtered queries, deletes."""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from typing import Any

from pinecone import Pinecone, ServerlessSpec
from tenacity import retry, stop_after_attempt, wait_exponential

from lorelens.config import Settings, get_settings
from lorelens.log import get_logger
from lorelens.models import Chunk, RetrievedChunk

logger = get_logger(__name__)

# Pinecone caps metadata at 40KB per vector; keep stored chunk text well below it.
MAX_METADATA_TEXT = 30_000


def _batched(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


class PineconeStore:
    def __init__(self, settings: Settings | None = None, *, client: Pinecone | None = None) -> None:
        self.settings = settings or get_settings()
        if client is None:
            if self.settings.pinecone_api_key is None:
                raise RuntimeError("PINECONE_API_KEY is not set")
            client = Pinecone(api_key=self.settings.pinecone_api_key.get_secret_value())
        self.client = client
        self.index_name = self.settings.pinecone_index
        self.namespace = self.settings.pinecone_namespace
        self._index = None

    # ---------------------------------------------------------------- lifecycle
    def ensure_index(self) -> None:
        existing = self.client.list_indexes().names()
        if self.index_name in existing:
            desc = self.client.describe_index(self.index_name)
            if desc.dimension != self.settings.embedding_dimension:
                raise RuntimeError(
                    f"Index {self.index_name} has dimension {desc.dimension}, but embedding model "
                    f"{self.settings.embedding_model} produces {self.settings.embedding_dimension}. "
                    "Use a different LORELENS_PINECONE_INDEX."
                )
            return
        logger.info(
            "Creating Pinecone index %s (dim=%d)", self.index_name, self.settings.embedding_dimension
        )
        self.client.create_index(
            name=self.index_name,
            dimension=self.settings.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=self.settings.pinecone_cloud, region=self.settings.pinecone_region),
        )
        while not self.client.describe_index(self.index_name).status.get("ready"):
            time.sleep(1)

    @property
    def index(self):  # noqa: ANN201 - pinecone.Index type differs across SDK versions
        if self._index is None:
            self._index = self.client.Index(self.index_name)
        return self._index

    # ------------------------------------------------------------------ writes
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=20), reraise=True)
    def _upsert_batch(self, vectors: list[dict[str, Any]], namespace: str) -> None:
        self.index.upsert(vectors=vectors, namespace=namespace)

    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        namespace: str | None = None,
    ) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        ns = namespace or self.namespace
        payload = []
        for chunk, values in zip(chunks, vectors, strict=True):
            meta = chunk.metadata.to_pinecone()
            meta.update(
                text=chunk.text[:MAX_METADATA_TEXT],
                position=chunk.position,
                token_count=chunk.token_count,
            )
            if chunk.section:
                meta["section"] = chunk.section
            payload.append({"id": chunk.chunk_id, "values": list(values), "metadata": meta})
        for batch in _batched(payload, self.settings.upsert_batch_size):
            self._upsert_batch(list(batch), ns)
        return len(payload)

    def delete_ids(self, ids: Sequence[str], namespace: str | None = None) -> None:
        ns = namespace or self.namespace
        for batch in _batched(list(ids), 1000):
            self.index.delete(ids=list(batch), namespace=ns)

    # ------------------------------------------------------------------- reads
    @staticmethod
    def _to_retrieved(chunk_id: str, score: float, meta: dict[str, Any]) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=chunk_id,
            doc_id=meta.get("doc_id", chunk_id.split("#")[0]),
            text=meta.get("text", ""),
            score=float(score),
            title=meta.get("title"),
            url=meta.get("url"),
            source=meta.get("source"),
            section=meta.get("section"),
            product=meta.get("product"),
            version=meta.get("version"),
            doc_type=meta.get("doc_type"),
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=0.5, max=5), reraise=True)
    def query(
        self,
        vector: Sequence[float],
        top_k: int,
        filter: dict[str, Any] | None = None,  # noqa: A002 - mirrors Pinecone API
        namespace: str | None = None,
    ) -> list[RetrievedChunk]:
        res = self.index.query(
            vector=list(vector),
            top_k=top_k,
            filter=filter,
            include_metadata=True,
            namespace=namespace or self.namespace,
        )
        return [self._to_retrieved(m.id, m.score, m.metadata or {}) for m in res.matches]

    def fetch(self, ids: Sequence[str], namespace: str | None = None) -> list[RetrievedChunk]:
        res = self.index.fetch(ids=list(ids), namespace=namespace or self.namespace)
        out = [self._to_retrieved(vid, 1.0, v.metadata or {}) for vid, v in res.vectors.items()]
        order = {cid: i for i, cid in enumerate(ids)}
        return sorted(out, key=lambda c: order.get(c.chunk_id, 1_000_000))

    def stats(self) -> dict[str, Any]:
        s = self.index.describe_index_stats()
        return {
            "dimension": s.dimension,
            "total_vector_count": s.total_vector_count,
            "namespaces": {k: v.vector_count for k, v in (s.namespaces or {}).items()},
        }
