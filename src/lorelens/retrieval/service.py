"""DocSearchService: the single retrieval core behind every tool transport.

query -> (cached) embedding -> Pinecone top-`candidate_k` with metadata filter
      -> score floor -> optional cross-encoder rerank -> section de-dup -> top-`k`
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from functools import lru_cache

from langchain_core.embeddings import Embeddings

from lorelens.config import Settings, get_settings
from lorelens.ingestion.manifest import Manifest
from lorelens.log import get_logger
from lorelens.models import RetrievedChunk, SearchFilters
from lorelens.retrieval.cache import RetrievalCaches, make_key
from lorelens.retrieval.rerank import CrossEncoderReranker, NoopReranker, Reranker
from lorelens.vectorstore import PineconeStore

logger = get_logger(__name__)


def _dedupe(chunks: list[RetrievedChunk], per_section: int = 2) -> list[RetrievedChunk]:
    """Keep at most `per_section` chunks from the same page section so a long section
    cannot crowd out other relevant pages."""
    seen: dict[tuple[str, str | None], int] = {}
    out: list[RetrievedChunk] = []
    for c in chunks:
        key = (c.doc_id, c.section)
        if seen.get(key, 0) >= per_section:
            continue
        seen[key] = seen.get(key, 0) + 1
        out.append(c)
    return out


class DocSearchService:
    def __init__(
        self,
        store: PineconeStore,
        embeddings: Embeddings,
        settings: Settings | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store
        self.embeddings = embeddings
        if reranker is None:
            reranker = (
                CrossEncoderReranker(self.settings.rerank_model)
                if self.settings.rerank_enabled
                else NoopReranker()
            )
        self.reranker = reranker
        self.caches = RetrievalCaches(
            self.settings.retrieval_cache_ttl, self.settings.retrieval_cache_size
        )

    # ------------------------------------------------------------------ search
    def embed_query(self, query: str) -> list[float]:
        return self.caches.query_embeddings.get_or_set(
            make_key(self.settings.embedding_model, query.strip().lower()),
            lambda: self.embeddings.embed_query(query),
        )

    def _search_uncached(
        self, query: str, filters: SearchFilters, top_k: int, namespace: str | None
    ) -> list[RetrievedChunk]:
        t0 = time.perf_counter()
        vector = self.embed_query(query)
        t1 = time.perf_counter()
        candidates = self.store.query(
            vector,
            top_k=max(self.settings.candidate_k, top_k),
            filter=filters.to_pinecone(),
            namespace=namespace,
        )
        t2 = time.perf_counter()
        candidates = [c for c in candidates if c.score >= self.settings.min_score]
        ranked = self.reranker.rerank(query, _dedupe(candidates), top_k)
        t3 = time.perf_counter()
        logger.debug(
            "search q=%r embed=%.0fms pinecone=%.0fms rerank=%.0fms hits=%d",
            query, (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, len(ranked),
        )
        return ranked

    def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
        namespace: str | None = None,
        *,
        relax_filters: bool = True,
    ) -> list[RetrievedChunk]:
        """Semantic search with metadata filtering.

        When a filtered search finds nothing (e.g. the model guessed a version that does not
        exist), filters are relaxed progressively: drop doc_type/tags, then version, then all.
        """
        filters = filters or SearchFilters()
        top_k = top_k or self.settings.top_k

        attempts = [filters]
        if relax_filters and not filters.is_empty():
            attempts += [
                SearchFilters(product=filters.product, version=filters.version),
                SearchFilters(product=filters.product),
                SearchFilters(),
            ]
        tried: set[str] = set()
        results: list[RetrievedChunk] = []
        for attempt in attempts:
            key = make_key(query, attempt.model_dump(), top_k, namespace or self.store.namespace)
            if key in tried:
                continue
            tried.add(key)
            results = self.caches.results.get_or_set(
                key, lambda a=attempt: self._search_uncached(query, a, top_k, namespace)
            )
            if results:
                if attempt is not filters:
                    logger.info("Relaxed filters %s -> %s for %r", filters, attempt, query)
                break
        return results

    async def asearch(self, query: str, filters: SearchFilters | None = None,
                      top_k: int | None = None, namespace: str | None = None
                      ) -> list[RetrievedChunk]:
        return await asyncio.to_thread(self.search, query, filters, top_k, namespace)

    async def amulti_search(
        self, queries: Sequence[str], filters: SearchFilters | None = None, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        """Run sub-queries in parallel and merge by best score per chunk."""
        batches = await asyncio.gather(*(self.asearch(q, filters, top_k) for q in queries))
        merged: dict[str, RetrievedChunk] = {}
        for batch in batches:
            for c in batch:
                if c.chunk_id not in merged or c.best_score > merged[c.chunk_id].best_score:
                    merged[c.chunk_id] = c
        return sorted(merged.values(), key=lambda c: c.best_score, reverse=True)

    # ------------------------------------------------------------- page access
    def get_page(self, doc_id: str, namespace: str | None = None, max_chars: int = 12_000) -> dict:
        """Reassemble a full page from its chunks (useful for follow-up reading)."""
        ids = self.store.list_ids(prefix=f"{doc_id}#", namespace=namespace)
        if not ids:
            return {"doc_id": doc_id, "found": False}
        chunks = self.store.fetch(ids, namespace=namespace)
        chunks.sort(key=lambda c: int(c.chunk_id.rsplit("#", 1)[-1]))
        text = "\n\n".join(c.text for c in chunks)
        first = chunks[0]
        return {
            "doc_id": doc_id,
            "found": True,
            "title": first.title,
            "url": first.url,
            "source": first.source,
            "product": first.product,
            "version": first.version,
            "text": text[:max_chars],
            "truncated": len(text) > max_chars,
        }

    def catalog(self) -> dict[str, dict[str, int]]:
        manifest = Manifest.load(
            self.settings.manifest_path, self.store.namespace, self.settings.embedding_model
        )
        return manifest.catalog()


@lru_cache(maxsize=1)
def get_search_service() -> DocSearchService:
    from lorelens.embeddings import get_embeddings

    settings = get_settings()
    return DocSearchService(PineconeStore(settings), get_embeddings(), settings)
