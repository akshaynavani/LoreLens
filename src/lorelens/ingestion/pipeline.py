"""End-to-end ingestion: documents -> metadata -> chunks -> embeddings -> Pinecone."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field

from langchain_core.embeddings import Embeddings

from lorelens.config import Settings, get_settings
from lorelens.ingestion.chunking import Chunker, get_chunker
from lorelens.ingestion.manifest import Manifest, ManifestEntry
from lorelens.ingestion.metadata import extract_metadata
from lorelens.log import get_logger
from lorelens.models import Chunk, SourceDocument
from lorelens.vectorstore import PineconeStore

logger = get_logger(__name__)


@dataclass
class IngestStats:
    documents_seen: int = 0
    documents_skipped: int = 0
    documents_indexed: int = 0
    documents_pruned: int = 0
    chunks_upserted: int = 0
    chunks_deleted: int = 0
    tokens_embedded: int = 0
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class IngestionPipeline:
    def __init__(
        self,
        store: PineconeStore,
        embeddings: Embeddings,
        settings: Settings | None = None,
        chunker: Chunker | None = None,
        namespace: str | None = None,
        flush_size: int = 256,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store
        self.embeddings = embeddings
        self.namespace = namespace or self.settings.pinecone_namespace
        self.chunker = chunker or get_chunker(
            self.settings.chunk_strategy, self.settings.chunk_size, self.settings.chunk_overlap
        )
        self.flush_size = flush_size
        self.manifest = Manifest.load(
            self.settings.manifest_path, self.namespace, self.settings.embedding_model
        )
        self._pending: list[Chunk] = []
        self._pending_entries: dict[str, ManifestEntry] = {}

    # ------------------------------------------------------------------ helpers
    def _flush(self, stats: IngestStats) -> None:
        if not self._pending:
            return
        texts = [c.embedding_text() for c in self._pending]
        vectors = self.embeddings.embed_documents(texts)
        stats.chunks_upserted += self.store.upsert_chunks(self._pending, vectors, self.namespace)
        stats.tokens_embedded += sum(c.token_count for c in self._pending)

        # Only after a successful upsert do we delete stale chunks and record the new state.
        for doc_id, entry in self._pending_entries.items():
            old = self.manifest.entries.get(doc_id)
            if old:
                stale = set(old.chunk_ids) - set(entry.chunk_ids)
                if stale:
                    self.store.delete_ids(sorted(stale), self.namespace)
                    stats.chunks_deleted += len(stale)
            self.manifest.entries[doc_id] = entry
        self.manifest.save(self.settings.manifest_path)
        self._pending.clear()
        self._pending_entries.clear()

    def _prune(self, seen: set[str], stats: IngestStats) -> None:
        missing = [doc_id for doc_id in self.manifest.entries if doc_id not in seen]
        for doc_id in missing:
            entry = self.manifest.entries.pop(doc_id)
            self.store.delete_ids(entry.chunk_ids, self.namespace)
            stats.chunks_deleted += len(entry.chunk_ids)
            stats.documents_pruned += 1
        if missing:
            self.manifest.save(self.settings.manifest_path)

    # --------------------------------------------------------------------- run
    def run(
        self, documents: Iterable[SourceDocument], *, force: bool = False, prune: bool = False
    ) -> IngestStats:
        stats = IngestStats()
        start = time.perf_counter()
        self.store.ensure_index()
        seen: set[str] = set()
        strategy = self.settings.chunk_strategy

        for doc in documents:
            stats.documents_seen += 1
            seen.add(doc.doc_id)
            try:
                meta = extract_metadata(doc)
                if not force and self.manifest.is_unchanged(doc.doc_id, meta.content_hash, strategy):
                    stats.documents_skipped += 1
                    continue
                chunks = self.chunker.split(doc, meta)
                if not chunks:
                    continue
                self._pending.extend(chunks)
                self._pending_entries[doc.doc_id] = ManifestEntry(
                    source=doc.source,
                    content_hash=meta.content_hash,
                    chunk_ids=[c.chunk_id for c in chunks],
                    strategy=strategy,
                )
                stats.documents_indexed += 1
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to process %s", doc.source)
                stats.errors.append(f"{doc.source}: {exc}")
                continue

            if len(self._pending) >= self.flush_size:
                self._flush(stats)
            if stats.documents_seen % 500 == 0:
                logger.info("Processed %d documents (%d chunks so far)",
                            stats.documents_seen, stats.chunks_upserted)

        self._flush(stats)
        if prune:
            self._prune(seen, stats)
        stats.seconds = round(time.perf_counter() - start, 2)
        logger.info("Ingestion finished: %s", stats.as_dict())
        return stats
