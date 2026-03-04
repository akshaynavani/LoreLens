"""Ingestion manifest: remembers each page's content hash and chunk ids so re-ingests are
incremental (unchanged pages skipped, removed chunks deleted from Pinecone)."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class ManifestEntry(BaseModel):
    source: str
    content_hash: str
    chunk_ids: list[str] = Field(default_factory=list)
    strategy: str


class Manifest(BaseModel):
    namespace: str
    embedding_model: str
    entries: dict[str, ManifestEntry] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path, namespace: str, embedding_model: str) -> Manifest:
        if path.exists():
            data = cls.model_validate_json(path.read_text())
            # A different namespace or embedding model invalidates everything.
            if data.namespace == namespace and data.embedding_model == embedding_model:
                return data
        return cls(namespace=namespace, embedding_model=embedding_model)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.model_dump(), indent=2))
        tmp.replace(path)

    def is_unchanged(self, doc_id: str, content_hash: str, strategy: str) -> bool:
        entry = self.entries.get(doc_id)
        return bool(entry and entry.content_hash == content_hash and entry.strategy == strategy)
