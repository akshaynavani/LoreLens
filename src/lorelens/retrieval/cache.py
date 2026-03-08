"""Small in-process caches that remove repeated embedding and Pinecone round trips.

Agent runs frequently re-issue the same (or an identical rewritten) query across graph
iterations and across users asking popular questions; caching both the query embedding
and the retrieval result is the single biggest latency win in the pipeline.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable
from typing import Any, TypeVar

from cachetools import LRUCache, TTLCache

T = TypeVar("T")


def make_key(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


class ThreadSafeCache:
    def __init__(self, cache: LRUCache | TTLCache) -> None:
        self._cache = cache
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get_or_set(self, key: str, factory: Callable[[], T]) -> T:
        with self._lock:
            if key in self._cache:
                self.hits += 1
                return self._cache[key]
        value = factory()
        with self._lock:
            self.misses += 1
            self._cache[key] = value
        return value

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "size": len(self._cache)}


class RetrievalCaches:
    def __init__(self, ttl_seconds: int, maxsize: int) -> None:
        self.query_embeddings = ThreadSafeCache(LRUCache(maxsize=maxsize * 4))
        self.results = ThreadSafeCache(TTLCache(maxsize=maxsize, ttl=ttl_seconds))

    def stats(self) -> dict[str, dict[str, int]]:
        return {"query_embeddings": self.query_embeddings.stats(), "results": self.results.stats()}
