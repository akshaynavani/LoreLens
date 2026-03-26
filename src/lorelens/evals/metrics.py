"""Retrieval and latency metrics (pure functions, no I/O)."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _norm(source: str | None) -> str:
    return (source or "").strip().lower().rstrip("/")


def _matches(retrieved: str | None, expected: set[str]) -> bool:
    r = _norm(retrieved)
    # Allow URL vs relative-path comparisons: `https://docs.x/nimbus/v2/a` ends with `nimbus/v2/a`.
    return bool(r) and any(r == e or r.endswith("/" + e) or e.endswith("/" + r) for e in expected)


def hit_rate_at_k(retrieved: Sequence[str | None], expected: Sequence[str], k: int) -> float:
    exp = {_norm(e) for e in expected}
    if not exp:
        return math.nan
    return 1.0 if any(_matches(r, exp) for r in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: Sequence[str | None], expected: Sequence[str]) -> float:
    exp = {_norm(e) for e in expected}
    if not exp:
        return math.nan
    for rank, r in enumerate(retrieved, start=1):
        if _matches(r, exp):
            return 1.0 / rank
    return 0.0


def recall(retrieved: Sequence[str | None], expected: Sequence[str]) -> float:
    exp = {_norm(e) for e in expected}
    if not exp:
        return math.nan
    found = {e for e in exp if any(_matches(r, {e}) for r in retrieved)}
    return len(found) / len(exp)


def citation_precision(cited: Sequence[str | None], expected: Sequence[str]) -> float:
    exp = {_norm(e) for e in expected}
    if not cited or not exp:
        return math.nan
    return sum(1 for c in cited if _matches(c, exp)) / len(cited)


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct / 100
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)


def mean(values: Sequence[float]) -> float:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    return sum(clean) / len(clean) if clean else math.nan
