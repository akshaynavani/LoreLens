"""Derive normalized, filterable metadata (product, version, doc type, tags) for a page."""

from __future__ import annotations

import hashlib
import re
from pathlib import PurePosixPath

from lorelens.models import DocMetadata, DocType, SourceDocument

_VERSION_SEGMENT_RE = re.compile(r"v?(\d+(?:\.\d+){0,2}|latest|stable)", re.I)

_DOC_TYPE_RULES: list[tuple[DocType, re.Pattern[str]]] = [
    ("api_reference", re.compile(r"(api|reference|endpoints?|sdk|cli-reference)", re.I)),
    ("release_notes", re.compile(r"(release[-_ ]?notes|changelog|what'?s[-_ ]new)", re.I)),
    ("tutorial", re.compile(r"(tutorials?|quick[-_ ]?start|getting[-_ ]started|walkthrough)", re.I)),
    ("faq", re.compile(r"(faq|troubleshoot)", re.I)),
    ("concept", re.compile(r"(concepts?|overview|architecture|explanation)", re.I)),
    ("guide", re.compile(r"(guides?|how[-_ ]?to|docs)", re.I)),
]

_GENERIC_SEGMENTS = {"docs", "doc", "documentation", "en", "content", "pages", "www", "latest"}


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _path_segments(doc: SourceDocument) -> list[str]:
    raw = doc.url or doc.source
    raw = re.sub(r"^https?://[^/]+", "", raw)
    return [s for s in PurePosixPath(raw).parts if s not in ("/", "")]


def normalize_version(value: str) -> str:
    """`2`, `v2`, `V2` -> `v2`; `latest`/`stable` are kept as-is."""
    value = str(value).strip().lower()
    if value in ("latest", "stable"):
        return value
    return value if value.startswith("v") else f"v{value}"


def detect_version(doc: SourceDocument) -> str | None:
    if v := doc.metadata.get("version"):
        return normalize_version(v)
    for seg in _path_segments(doc)[:-1]:
        if m := _VERSION_SEGMENT_RE.fullmatch(seg):
            return normalize_version(m.group(1))
    return None


def detect_product(doc: SourceDocument) -> str | None:
    """Product is the first meaningful path segment (e.g. `payments/v2/...` -> `payments`)."""
    if p := doc.metadata.get("product"):
        return str(p).lower()
    for seg in _path_segments(doc)[:-1]:
        s = seg.lower()
        if s in _GENERIC_SEGMENTS or re.fullmatch(r"v?\d+(\.\d+)*", s):
            continue
        return s
    return None


def detect_doc_type(doc: SourceDocument) -> DocType:
    if t := doc.metadata.get("doc_type"):
        return t  # type: ignore[return-value]
    haystack = " ".join(_path_segments(doc)) + " " + (doc.title or "")
    for doc_type, pattern in _DOC_TYPE_RULES:
        if pattern.search(haystack):
            return doc_type
    # Heuristic: pages dominated by code fences are usually reference material.
    fences = doc.text.count("```")
    if fences >= 6 and fences * 200 > len(doc.text) * 0.5:
        return "api_reference"
    return "other"


def extract_metadata(doc: SourceDocument) -> DocMetadata:
    tags = doc.metadata.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    return DocMetadata(
        doc_id=doc.doc_id,
        source=doc.source,
        url=doc.url,
        title=doc.title,
        product=detect_product(doc),
        version=detect_version(doc),
        doc_type=detect_doc_type(doc),
        tags=[str(t).lower() for t in tags],
        content_hash=content_hash(doc.text),
    )
