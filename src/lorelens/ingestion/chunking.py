"""Chunking strategies for technical documentation.

* ``token``      – fixed token windows with overlap (baseline).
* ``markdown``   – split on Markdown headers first, then token windows inside each section;
                   every chunk carries its header breadcrumb.
* ``code_aware`` – like ``markdown`` but packs prose/code blocks greedily and never splits a
                   fenced code block unless it alone exceeds the budget (then it is split on
                   line boundaries and re-fenced).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import tiktoken
from langchain_text_splitters import (
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from lorelens.models import Chunk, DocMetadata, SourceDocument

_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]
_FENCE_RE = re.compile(r"(^```[^\n]*\n.*?^```[ \t]*$)", re.MULTILINE | re.DOTALL)


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text, disallowed_special=()))


@dataclass
class Section:
    text: str
    breadcrumb: str | None


class Chunker(Protocol):
    def split(self, doc: SourceDocument, metadata: DocMetadata) -> list[Chunk]: ...


def _token_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=RecursiveCharacterTextSplitter.get_separators_for_language(Language.MARKDOWN),
    )


def split_sections(text: str) -> list[Section]:
    """Split Markdown by headers, keeping header lines in the body for readability."""
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_HEADERS, strip_headers=False)
    sections: list[Section] = []
    for part in splitter.split_text(text):
        crumbs = [part.metadata[k] for _, k in _HEADERS if k in part.metadata]
        sections.append(Section(text=part.page_content, breadcrumb=" > ".join(crumbs) or None))
    return sections or [Section(text=text, breadcrumb=None)]


class _BaseChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = _token_splitter(chunk_size, chunk_overlap)

    def _pieces(self, doc: SourceDocument) -> list[tuple[str, str | None]]:
        raise NotImplementedError

    def split(self, doc: SourceDocument, metadata: DocMetadata) -> list[Chunk]:
        chunks: list[Chunk] = []
        for text, breadcrumb in self._pieces(doc):
            text = text.strip()
            if len(text) < 20:  # drop empty headers / stray separators
                continue
            position = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{position}",
                    doc_id=doc.doc_id,
                    text=text,
                    section=breadcrumb,
                    position=position,
                    token_count=count_tokens(text),
                    metadata=metadata,
                )
            )
        return chunks


class TokenChunker(_BaseChunker):
    def _pieces(self, doc: SourceDocument) -> list[tuple[str, str | None]]:
        return [(t, None) for t in self._splitter.split_text(doc.text)]


class MarkdownChunker(_BaseChunker):
    def _pieces(self, doc: SourceDocument) -> list[tuple[str, str | None]]:
        pieces: list[tuple[str, str | None]] = []
        for section in split_sections(doc.text):
            if count_tokens(section.text) <= self.chunk_size:
                pieces.append((section.text, section.breadcrumb))
            else:
                pieces.extend((t, section.breadcrumb) for t in self._splitter.split_text(section.text))
        return pieces


class CodeAwareChunker(_BaseChunker):
    def _split_code_block(self, block: str) -> list[str]:
        lines = block.splitlines()
        opener, body = lines[0], lines[1:-1]
        parts: list[str] = []
        current: list[str] = []
        budget = self.chunk_size - 8  # leave room for the fences
        for line in body:
            candidate = "\n".join([*current, line])
            if current and count_tokens(candidate) > budget:
                parts.append("\n".join([opener, *current, "```"]))
                current = []
            current.append(line)
        if current:
            parts.append("\n".join([opener, *current, "```"]))
        return parts

    def _blocks(self, text: str) -> list[str]:
        blocks: list[str] = []
        for i, part in enumerate(_FENCE_RE.split(text)):
            if not part.strip():
                continue
            is_code = i % 2 == 1  # re.split with one capture group alternates prose/code
            if count_tokens(part) <= self.chunk_size:
                blocks.append(part)
            elif is_code:
                blocks.extend(self._split_code_block(part.strip()))
            else:
                blocks.extend(self._splitter.split_text(part))
        return blocks

    def _pieces(self, doc: SourceDocument) -> list[tuple[str, str | None]]:
        pieces: list[tuple[str, str | None]] = []
        for section in split_sections(doc.text):
            buffer: list[str] = []
            size = 0
            for block in self._blocks(section.text):
                n = count_tokens(block)
                if buffer and size + n > self.chunk_size:
                    pieces.append(("\n\n".join(buffer), section.breadcrumb))
                    buffer, size = [], 0
                buffer.append(block.strip("\n"))
                size += n
            if buffer:
                pieces.append(("\n\n".join(buffer), section.breadcrumb))
        return pieces


_REGISTRY: dict[str, Callable[[int, int], Chunker]] = {
    "token": TokenChunker,
    "markdown": MarkdownChunker,
    "code_aware": CodeAwareChunker,
}


def get_chunker(strategy: str, chunk_size: int, chunk_overlap: int) -> Chunker:
    try:
        return _REGISTRY[strategy](chunk_size, chunk_overlap)
    except KeyError as exc:
        raise ValueError(f"Unknown chunk strategy {strategy!r}; choose from {list(_REGISTRY)}") from exc
