"""Documentation loaders: local directories (Markdown/HTML/RST/text) and sitemap crawls."""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree

import frontmatter
import httpx
from bs4 import BeautifulSoup

from lorelens.log import get_logger
from lorelens.models import SourceDocument

logger = get_logger(__name__)

SUFFIX_FORMATS = {
    ".md": "markdown",
    ".mdx": "markdown",
    ".markdown": "markdown",
    ".html": "html",
    ".htm": "html",
    ".rst": "rst",
    ".txt": "text",
}

_STRIP_TAGS = ("script", "style", "nav", "footer", "header", "aside", "noscript")


def make_doc_id(source: str) -> str:
    """Stable, Pinecone-safe id derived from the source path / URL."""
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]


def html_to_markdownish(html: str) -> tuple[str, str | None]:
    """Convert HTML to light Markdown so the header-aware chunker still works.

    Returns (text, title).
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_STRIP_TAGS):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup

    title = None
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    for level in range(1, 7):
        for h in main.find_all(f"h{level}"):
            h.replace_with(f"\n\n{'#' * level} {h.get_text(' ', strip=True)}\n\n")
    for pre in main.find_all("pre"):
        code = pre.get_text()
        pre.replace_with(f"\n\n```\n{code.rstrip()}\n```\n\n")
    for code in main.find_all("code"):
        code.replace_with(f"`{code.get_text()}`")
    for li in main.find_all("li"):
        li.replace_with(f"\n- {li.get_text(' ', strip=True)}")

    text = main.get_text("\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), title


def rst_to_markdownish(text: str) -> str:
    """Translate RST section underlines into Markdown headers (good enough for chunking)."""
    lines = text.splitlines()
    out: list[str] = []
    levels: dict[str, int] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if (
            line.strip()
            and nxt
            and len(set(nxt.strip())) == 1
            and nxt.strip()[0] in "=-~^\"'`#*+"
            and len(nxt.strip()) >= len(line.strip())
        ):
            char = nxt.strip()[0]
            level = levels.setdefault(char, len(levels) + 1)
            out.append(f"{'#' * min(level, 6)} {line.strip()}")
            i += 2
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _first_heading(text: str) -> str | None:
    match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


class DirectoryLoader:
    """Recursively loads documentation files from a directory."""

    def __init__(self, root: Path, base_url: str | None = None) -> None:
        self.root = root.resolve()
        self.base_url = base_url.rstrip("/") if base_url else None

    def _url_for(self, rel: Path) -> str | None:
        if not self.base_url:
            return None
        slug = rel.with_suffix("").as_posix()
        if slug.endswith("index"):
            slug = slug[: -len("index")].rstrip("/")
        return f"{self.base_url}/{slug}" if slug else self.base_url

    def load_file(self, path: Path) -> SourceDocument | None:
        fmt = SUFFIX_FORMATS.get(path.suffix.lower())
        if fmt is None:
            return None
        raw = path.read_text(encoding="utf-8", errors="replace")
        rel = path.resolve().relative_to(self.root)
        meta: dict = {}
        title: str | None = None

        if fmt == "markdown":
            post = frontmatter.loads(raw)
            text, meta = post.content, dict(post.metadata)
            title = meta.get("title") or _first_heading(text)
        elif fmt == "html":
            text, title = html_to_markdownish(raw)
        elif fmt == "rst":
            text = rst_to_markdownish(raw)
            title = _first_heading(text)
        else:
            text = raw

        if not text.strip():
            return None

        source = rel.as_posix()
        return SourceDocument(
            doc_id=make_doc_id(source),
            text=text,
            source=source,
            url=meta.get("url") or self._url_for(rel),
            title=title or path.stem.replace("-", " ").replace("_", " ").title(),
            format=fmt,  # type: ignore[arg-type]
            metadata=meta,
            last_modified=datetime.fromtimestamp(path.stat().st_mtime, tz=UTC),
        )

    def __iter__(self) -> Iterator[SourceDocument]:
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or any(part.startswith(".") for part in path.parts):
                continue
            try:
                doc = self.load_file(path)
            except Exception:  # noqa: BLE001 - one bad file must not stop a 10k-page ingest
                logger.exception("Failed to load %s", path)
                continue
            if doc is not None:
                yield doc


class SitemapLoader:
    """Fetches every page listed in a sitemap.xml concurrently."""

    def __init__(
        self,
        sitemap_url: str,
        *,
        include: str | None = None,
        concurrency: int = 8,
        timeout: float = 20.0,
    ) -> None:
        self.sitemap_url = sitemap_url
        self.include = re.compile(include) if include else None
        self.concurrency = concurrency
        self.timeout = timeout

    async def _urls(self, client: httpx.AsyncClient) -> list[str]:
        resp = await client.get(self.sitemap_url)
        resp.raise_for_status()
        tree = ElementTree.fromstring(resp.content)
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        urls = [loc.text.strip() for loc in tree.findall(".//sm:loc", ns) if loc.text]
        if self.include:
            urls = [u for u in urls if self.include.search(u)]
        return urls

    async def aiter(self) -> AsyncIterator[SourceDocument]:
        sem = asyncio.Semaphore(self.concurrency)
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            urls = await self._urls(client)
            logger.info("Sitemap lists %d pages", len(urls))

            async def fetch(url: str) -> SourceDocument | None:
                async with sem:
                    try:
                        resp = await client.get(url)
                        resp.raise_for_status()
                    except httpx.HTTPError as exc:
                        logger.warning("Skipping %s: %s", url, exc)
                        return None
                text, title = html_to_markdownish(resp.text)
                if not text:
                    return None
                return SourceDocument(
                    doc_id=make_doc_id(url),
                    text=text,
                    source=url,
                    url=url,
                    title=title,
                    format="html",
                )

            for coro in asyncio.as_completed([fetch(u) for u in urls]):
                doc = await coro
                if doc is not None:
                    yield doc
