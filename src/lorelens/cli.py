"""`lorelens` command-line interface."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console

from lorelens import __version__
from lorelens.config import get_settings
from lorelens.log import configure_logging

app = typer.Typer(help="LoreLens: agentic RAG search for technical documentation.")
console = Console()


@app.callback()
def main() -> None:
    configure_logging(get_settings().log_level)


@app.command()
def version() -> None:
    """Print the LoreLens version."""
    console.print(f"lorelens {__version__}")


@app.command()
def config() -> None:
    """Show the effective (non-secret) configuration."""
    settings = get_settings()
    redacted = settings.model_dump(
        exclude={"openai_api_key", "pinecone_api_key", "langfuse_secret_key", "api_key"}
    )
    for key, value in sorted(redacted.items()):
        console.print(f"[bold]{key}[/bold] = {value}")


@app.command()
def ingest(
    path: Path | None = typer.Argument(None, help="Directory of docs to ingest."),
    sitemap: str | None = typer.Option(None, help="Sitemap URL to crawl instead of a directory."),
    include: str | None = typer.Option(None, help="Regex filter for sitemap URLs."),
    base_url: str | None = typer.Option(None, help="Public base URL used to build citations."),
    namespace: str | None = typer.Option(None, help="Pinecone namespace override."),
    force: bool = typer.Option(False, help="Re-embed even unchanged pages."),
    prune: bool = typer.Option(False, help="Delete pages no longer present in the source."),
) -> None:
    """Load, chunk, embed and upsert documentation into Pinecone."""
    from lorelens.embeddings import get_embeddings
    from lorelens.ingestion.loaders import DirectoryLoader, SitemapLoader
    from lorelens.ingestion.pipeline import IngestionPipeline
    from lorelens.vectorstore import PineconeStore

    settings = get_settings()
    if (path is None) == (sitemap is None):
        raise typer.BadParameter("Provide exactly one of PATH or --sitemap")

    if path is not None:
        documents = iter(DirectoryLoader(path, base_url or settings.docs_base_url))
    else:
        async def _collect() -> list:
            return [d async for d in SitemapLoader(sitemap, include=include).aiter()]

        documents = iter(asyncio.run(_collect()))

    pipeline = IngestionPipeline(PineconeStore(settings), get_embeddings(), settings, namespace=namespace)
    stats = pipeline.run(documents, force=force, prune=prune)
    console.print_json(json.dumps(stats.as_dict()))


@app.command()
def stats() -> None:
    """Show Pinecone index statistics."""
    from lorelens.vectorstore import PineconeStore

    console.print_json(json.dumps(PineconeStore().stats()))


if __name__ == "__main__":
    app()
