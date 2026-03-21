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


@app.command()
def search(
    query: str,
    product: str | None = typer.Option(None),
    version: str | None = typer.Option(None),
    doc_type: str | None = typer.Option(None),
    top_k: int = typer.Option(5),
) -> None:
    """Run raw retrieval (no LLM) and print the matching chunks."""
    from lorelens.retrieval.service import get_search_service
    from lorelens.tools.core import search_docs

    payload = json.loads(search_docs(get_search_service(), query, product, version, doc_type, top_k))
    for i, r in enumerate(payload["results"], start=1):
        console.rule(f"[{i}] {r['title']} > {r.get('section') or ''}  score={r['score']}")
        console.print(f"[dim]{r.get('url') or r.get('source')}  {r.get('product')} {r.get('version')}[/dim]")
        console.print(r["text"][:600])


@app.command()
def ask(
    question: str,
    product: str | None = typer.Option(None, help="Restrict to a product."),
    version: str | None = typer.Option(None, help="Restrict to a version, e.g. v2."),
    mode: str | None = typer.Option(None, help="Tool transport override: local | mcp."),
    show_trace: bool = typer.Option(False, help="Print node timings and evidence."),
) -> None:
    """Ask the agent a question and print a cited answer."""
    from lorelens.agent.graph import LoreLensAgent
    from lorelens.models import SearchFilters
    from lorelens.observability import flush, traced_run

    settings = get_settings()
    if mode:
        settings = settings.model_copy(update={"tool_mode": mode})

    async def _run() -> None:
        agent = await LoreLensAgent.create(settings)
        async with traced_run(
            "lorelens-ask", input={"question": question}, tags=["cli", settings.tool_mode]
        ) as trace:
            answer, state = await agent.arun(
                question,
                filters=SearchFilters(product=product, version=version),
                config=trace.config,
            )
            trace.set_output(answer.answer, citations=len(answer.citations))
        flush()
        console.rule("Answer")
        console.print(answer.answer)
        if answer.citations:
            console.rule("Sources")
            for c in answer.citations:
                console.print(f"[{c.index}] {c.title} > {c.section or ''}  {c.url or c.source}")
        console.print(
            f"[dim]latency={answer.latency_ms}ms rewrites={answer.rewrites} "
            f"tool_calls={answer.tool_calls} grounded={answer.grounded}[/dim]"
        )
        if trace.trace_id:
            console.print(f"[dim]langfuse trace: {trace.trace_id}[/dim]")
        if show_trace:
            console.rule("Trace")
            console.print_json(json.dumps(state.get("node_timings", {})))
            console.print(f"queries: {state.get('tried_queries')}")
            console.print(f"filters: {state.get('filters')}")

    asyncio.run(_run())


@app.command("push-prompts")
def push_prompts(label: list[str] = typer.Option(None, help="Labels to attach.")) -> None:
    """Upload the bundled prompt templates to LangFuse prompt management."""
    from lorelens.observability import push_default_prompts

    for name in push_default_prompts(label or None):
        console.print(f"pushed {name}")


@app.command()
def mcp(
    transport: str = typer.Option("stdio", help="stdio | sse | streamable-http"),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8765),
) -> None:
    """Run the LoreLens MCP server."""
    from lorelens.mcp_server import build_server

    build_server(host, port).run(transport=transport)


if __name__ == "__main__":
    app()
