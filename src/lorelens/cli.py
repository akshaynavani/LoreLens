"""`lorelens` command-line interface."""

from __future__ import annotations

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


if __name__ == "__main__":
    app()
