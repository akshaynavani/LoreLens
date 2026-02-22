"""Logging setup shared by the CLI, API and MCP server."""

from __future__ import annotations

import logging
import sys

from rich.logging import RichHandler

_CONFIGURED = False


def configure_logging(level: str = "INFO", *, stderr_only: bool = False) -> None:
    """Configure root logging once.

    The MCP stdio server must never write to stdout (it carries the protocol), so
    `stderr_only=True` routes everything through a plain stderr handler.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler: logging.Handler
    if stderr_only:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    else:
        handler = RichHandler(rich_tracebacks=True, show_path=False)

    logging.basicConfig(level=level.upper(), handlers=[handler], format="%(message)s")
    for noisy in ("httpx", "httpcore", "urllib3", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
