"""Load LoreLens tools over MCP using langchain-mcp-adapters.

In `mcp` tool mode the agent never touches Pinecone directly: it discovers and calls the
tools exposed by the MCP server (spawned over stdio, or reached over streamable HTTP),
exactly like any third-party MCP client would.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from lorelens.config import Settings, get_settings
from lorelens.log import get_logger

logger = get_logger(__name__)


def mcp_connections(settings: Settings) -> dict[str, dict[str, Any]]:
    if settings.mcp_server_url:
        return {"lorelens": {"url": settings.mcp_server_url, "transport": "streamable_http"}}
    return {
        "lorelens": {
            "command": settings.mcp_server_command,
            "args": settings.mcp_server_args,
            "transport": "stdio",
            # The child process needs the same credentials/config as the parent.
            "env": dict(os.environ),
        }
    }


async def load_mcp_tools(settings: Settings | None = None) -> list[BaseTool]:
    settings = settings or get_settings()
    client = MultiServerMCPClient(mcp_connections(settings))
    tools = await client.get_tools()
    logger.info("Loaded %d MCP tools: %s", len(tools), [t.name for t in tools])
    return tools
