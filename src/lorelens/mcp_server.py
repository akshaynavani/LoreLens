"""LoreLens MCP server.

Exposes the documentation index to any MCP client (the LoreLens agent, Claude Desktop,
IDEs, ...) with three tools: `search_docs`, `read_page`, `list_products`, plus a
`lorelens://catalog` resource.

Run:  python -m lorelens.mcp_server                 # stdio (default)
      python -m lorelens.mcp_server --transport streamable-http --port 8765
"""

from __future__ import annotations

import argparse
import asyncio

from mcp.server.fastmcp import FastMCP

from lorelens.config import get_settings
from lorelens.log import configure_logging
from lorelens.retrieval.service import get_search_service
from lorelens.tools import core


def build_server(host: str = "127.0.0.1", port: int = 8765) -> FastMCP:
    mcp = FastMCP(
        "lorelens",
        instructions=(
            "Search and read technical documentation. Start with search_docs; use read_page "
            "for full context; use list_products to discover products and versions."
        ),
        host=host,
        port=port,
    )

    @mcp.tool(name="search_docs", description=core.SEARCH_DOCS_DESCRIPTION)
    async def search_docs(
        query: str,
        product: str | None = None,
        version: str | None = None,
        doc_type: str | None = None,
        top_k: int = 6,
    ) -> str:
        return await asyncio.to_thread(
            core.search_docs, get_search_service(), query, product, version, doc_type, top_k
        )

    @mcp.tool(name="read_page", description=core.READ_PAGE_DESCRIPTION)
    async def read_page(doc_id: str) -> str:
        return await asyncio.to_thread(core.read_page, get_search_service(), doc_id)

    @mcp.tool(name="list_products", description=core.LIST_PRODUCTS_DESCRIPTION)
    async def list_products() -> str:
        return core.list_products(get_search_service())

    @mcp.resource("lorelens://catalog", mime_type="application/json")
    def catalog() -> str:
        return core.list_products(get_search_service())

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(description="LoreLens MCP server")
    parser.add_argument("--transport", choices=["stdio", "sse", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    # stdout carries the MCP protocol in stdio mode: log to stderr only.
    configure_logging(get_settings().log_level, stderr_only=True)
    build_server(args.host, args.port).run(transport=args.transport)


if __name__ == "__main__":
    main()
