"""Retrieval layer shared by local LangChain tools, the MCP server and the API."""

from lorelens.retrieval.service import DocSearchService, get_search_service

__all__ = ["DocSearchService", "get_search_service"]
