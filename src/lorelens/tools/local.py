"""In-process LangChain tools backed directly by DocSearchService."""

from __future__ import annotations

import asyncio

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from lorelens.retrieval.service import DocSearchService
from lorelens.tools import core


class SearchDocsInput(BaseModel):
    query: str = Field(description="Natural-language or keyword search query.")
    product: str | None = Field(default=None, description="Product name filter, e.g. 'nimbus'.")
    version: str | None = Field(default=None, description="Version filter, e.g. 'v2'.")
    doc_type: str | None = Field(
        default=None,
        description="One of api_reference, guide, tutorial, release_notes, faq, concept.",
    )
    top_k: int = Field(default=6, ge=1, le=15, description="Number of chunks to return.")


class ReadPageInput(BaseModel):
    doc_id: str = Field(description="doc_id from a search_docs result.")


def build_local_tools(service: DocSearchService) -> list[BaseTool]:
    def _search(query: str, product: str | None = None, version: str | None = None,
                doc_type: str | None = None, top_k: int = 6) -> str:
        return core.search_docs(service, query, product, version, doc_type, top_k)

    async def _asearch(query: str, product: str | None = None, version: str | None = None,
                       doc_type: str | None = None, top_k: int = 6) -> str:
        return await asyncio.to_thread(_search, query, product, version, doc_type, top_k)

    def _read(doc_id: str) -> str:
        return core.read_page(service, doc_id)

    async def _aread(doc_id: str) -> str:
        return await asyncio.to_thread(_read, doc_id)

    def _list() -> str:
        return core.list_products(service)

    async def _alist() -> str:
        return _list()

    return [
        StructuredTool.from_function(
            func=_search, coroutine=_asearch, name="search_docs",
            description=core.SEARCH_DOCS_DESCRIPTION, args_schema=SearchDocsInput,
        ),
        StructuredTool.from_function(
            func=_read, coroutine=_aread, name="read_page",
            description=core.READ_PAGE_DESCRIPTION, args_schema=ReadPageInput,
        ),
        StructuredTool.from_function(
            func=_list, coroutine=_alist, name="list_products",
            description=core.LIST_PRODUCTS_DESCRIPTION,
        ),
    ]
