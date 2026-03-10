"""Transport-agnostic tool implementations.

Both the in-process LangChain tools and the MCP server call these functions, and both
return the same JSON payloads, so the agent behaves identically in `local` and `mcp` mode.
"""

from __future__ import annotations

import json

from lorelens.models import SearchFilters
from lorelens.retrieval.service import DocSearchService

SEARCH_DOCS_DESCRIPTION = (
    "Semantic search over the technical documentation index. Returns the most relevant "
    "chunks with title, section, url, product, version and a chunk_id. Use specific, "
    "keyword-rich queries. Optionally filter by product, version (e.g. 'v2') and doc_type "
    "(api_reference, guide, tutorial, release_notes, faq, concept). Call several times with "
    "different queries for multi-part questions."
)
READ_PAGE_DESCRIPTION = (
    "Read the full text of a documentation page by doc_id (taken from a search result) when "
    "a chunk is relevant but incomplete, e.g. a procedure that continues past the chunk."
)
LIST_PRODUCTS_DESCRIPTION = (
    "List the documented products and their versions with page counts. Use it when the "
    "user's product or version is ambiguous."
)


def search_docs(
    service: DocSearchService,
    query: str,
    product: str | None = None,
    version: str | None = None,
    doc_type: str | None = None,
    top_k: int = 6,
) -> str:
    filters = SearchFilters(
        product=product.lower() if product else None,
        version=version.lower() if version else None,
        doc_type=doc_type or None,  # type: ignore[arg-type]
    )
    hits = service.search(query, filters, top_k=max(1, min(top_k, 15)))
    return json.dumps(
        {
            "query": query,
            "filters": filters.model_dump(exclude_none=True, exclude_defaults=True),
            "results": [
                {
                    "chunk_id": h.chunk_id,
                    "doc_id": h.doc_id,
                    "title": h.title,
                    "section": h.section,
                    "url": h.url,
                    "source": h.source,
                    "product": h.product,
                    "version": h.version,
                    "doc_type": h.doc_type,
                    "score": round(h.best_score, 4),
                    "text": h.text,
                }
                for h in hits
            ],
        }
    )


def read_page(service: DocSearchService, doc_id: str) -> str:
    return json.dumps(service.get_page(doc_id))


def list_products(service: DocSearchService) -> str:
    return json.dumps({"products": service.catalog()})
