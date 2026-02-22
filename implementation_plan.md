# LoreLens — Implementation Plan

**LoreLens** is an agentic RAG search assistant for technical documentation. It ingests
large documentation sets (10k+ pages of Markdown / HTML / reStructuredText or crawled from a
sitemap), chunks and embeds them with a configurable pipeline (OpenAI or HuggingFace
embeddings), stores them in Pinecone with rich metadata, and answers natural-language
questions through a LangGraph agent that plans, calls retrieval tools (locally or over MCP),
grades evidence, rewrites queries when needed, and produces cited answers. LangFuse is used for
tracing, prompt management, user feedback scores, and offline evaluation runs.

## Stack

| Concern | Choice |
|---|---|
| Language | Python 3.11+ |
| LLM orchestration | LangChain (splitters, embeddings, chat models), LangGraph (agent graph) |
| Tool protocol | MCP (`mcp` FastMCP server + `langchain-mcp-adapters` client) |
| Vector DB | Pinecone serverless (namespaces + metadata filtering) |
| Models | OpenAI chat + embeddings; HuggingFace sentence-transformers embeddings + cross-encoder reranker |
| API | FastAPI (JSON + Server-Sent Events streaming) |
| Observability / evals | LangFuse (traces, prompts, scores, datasets) |
| CLI | Typer |

## Key technical decisions

1. **Embedding provider is pluggable** (`openai` | `huggingface`); index dimension is derived
   from the chosen model so the same pipeline works for both. Embeddings are cached on disk
   (LangChain `CacheBackedEmbeddings`) so re-ingests and repeated queries are cheap.
2. **Chunking strategies are pluggable** (`markdown` header-aware, `code_aware` that never splits
   fenced code blocks, and plain `token` windows). Each chunk carries breadcrumb headers so the
   section path is both embedded and filterable.
3. **Metadata is first-class**: product, version, doc type, section path, URL, title, content hash.
   The agent extracts filters from the question and passes them to Pinecone `filter`.
4. **Incremental ingestion** via a manifest of content hashes: unchanged pages are skipped, stale
   chunks are deleted.
5. **One retrieval core, two tool transports**: the same `DocSearchService` backs in-process
   LangChain tools and an MCP server. The agent can run with `TOOL_MODE=local` or `TOOL_MODE=mcp`.
6. **Agent graph** (LangGraph): `analyze → research ⇄ tools → collect → grade → (rewrite → research)* → generate`,
   with a short-circuit for out-of-scope questions and a bounded rewrite loop.
7. **Latency work**: query-embedding LRU cache, TTL retrieval cache, parallel sub-question
   retrieval, reranking a small candidate set, a cheap model for analysis/grading and a stronger
   model only for the final answer. Latency is measured per node in LangFuse and in the eval harness.
8. **Prompts live in LangFuse** (label `production`) with local fallbacks, so prompt iterations
   can be evaluated against a dataset without redeploying.

## Phases

### Phase 1 — Scaffolding & configuration
Deliverables: project layout, dependency manifest, typed settings, logging, env template, CLI skeleton.
- `pyproject.toml`, `.env.example`
- `src/lorelens/__init__.py`, `config.py`, `logging.py`, `cli.py`
- `src/lorelens/models.py` (shared Pydantic schemas: `Chunk`, `RetrievedChunk`, `Citation`, ...)

### Phase 2 — Ingestion pipeline
Deliverables: loaders, metadata extraction, chunking strategies, embeddings factory, Pinecone
store wrapper, incremental indexer, `lorelens ingest` command, sample docs.
- `src/lorelens/ingestion/loaders.py`, `metadata.py`, `chunking.py`, `pipeline.py`, `manifest.py`
- `src/lorelens/embeddings.py`, `src/lorelens/vectorstore.py`
- `data/sample_docs/**`

### Phase 3 — Retrieval, tools, MCP and the LangGraph agent
Deliverables: retrieval service with filters/rerank/caching, LangChain tools, MCP server and
client loader, agent state + nodes + graph, prompts, `lorelens ask` / `lorelens mcp` commands.
- `src/lorelens/retrieval/service.py`, `rerank.py`, `cache.py`
- `src/lorelens/tools/local.py`, `src/lorelens/mcp_server.py`, `src/lorelens/tools/mcp_client.py`
- `src/lorelens/agent/state.py`, `nodes.py`, `graph.py`, `prompts.py`, `llm.py`

### Phase 4 — FastAPI service & LangFuse observability
Deliverables: LangFuse tracing/prompt registry/feedback scoring, FastAPI app with ask, streaming
ask, search, ingest and feedback endpoints, optional API-key auth, Dockerfile.
- `src/lorelens/observability.py`, `src/lorelens/agent/prompts.py` (registry)
- `src/lorelens/api/app.py`, `schemas.py`, `deps.py`, `routes.py`
- `Dockerfile`, `docker-compose.yml`

### Phase 5 — Evaluation harness, tests & polish
Deliverables: eval dataset format + LangFuse dataset sync, retrieval metrics (hit-rate, MRR),
LLM-judge faithfulness/correctness, latency percentiles, `lorelens eval` + prompt sync commands,
unit tests with fakes (no network), Makefile.
- `src/lorelens/evals/dataset.py`, `metrics.py`, `judge.py`, `runner.py`
- `data/eval/sample_eval.jsonl`, `scripts/push_prompts.py`
- `tests/**`, `Makefile`
