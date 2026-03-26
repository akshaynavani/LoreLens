"""Evaluation runner.

For each example: run the agent (traced), compute retrieval metrics against the expected
sources, ask the judge for faithfulness/correctness, record everything as LangFuse scores
on the example's trace (linked to a dataset run when a LangFuse dataset is used), and
aggregate a report with quality and latency percentiles so prompt / chunking / retrieval
changes can be compared run over run.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from lorelens.agent.graph import LoreLensAgent
from lorelens.evals import metrics
from lorelens.evals.dataset import EvalExample
from lorelens.evals.judge import AnswerJudge
from lorelens.log import get_logger
from lorelens.observability import TraceContext, get_langfuse, langchain_handler, traced_run

logger = get_logger(__name__)


@dataclass
class ExampleResult:
    id: str
    question: str
    answer: str = ""
    hit_rate_at_k: float = math.nan
    mrr: float = math.nan
    recall: float = math.nan
    citation_precision: float = math.nan
    faithfulness: float = math.nan
    correctness: float = math.nan
    completeness: float = math.nan
    grounded: bool = False
    rewrites: int = 0
    tool_calls: int = 0
    latency_ms: float = 0.0
    node_timings: dict[str, float] = field(default_factory=dict)
    trace_id: str | None = None
    error: str | None = None


@asynccontextmanager
async def _dataset_item_run(item: Any, run_name: str, metadata: dict) -> AsyncIterator[TraceContext]:
    """Trace an example inside a LangFuse dataset run (v3 `item.run`)."""
    with item.run(run_name=run_name, run_metadata=metadata) as span:
        handler = langchain_handler()
        yield TraceContext(
            trace_id=span.trace_id,
            config={"callbacks": [handler]} if handler else {},  # type: ignore[typeddict-item]
            _span=span,
        )


class EvalRunner:
    def __init__(
        self,
        agent: LoreLensAgent,
        judge: AnswerJudge | None,
        *,
        k: int = 5,
        concurrency: int = 4,
        run_name: str | None = None,
    ) -> None:
        self.agent = agent
        self.judge = judge
        self.k = k
        self.sem = asyncio.Semaphore(concurrency)
        self.run_name = run_name or f"eval-{datetime.now(UTC):%Y%m%d-%H%M%S}"

    def _context(self, example: EvalExample, item: Any | None):  # noqa: ANN202
        meta = {"example_id": example.id, "tool_mode": self.agent.settings.tool_mode}
        if item is not None:
            return _dataset_item_run(item, self.run_name, meta)
        return traced_run(
            "lorelens-eval",
            input={"question": example.question},
            tags=["eval", self.run_name],
            metadata=meta,
        )

    async def run_example(self, example: EvalExample, item: Any | None = None) -> ExampleResult:
        result = ExampleResult(id=example.id, question=example.question)
        async with self.sem, self._context(example, item) as trace:
            result.trace_id = trace.trace_id
            try:
                start = time.perf_counter()
                answer, state = await self.agent.arun(
                    example.question, filters=example.filters, config=trace.config
                )
                result.latency_ms = round((time.perf_counter() - start) * 1000, 1)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Example %s failed", example.id)
                result.error = str(exc)
                return result

            docs = state.get("documents", [])
            retrieved = [d.source or d.url for d in docs]
            cited = [c.source or c.url for c in answer.citations]
            result.answer = answer.answer
            result.grounded = answer.grounded
            result.rewrites = answer.rewrites
            result.tool_calls = answer.tool_calls
            result.node_timings = state.get("node_timings", {})
            result.hit_rate_at_k = metrics.hit_rate_at_k(retrieved, example.expected_sources, self.k)
            result.mrr = metrics.reciprocal_rank(retrieved, example.expected_sources)
            result.recall = metrics.recall(retrieved, example.expected_sources)
            result.citation_precision = metrics.citation_precision(cited, example.expected_sources)

            if self.judge is not None:
                try:
                    scores = await self.judge.score(
                        example.question, example.expected_answer, answer.answer, docs
                    )
                    result.faithfulness = scores.faithfulness
                    result.correctness = scores.correctness
                    result.completeness = scores.completeness
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Judge failed for %s: %s", example.id, exc)

            trace.set_output(answer.answer)
            self._log_scores(trace, result)
        return result

    @staticmethod
    def _log_scores(trace: TraceContext, r: ExampleResult) -> None:
        client = get_langfuse()
        if client is None or trace.trace_id is None:
            return
        for name in (
            "hit_rate_at_k", "mrr", "recall", "citation_precision",
            "faithfulness", "correctness", "completeness", "latency_ms",
        ):
            value = getattr(r, name)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            client.create_score(trace_id=trace.trace_id, name=name, value=float(value))

    async def run(
        self, examples: Sequence[EvalExample], items: Sequence[Any] | None = None
    ) -> dict[str, Any]:
        items = items or [None] * len(examples)
        results = await asyncio.gather(
            *(self.run_example(ex, it) for ex, it in zip(examples, items, strict=True))
        )
        client = get_langfuse()
        if client is not None:
            client.flush()
        return self.report(results)

    def report(self, results: Sequence[ExampleResult]) -> dict[str, Any]:
        ok = [r for r in results if r.error is None]
        latencies = [r.latency_ms for r in ok]
        node_totals: dict[str, list[float]] = {}
        for r in ok:
            for node, ms in r.node_timings.items():
                node_totals.setdefault(node, []).append(ms)
        summary = {
            "run_name": self.run_name,
            "examples": len(results),
            "errors": len(results) - len(ok),
            f"hit_rate@{self.k}": metrics.mean([r.hit_rate_at_k for r in ok]),
            "mrr": metrics.mean([r.mrr for r in ok]),
            "recall": metrics.mean([r.recall for r in ok]),
            "citation_precision": metrics.mean([r.citation_precision for r in ok]),
            "faithfulness": metrics.mean([r.faithfulness for r in ok]),
            "correctness": metrics.mean([r.correctness for r in ok]),
            "completeness": metrics.mean([r.completeness for r in ok]),
            "grounded_rate": metrics.mean([1.0 if r.grounded else 0.0 for r in ok]),
            "avg_rewrites": metrics.mean([float(r.rewrites) for r in ok]),
            "avg_tool_calls": metrics.mean([float(r.tool_calls) for r in ok]),
            "latency_ms": {
                "p50": metrics.percentile(latencies, 50),
                "p95": metrics.percentile(latencies, 95),
                "mean": metrics.mean(latencies),
            },
            "node_latency_ms_mean": {n: metrics.mean(v) for n, v in sorted(node_totals.items())},
        }
        return {"summary": summary, "results": [asdict(r) for r in results]}
