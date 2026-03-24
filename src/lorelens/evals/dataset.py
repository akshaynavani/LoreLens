"""Evaluation dataset: JSONL on disk, mirrored to a LangFuse dataset."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from lorelens.models import SearchFilters


class EvalExample(BaseModel):
    id: str
    question: str
    expected_answer: str
    # Source paths / URLs of pages that contain the answer (as stored in chunk metadata).
    expected_sources: list[str] = Field(default_factory=list)
    filters: SearchFilters | None = None
    tags: list[str] = Field(default_factory=list)


def load_jsonl(path: Path) -> list[EvalExample]:
    examples = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            examples.append(EvalExample.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: invalid example: {exc}") from exc
    return examples


def sync_to_langfuse(client, name: str, examples: list[EvalExample]) -> int:  # noqa: ANN001
    """Create/refresh a LangFuse dataset so experiment runs can be compared in the UI."""
    client.create_dataset(name=name, description="LoreLens documentation QA evaluation set")
    for ex in examples:
        client.create_dataset_item(
            dataset_name=name,
            id=f"{name}-{ex.id}",
            input={
                "question": ex.question,
                "filters": ex.filters.model_dump(exclude_none=True) if ex.filters else None,
            },
            expected_output={"answer": ex.expected_answer, "sources": ex.expected_sources},
            metadata={"tags": ex.tags, "example_id": ex.id},
        )
    client.flush()
    return len(examples)


def example_from_item(item) -> EvalExample:  # noqa: ANN001 - langfuse DatasetItemClient
    inp = item.input or {}
    expected = item.expected_output or {}
    meta = item.metadata or {}
    return EvalExample(
        id=meta.get("example_id", item.id),
        question=inp["question"],
        expected_answer=expected.get("answer", ""),
        expected_sources=expected.get("sources", []),
        filters=SearchFilters(**inp["filters"]) if inp.get("filters") else None,
        tags=meta.get("tags", []),
    )


def dump_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str))
