"""LLM-as-judge for answer faithfulness (grounded in sources) and correctness (vs reference)."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from lorelens.models import RetrievedChunk

JUDGE_PROMPT = """You are grading an answer produced by a documentation assistant.

Question:
{question}

Reference answer (ground truth):
{reference}

Sources the assistant was given:
{sources}

Assistant answer:
{answer}

Score each criterion from 0.0 to 1.0:
- faithfulness: every factual claim in the answer is supported by the sources (1.0 = fully supported, 0.0 = mostly unsupported/hallucinated). An answer that correctly says the docs do not cover the question is faithful.
- correctness: the answer agrees with the reference answer and contains its key facts (numbers, commands, names). Extra correct detail is fine; contradictions are not.
- completeness: the answer addresses every part of the question.
Explain briefly."""


class JudgeScores(BaseModel):
    faithfulness: float = Field(ge=0, le=1)
    correctness: float = Field(ge=0, le=1)
    completeness: float = Field(ge=0, le=1)
    reasoning: str = ""


class AnswerJudge:
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm.with_structured_output(JudgeScores)

    async def score(
        self,
        question: str,
        reference: str,
        answer: str,
        sources: Sequence[RetrievedChunk],
    ) -> JudgeScores:
        src = "\n\n".join(
            f"[{i}] {c.title or c.doc_id}: {c.text[:1200]}" for i, c in enumerate(sources, start=1)
        ) or "(no sources)"
        prompt = JUDGE_PROMPT.format(
            question=question, reference=reference, sources=src, answer=answer
        )
        return await self.llm.ainvoke([HumanMessage(prompt)], {"run_name": "judge_answer"})
