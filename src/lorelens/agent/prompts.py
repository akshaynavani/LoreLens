"""Prompt templates (local defaults).

Templates use LangFuse's `{{variable}}` syntax so the same text can be pushed to LangFuse
prompt management and iterated there; the registry falls back to these defaults.
"""

from __future__ import annotations

import re

ANALYZE_PROMPT = """You analyze questions for a technical documentation search assistant.

Documented products and versions (product -> version -> page count):
{{catalog}}

Conversation so far (may be empty):
{{history}}

Tasks:
1. Rewrite the latest user question into a standalone question (resolve pronouns using the conversation).
2. Decide if it is in scope: anything about using, configuring, integrating, troubleshooting or understanding the documented products is in scope. Greetings and unrelated topics are out of scope.
3. Extract metadata filters ONLY when the user states or clearly implies them: product (must be one of the listed products), version (e.g. "v2"), doc_type (api_reference, guide, tutorial, release_notes, faq, concept).
4. Classify complexity: "simple" if one search should answer it, "complex" for comparisons, multi-step procedures, or questions spanning several features/versions.
5. Produce 1-4 focused, keyword-rich search queries that together cover the question.

Latest user question: {{question}}"""

RESEARCH_PROMPT = """You are the research step of a documentation assistant. Your job is to gather evidence, not to answer.

Question: {{question}}
Suggested filters: {{filters}}

Evidence gathered so far:
{{evidence}}

Use the tools to fill gaps: run additional search_docs queries with different wording or filters, read_page to see a full page when a chunk is cut off, list_products if the product/version is unclear. Do not repeat a query that was already run.
When the evidence is sufficient (or nothing more can be found), reply with the single word DONE."""

GRADE_PROMPT = """You grade retrieved documentation chunks for relevance to a question.

Question: {{question}}

Chunks:
{{documents}}

Return the numbers of the chunks that contain information useful for answering the question, whether together they are sufficient to answer it fully and correctly, and if not, what specific information is still missing."""

REWRITE_PROMPT = """The search for the question below did not return sufficient evidence.

Question: {{question}}
Missing information: {{missing}}
Queries already tried:
{{tried}}

Write 1-3 NEW search queries that are likely to find the missing information. Use documentation vocabulary (feature names, API terms, error codes, config keys) and vary wording from the queries already tried."""

ANSWER_PROMPT = """You are LoreLens, an assistant that answers questions about technical documentation.

Answer ONLY from the numbered sources below. Rules:
- Cite every factual sentence with the source number in square brackets, e.g. [2] or [1][3].
- Prefer the version the user asked about; if sources disagree across versions, say so and name the versions.
- Include exact commands, parameters, limits and code from the sources when relevant, in fenced code blocks.
- If the sources do not contain the answer, say clearly that the documentation does not cover it and suggest what to search for. Never invent APIs, flags or numbers.
- Be concise: lead with the direct answer, then details.

Sources:
{{sources}}

Conversation so far (may be empty):
{{history}}

Question: {{question}}"""

OUT_OF_SCOPE_ANSWER = (
    "I can only help with questions about the indexed technical documentation. "
    "Try asking about a product feature, API, configuration option or error message."
)

DEFAULT_PROMPTS: dict[str, str] = {
    "lorelens-analyze": ANALYZE_PROMPT,
    "lorelens-research": RESEARCH_PROMPT,
    "lorelens-grade": GRADE_PROMPT,
    "lorelens-rewrite": REWRITE_PROMPT,
    "lorelens-answer": ANSWER_PROMPT,
}

_VAR_RE = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def compile_template(template: str, **variables: object) -> str:
    """Mustache-style `{{var}}` substitution matching LangFuse's `prompt.compile`."""
    return _VAR_RE.sub(lambda m: str(variables.get(m.group(1), m.group(0))), template)
