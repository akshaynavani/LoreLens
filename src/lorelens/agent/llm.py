"""Chat model factory.

Two tiers: a fast/cheap model for query analysis, grading and rewriting (latency-critical,
structured output) and a stronger model for the final cited answer.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from lorelens.config import get_settings


@lru_cache(maxsize=4)
def get_chat_model(tier: str = "answer") -> BaseChatModel:
    settings = get_settings()
    if settings.openai_api_key is None:
        raise RuntimeError("OPENAI_API_KEY is required")
    model = settings.fast_model if tier == "fast" else settings.chat_model
    return ChatOpenAI(
        model=model,
        api_key=settings.openai_api_key,
        temperature=0,
        timeout=60,
        max_retries=2,
        streaming=tier == "answer",
    )
