"""LangFuse integration: tracing, prompt management and scores.

Everything degrades gracefully: without LANGFUSE_* keys, tracing is a no-op and prompts
come from the local defaults in `lorelens.agent.prompts`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from langchain_core.runnables import RunnableConfig

from lorelens.agent.prompts import (
    DEFAULT_PROMPTS,
    LocalPromptProvider,
    PromptProvider,
    RenderedPrompt,
)
from lorelens.config import Settings, get_settings
from lorelens.log import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_langfuse():  # noqa: ANN201 - langfuse.Langfuse | None
    settings = get_settings()
    if not settings.langfuse_enabled:
        return None
    from langfuse import Langfuse

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key.get_secret_value(),  # type: ignore[union-attr]
        host=settings.langfuse_host,
    )


def langchain_handler():  # noqa: ANN201 - CallbackHandler | None
    """LangChain/LangGraph callback that turns every node, LLM and tool call into spans."""
    if get_langfuse() is None:
        return None
    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


# ----------------------------------------------------------------------- tracing
@dataclass
class TraceContext:
    trace_id: str | None = None
    config: RunnableConfig = field(default_factory=dict)  # type: ignore[assignment]
    _span: Any = None

    def set_output(self, output: Any, **metadata: Any) -> None:
        if self._span is not None:
            self._span.update_trace(output=output, metadata=metadata or None)


@asynccontextmanager
async def traced_run(
    name: str,
    *,
    input: Any = None,  # noqa: A002
    user_id: str | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> AsyncIterator[TraceContext]:
    """Open a LangFuse trace and hand back a RunnableConfig whose callbacks nest every
    LangGraph step under it. The trace id is returned to API clients for feedback."""
    client = get_langfuse()
    if client is None:
        yield TraceContext()
        return

    with client.start_as_current_span(name=name, input=input) as span:
        span.update_trace(
            name=name,
            user_id=user_id,
            session_id=session_id,
            tags=tags,
            metadata=metadata,
            input=input,
        )
        handler = langchain_handler()
        config: RunnableConfig = {"callbacks": [handler]} if handler else {}  # type: ignore[assignment]
        ctx = TraceContext(trace_id=span.trace_id, config=config, _span=span)
        try:
            yield ctx
        except Exception as exc:
            span.update(level="ERROR", status_message=str(exc))
            raise


def score_trace(
    trace_id: str,
    name: str,
    value: float | str,
    *,
    comment: str | None = None,
    data_type: str | None = None,
) -> bool:
    client = get_langfuse()
    if client is None:
        return False
    client.create_score(
        trace_id=trace_id, name=name, value=value, comment=comment, data_type=data_type
    )
    return True


def flush() -> None:
    client = get_langfuse()
    if client is not None:
        client.flush()


# --------------------------------------------------------------------- prompts
class LangfusePromptProvider:
    """Fetch prompts by name + label from LangFuse (cached client-side), falling back to the
    bundled defaults. Generations are linked to the prompt version that produced them."""

    def __init__(self, client: Any, settings: Settings, cache_ttl_seconds: int = 60) -> None:
        self.client = client
        self.label = settings.prompt_label
        self.cache_ttl_seconds = cache_ttl_seconds
        self._local = LocalPromptProvider()

    def render(self, name: str, **variables: object) -> RenderedPrompt:
        try:
            prompt = self.client.get_prompt(
                name,
                label=self.label,
                type="text",
                cache_ttl_seconds=self.cache_ttl_seconds,
                fallback=DEFAULT_PROMPTS[name],
            )
        except Exception:  # noqa: BLE001 - never fail a user request on prompt fetch
            logger.warning("Prompt %s unavailable in LangFuse; using local default", name)
            return self._local.render(name, **variables)
        text = prompt.compile(**{k: str(v) for k, v in variables.items()})
        metadata = {} if getattr(prompt, "is_fallback", False) else {"langfuse_prompt": prompt}
        return RenderedPrompt(text=text, metadata=metadata)


def get_prompt_provider() -> PromptProvider:
    client = get_langfuse()
    if client is None:
        return LocalPromptProvider()
    return LangfusePromptProvider(client, get_settings())


def push_default_prompts(labels: list[str] | None = None) -> list[str]:
    """Create (a new version of) every local prompt in LangFuse."""
    client = get_langfuse()
    if client is None:
        raise RuntimeError("LangFuse is not configured (LANGFUSE_PUBLIC_KEY / SECRET_KEY)")
    labels = labels or [get_settings().prompt_label]
    for name, text in DEFAULT_PROMPTS.items():
        client.create_prompt(name=name, prompt=text, labels=labels, type="text")
    client.flush()
    return list(DEFAULT_PROMPTS)
