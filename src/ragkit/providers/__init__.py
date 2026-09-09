"""Provider registry.

`RAGKIT_PROVIDER` selects the backend for the whole app. It defaults to `mock`
so that a fresh clone runs green with no configuration at all.
"""

from __future__ import annotations

import os

from .base import LLM, Completion, Embedder, Message, ProviderError, Usage
from .mock import MockEmbedder, MockLLM

__all__ = [
    "Completion",
    "Embedder",
    "LLM",
    "Message",
    "ProviderError",
    "Usage",
    "MockLLM",
    "MockEmbedder",
    "get_llm",
    "get_embedder",
]

_LLM_CHOICES = ("mock", "anthropic", "openai", "ollama")
_EMBED_CHOICES = ("mock", "openai", "ollama")


def _selected(explicit: str | None) -> str:
    return (explicit or os.getenv("RAGKIT_PROVIDER") or "mock").lower()


def get_llm(provider: str | None = None, **kwargs) -> LLM:
    name = _selected(provider)
    if name == "mock":
        return MockLLM(**kwargs)
    if name == "anthropic":
        from .remote import AnthropicLLM

        return AnthropicLLM(**kwargs)
    if name == "openai":
        from .remote import OpenAILLM

        return OpenAILLM(**kwargs)
    if name == "ollama":
        from .remote import OllamaLLM

        return OllamaLLM(**kwargs)
    raise ProviderError(f"unknown LLM provider {name!r}; expected one of {_LLM_CHOICES}")


def get_embedder(provider: str | None = None, **kwargs) -> Embedder:
    name = _selected(provider)
    # Anthropic ships no embedding endpoint; fall back to the local embedder so
    # `RAGKIT_PROVIDER=anthropic` is still a usable end-to-end configuration.
    if name in ("mock", "anthropic"):
        return MockEmbedder(**kwargs)
    if name == "openai":
        from .remote import OpenAIEmbedder

        return OpenAIEmbedder(**kwargs)
    if name == "ollama":
        from .remote import OllamaEmbedder

        return OllamaEmbedder(**kwargs)
    raise ProviderError(
        f"unknown embedding provider {name!r}; expected one of {_EMBED_CHOICES}"
    )
