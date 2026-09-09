"""Real providers: Anthropic, OpenAI and Ollama.

These are imported lazily so the package has no hard dependency on any vendor
SDK -- `pip install ragkit` gives you a working system via the mock provider,
and you only pay for an SDK if you actually select that backend.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence
from typing import Any

from .base import LLM, Completion, Embedder, Message, ProviderError, Usage

_TIMEOUT = float(os.getenv("RAGKIT_HTTP_TIMEOUT", "60"))


def _post(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"content-type": "application/json", **headers}
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:  # pragma: no cover - network path
        detail = exc.read().decode(errors="replace")[:500]
        raise ProviderError(f"{url} returned {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:  # pragma: no cover - network path
        raise ProviderError(f"could not reach {url}: {exc.reason}") from exc


def _split_system(messages: Sequence[Message]) -> tuple[str, list[dict[str, str]]]:
    """Anthropic takes `system` out-of-band rather than as a message role."""
    system = "\n\n".join(m.content for m in messages if m.role == "system")
    turns = [
        {"role": m.role, "content": m.content}
        for m in messages
        if m.role in ("user", "assistant")
    ]
    return system, turns


class AnthropicLLM(LLM):
    name = "anthropic"

    def __init__(
        self, model: str = "claude-sonnet-5", api_key: str | None = None
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set")

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Completion:
        system, turns = _split_system(messages)
        data = _post(
            "https://api.anthropic.com/v1/messages",
            {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system,
                "messages": turns,
            },
            self._headers(),
        )
        text = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        )
        usage = data.get("usage", {})
        return Completion(
            text=text,
            usage=Usage(usage.get("input_tokens", 0), usage.get("output_tokens", 0)),
            model=self.model,
        )

    def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        # Non-streaming fallback keeps the interface total; swap in the SSE
        # endpoint if you need true token-by-token delivery.
        yield self.complete(messages, temperature=temperature, max_tokens=max_tokens).text


class OpenAILLM(LLM):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ProviderError("OPENAI_API_KEY is not set")

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Completion:
        data = _post(
            "https://api.openai.com/v1/chat/completions",
            {
                "model": self.model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
            },
            {"authorization": f"Bearer {self.api_key}"},
        )
        usage = data.get("usage", {})
        return Completion(
            text=data["choices"][0]["message"]["content"],
            usage=Usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)),
            model=self.model,
        )

    def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        yield self.complete(messages, temperature=temperature, max_tokens=max_tokens).text


class OpenAIEmbedder(Embedder):
    name = "openai-embed"

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        dim: int = 1536,
    ) -> None:
        self.model = model
        self.dim = dim
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ProviderError("OPENAI_API_KEY is not set")

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        data = _post(
            "https://api.openai.com/v1/embeddings",
            {"model": self.model, "input": list(texts)},
            {"authorization": f"Bearer {self.api_key}"},
        )
        rows = sorted(data["data"], key=lambda r: r["index"])
        return [r["embedding"] for r in rows]


class OllamaLLM(LLM):
    """Local models via Ollama -- no key, no cost, runs offline."""

    name = "ollama"

    def __init__(self, model: str = "llama3.2", host: str | None = None) -> None:
        self.model = model
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip(
            "/"
        )

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Completion:
        data = _post(
            f"{self.host}/api/chat",
            {
                "model": self.model,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
                "messages": [{"role": m.role, "content": m.content} for m in messages],
            },
            {},
        )
        return Completion(
            text=data["message"]["content"],
            usage=Usage(data.get("prompt_eval_count", 0), data.get("eval_count", 0)),
            model=self.model,
        )

    def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        yield self.complete(messages, temperature=temperature, max_tokens=max_tokens).text


class OllamaEmbedder(Embedder):
    name = "ollama-embed"

    def __init__(
        self, model: str = "nomic-embed-text", host: str | None = None, dim: int = 768
    ) -> None:
        self.model = model
        self.dim = dim
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip(
            "/"
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            data = _post(
                f"{self.host}/api/embeddings",
                {"model": self.model, "prompt": text},
                {},
            )
            out.append(data["embedding"])
        return out
