"""Deterministic offline provider.

The point of this module is that `pytest`, the eval harness and `docker compose
up` all work on a laptop with no network and no API keys, and produce the *same*
answer every run. Embeddings are a hashed bag-of-words projection -- crude, but
genuinely semantic enough that retrieval tests are meaningful rather than
tautological (documents sharing vocabulary really do score closer together).
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterator, Sequence

from .base import LLM, Completion, Embedder, Message, Usage

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _approx_tokens(text: str) -> int:
    """Rough token count. Good enough for accounting in tests."""
    return max(1, len(text) // 4)


def _trigrams(token: str) -> list[str]:
    """Character trigrams of a word, with boundary markers.

    `<vacuum>` -> `<va`, `vac`, `acu`, ... Boundaries let prefixes and suffixes
    carry signal, which is what makes `vacuum` and `vacuuming` land near each
    other.
    """
    padded = f"<{token}>"
    if len(padded) <= 3:
        return [padded]
    return [padded[i : i + 3] for i in range(len(padded) - 2)]


class MockEmbedder(Embedder):
    """Hashed subword embedding, L2-normalised.

    Tokens *and* their character trigrams are hashed into `dim` buckets with a
    signed contribution and sublinear term-frequency weighting.

    The trigrams are the important part. Without them this would be a
    reimplementation of lexical matching, and fusing it with BM25 would be
    tautological -- both arms would fail on exactly the same queries. Subword
    hashing makes the dense arm robust to morphology (`vacuum` / `vacuuming`,
    `index` / `indexes`) where BM25 sees unrelated terms, so the two retrievers
    have genuinely different failure modes and fusion has something to do.

    It is still not a semantic model: it cannot connect `bloat` to `dead
    tuples`. Swap in `RAGKIT_PROVIDER=ollama` for that.
    """

    name = "mock-embed"

    #: Trigrams are numerous, so damp them relative to whole tokens.
    TRIGRAM_WEIGHT = 0.35

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % self.dim
        sign = 1.0 if digest[4] & 1 else -1.0
        return bucket, sign

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim

        counts: dict[str, int] = {}
        for tok in _tokens(text):
            counts[tok] = counts.get(tok, 0) + 1

        for tok, count in counts.items():
            weight = 1.0 + math.log(count)  # sublinear tf
            bucket, sign = self._bucket(tok)
            vec[bucket] += sign * weight

            for gram in _trigrams(tok):
                g_bucket, g_sign = self._bucket(gram)
                vec[g_bucket] += g_sign * weight * self.TRIGRAM_WEIGHT

        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]


class MockLLM(LLM):
    """Extractive stand-in for a real chat model.

    It does not hallucinate because it does not generate: it selects the
    sentences from the supplied context that best overlap the question, and
    emits them with citation markers in the same format the real providers are
    prompted to use. That keeps citation-parsing and grading logic honest.
    """

    name = "mock-llm"

    def __init__(self, max_sentences: int = 3) -> None:
        self.max_sentences = max_sentences

    def _answer(self, messages: Sequence[Message]) -> str:
        question = next((m.content for m in reversed(messages) if m.role == "user"), "")
        context = "\n".join(m.content for m in messages if m.role == "system")

        # The retrieval prompt renders chunks as "[n] text"; recover them.
        chunks = re.findall(r"^\[(\d+)\]\s*(.+)$", context, re.MULTILINE)
        if not chunks:
            return "I don't have enough context to answer that."

        q_terms = set(_tokens(question))
        if not q_terms:
            return "I don't have enough context to answer that."

        scored: list[tuple[float, str, str]] = []
        for cid, body in chunks:
            for sentence in re.split(r"(?<=[.!?])\s+", body):
                s_terms = set(_tokens(sentence))
                if not s_terms:
                    continue
                overlap = len(q_terms & s_terms) / math.sqrt(len(s_terms))
                if overlap > 0:
                    scored.append((overlap, sentence.strip(), cid))

        if not scored:
            return "I don't have enough context to answer that."

        scored.sort(key=lambda x: -x[0])
        picked = scored[: self.max_sentences]
        return " ".join(f"{text} [{cid}]" for _, text, cid in picked)

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Completion:
        text = self._answer(messages)
        prompt_tokens = sum(_approx_tokens(m.content) for m in messages)
        return Completion(
            text=text,
            usage=Usage(prompt_tokens, _approx_tokens(text)),
            model=self.name,
        )

    def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        for word in self.complete(
            messages, temperature=temperature, max_tokens=max_tokens
        ).text.split(" "):
            yield word + " "
