"""Second-stage reranking.

Fusion gives a good candidate set but a mediocre ordering: it only knows about
ranks, never about whether a chunk actually answers the question. Reranking a
shortlist is where most of the precision@3 improvement in this project comes
from (see evals/README for the measured delta).

Two strategies ship here:

* `LexicalReranker` -- zero-cost, no model call. Coverage of query terms
  weighted by their rarity, plus a proximity bonus. Used as the default so the
  pipeline is fully offline.
* `LLMReranker` -- asks the model to score each candidate 0-10. Better, but
  costs one call per batch.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from ..providers.base import LLM, Message
from ..retrieval.bm25 import tokenize
from ..types import Hit


class LexicalReranker:
    name = "lexical"

    def rerank(self, query: str, hits: Sequence[Hit], top_k: int = 5) -> list[Hit]:
        q_terms = tokenize(query)
        if not q_terms:
            return list(hits)[:top_k]

        # Rarity across the shortlist: a term in every candidate discriminates
        # nothing, so weight it near zero.
        doc_freq: Counter[str] = Counter()
        tokenized = [set(tokenize(h.chunk.text)) for h in hits]
        for toks in tokenized:
            for term in set(q_terms) & toks:
                doc_freq[term] += 1

        n = len(hits)
        scored: list[tuple[float, Hit]] = []
        # strict=True: `tokenized` is built from `hits`, so a length mismatch
        # would be a bug worth surfacing rather than silently truncating.
        for hit, toks in zip(hits, tokenized, strict=True):
            coverage = 0.0
            for term in set(q_terms):
                if term in toks:
                    coverage += math.log(1.0 + n / (1 + doc_freq[term]))
            proximity = self._proximity(q_terms, hit.chunk.text)
            score = coverage + 0.5 * proximity
            enriched = Hit(
                chunk=hit.chunk,
                score=score,
                scores={**hit.scores, "rerank": score},
                rank=hit.rank,
            )
            scored.append((score, enriched))

        scored.sort(key=lambda x: -x[0])
        out = [h for _, h in scored[:top_k]]
        for i, hit in enumerate(out, start=1):
            hit.rank = i
        return out

    @staticmethod
    def _proximity(q_terms: Sequence[str], text: str) -> float:
        """Reward candidates where query terms appear close together."""
        toks = tokenize(text)
        positions = {t: i for i, t in enumerate(toks) if t in set(q_terms)}
        if len(positions) < 2:
            return 0.0
        span = max(positions.values()) - min(positions.values())
        return len(positions) / (1.0 + span / max(1, len(positions)))


class LLMReranker:
    name = "llm"

    _PROMPT = (
        "Rate how well each passage answers the question, 0-10.\n"
        "Reply with one line per passage in the form `<index>: <score>`. "
        "No other text.\n\n"
        "Question: {query}\n\n{passages}"
    )

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def rerank(self, query: str, hits: Sequence[Hit], top_k: int = 5) -> list[Hit]:
        if not hits:
            return []

        passages = "\n\n".join(f"[{i}] {h.chunk.text[:600]}" for i, h in enumerate(hits))
        prompt = self._PROMPT.format(query=query, passages=passages)
        reply = self.llm.complete(
            [Message("user", prompt)], temperature=0.0, max_tokens=256
        ).text

        scores: dict[int, float] = {}
        for line in reply.splitlines():
            m = re.match(r"\s*\[?(\d+)\]?\s*[:\-]\s*([0-9.]+)", line)
            if m:
                scores[int(m.group(1))] = float(m.group(2))

        # A model that returns nothing parseable must not silently destroy the
        # ordering -- fall back to the incoming fused rank.
        if not scores:
            return list(hits)[:top_k]

        ordered = sorted(enumerate(hits), key=lambda pair: -scores.get(pair[0], 0.0))
        out: list[Hit] = []
        for rank, (idx, hit) in enumerate(ordered[:top_k], start=1):
            score = scores.get(idx, 0.0)
            out.append(
                Hit(
                    chunk=hit.chunk,
                    score=score,
                    scores={**hit.scores, "rerank": score},
                    rank=rank,
                )
            )
        return out
