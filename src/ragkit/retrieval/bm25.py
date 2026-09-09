"""Okapi BM25, implemented directly.

Written out rather than pulled from a library because the ranking behaviour is
the thing being demonstrated here, and because it keeps the dependency list
short enough that the container image stays small.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Dropping these stops "what is the" style questions from matching every
# document equally and drowning the signal terms.
_STOPWORDS = frozenset(
    """
    a an and are as at be but by for from has have how i if in into is it its of
    on or that the their then there these they this to was were what when where
    which who why will with you your
    """.split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


@dataclass
class BM25:
    """Standard BM25 with the usual parameter defaults.

    k1 controls term-frequency saturation, b controls length normalisation.
    """

    k1: float = 1.5
    b: float = 0.75
    doc_ids: list[str] = field(default_factory=list)
    _tf: list[Counter[str]] = field(default_factory=list)
    _lengths: list[int] = field(default_factory=list)
    _df: Counter[str] = field(default_factory=Counter)
    _avg_len: float = 0.0

    def add(self, doc_id: str, text: str) -> None:
        toks = tokenize(text)
        tf = Counter(toks)
        self.doc_ids.append(doc_id)
        self._tf.append(tf)
        self._lengths.append(len(toks))
        for term in tf:
            self._df[term] += 1
        self._avg_len = sum(self._lengths) / len(self._lengths)

    def _idf(self, term: str) -> float:
        n = len(self.doc_ids)
        df = self._df.get(term, 0)
        if df == 0:
            return 0.0
        # Robertson/Sparck-Jones idf with +0.5 smoothing, floored at 0 so that
        # terms appearing in >half the corpus cannot contribute negatively.
        return max(0.0, math.log((n - df + 0.5) / (df + 0.5) + 1.0))

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        if not self.doc_ids:
            return []

        q_terms = tokenize(query)
        scores = [0.0] * len(self.doc_ids)

        for term in set(q_terms):
            idf = self._idf(term)
            if idf == 0.0:
                continue
            for i, tf in enumerate(self._tf):
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                norm = 1.0 - self.b + self.b * (self._lengths[i] / self._avg_len)
                scores[i] += idf * (freq * (self.k1 + 1.0)) / (freq + self.k1 * norm)

        ranked = sorted(
            ((self.doc_ids[i], s) for i, s in enumerate(scores) if s > 0.0),
            key=lambda x: -x[1],
        )
        return ranked[:k]
