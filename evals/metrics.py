"""Retrieval metrics.

Implemented directly rather than pulled in, because the exact conventions
matter when comparing configurations and most libraries differ subtly in how
they handle ties, truncation and queries with more relevant documents than k.

All functions take `retrieved` in rank order and `relevant` as a set.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the relevant set that appears in the top k."""
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the top k that is relevant.

    The denominator is `min(k, len(retrieved))`, not k. Dividing by k would
    penalise a query that legitimately has fewer than k candidates, which makes
    small-corpus comparisons misleading.
    """
    if not retrieved:
        return 0.0
    window = retrieved[:k]
    return len([d for d in window if d in relevant]) / len(window)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant result; 0 if none was retrieved."""
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def dcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    return sum(
        1.0 / math.log2(i + 1)
        for i, doc in enumerate(retrieved[:k], start=1)
        if doc in relevant
    )


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """DCG normalised by the best achievable ordering.

    The ideal DCG uses `min(k, len(relevant))` positions -- a query with two
    relevant documents cannot score against a five-slot ideal, or nDCG@5 would
    be capped below 1.0 for reasons that have nothing to do with the retriever.
    """
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    if ideal == 0.0:
        return 0.0
    return dcg_at_k(retrieved, relevant, k) / ideal


def hit_rate_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant document made the top k. Blunt but readable."""
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def aggregate(
    per_query: Sequence[dict[str, float]],
) -> dict[str, float]:
    """Macro-average across queries (each query weighted equally)."""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: sum(q[k] for q in per_query) / len(per_query) for k in keys}
