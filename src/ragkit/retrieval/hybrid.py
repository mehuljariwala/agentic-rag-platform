"""Hybrid retrieval via Reciprocal Rank Fusion.

Dense and lexical retrievers produce scores on incompatible scales -- a cosine
similarity of 0.82 and a BM25 score of 11.4 cannot be added or averaged in any
principled way, and min-max normalising them makes fusion sensitive to whatever
happens to be in the candidate set. RRF sidesteps this entirely by fusing on
*rank* instead of score:

    RRF(d) = sum over retrievers of  weight / (k + rank(d))

k=60 is the value from Cormack et al. (2009); it damps the influence of the
very top rank enough that a single retriever cannot dominate the fused list.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, Sequence[tuple[str, float]]],
    weights: Mapping[str, float] | None = None,
    k: int = RRF_K,
    top_k: int = 10,
) -> list[tuple[str, float, dict[str, float]]]:
    """Fuse several ranked lists into one.

    Args:
        ranked_lists: retriever name -> [(doc_id, score), ...] in rank order.
        weights: optional per-retriever weight, defaults to 1.0 each.
        k: RRF damping constant.
        top_k: how many fused results to return.

    Returns:
        [(doc_id, fused_score, per_retriever_raw_scores)] sorted by fused score.
    """
    weights = weights or {}
    fused: dict[str, float] = {}
    breakdown: dict[str, dict[str, float]] = {}

    for name, results in ranked_lists.items():
        weight = weights.get(name, 1.0)
        for rank, (doc_id, raw) in enumerate(results, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + rank)
            breakdown.setdefault(doc_id, {})[name] = raw

    ordered = sorted(fused.items(), key=lambda x: -x[1])
    return [(doc_id, score, breakdown[doc_id]) for doc_id, score in ordered[:top_k]]
