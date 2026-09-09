"""Retrieval evaluation harness and CI regression gate.

    python -m evals.run_eval                 # default config
    python -m evals.run_eval --compare       # ablation across configs
    python -m evals.run_eval --by-style      # break results down by query type
    python -m evals.run_eval --fail-under 0.85 --metric recall@5

Retrieval is scored at document level: a hit counts if the chunk's parent
document is in the labelled relevant set. Chunk-level labels would be more
precise but they make the dataset brittle -- relabelling would be required
every time the chunker changes.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ragkit import Document, RAGConfig, RAGPipeline

from .metrics import (
    aggregate,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

DATASET_DIR = Path(__file__).parent / "datasets"
DEFAULT_DATASET = DATASET_DIR / "infra_qa.json"


@dataclass
class Config:
    label: str
    config: RAGConfig


# The ablation ladder. Each row isolates one mechanism so the contribution of
# fusion and reranking is attributable rather than asserted.
ABLATIONS: list[Config] = [
    Config(
        "bm25 only",
        RAGConfig(rewrites=0, grade=False, rerank=False, vector_weight=0.0),
    ),
    Config(
        "vector only",
        RAGConfig(rewrites=0, grade=False, rerank=False, bm25_weight=0.0),
    ),
    Config("hybrid (RRF, 1:1)", RAGConfig(rewrites=0, grade=False, rerank=False)),
    # Equal-weight RRF assumes both arms are comparably good. When one is
    # clearly weaker -- as the offline subword embedder is against BM25 on this
    # corpus -- equal weighting drags the fused list below the stronger arm on
    # its own. Down-weighting the weak arm is the fix, and the fact that this
    # is visible in the table at all is the argument for having the harness.
    Config(
        "hybrid (RRF, 3:1)",
        RAGConfig(rewrites=0, grade=False, rerank=False, bm25_weight=3.0),
    ),
    Config(
        "hybrid 3:1 + rerank",
        RAGConfig(rewrites=0, grade=False, rerank=True, bm25_weight=3.0),
    ),
]


def load_dataset(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return json.load(fh)


def build_pipeline(dataset: dict[str, Any], config: RAGConfig) -> RAGPipeline:
    rag = RAGPipeline(config=config)
    rag.index(
        [
            Document(id=d["id"], text=d["text"], metadata=d.get("metadata", {}))
            for d in dataset["documents"]
        ]
    )
    return rag


def evaluate(
    rag: RAGPipeline, dataset: dict[str, Any], k: int = 5
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    scores: list[dict[str, float]] = []

    for q in dataset["queries"]:
        hits = rag.retrieve(q["query"], top_k=k)

        # Collapse chunk hits to parent documents, preserving rank order.
        seen: set[str] = set()
        docs: list[str] = []
        for hit in hits:
            if hit.chunk.doc_id not in seen:
                seen.add(hit.chunk.doc_id)
                docs.append(hit.chunk.doc_id)

        relevant = set(q["relevant"])
        # precision@3 is deliberately absent from the headline set: with a
        # dataset where most queries have a single correct answer it is capped
        # at 0.333 for reasons that have nothing to do with retrieval quality,
        # which makes cross-config comparison meaningless. recall@1 measures
        # the same "is the top hit right" question without the artificial
        # ceiling, and nDCG handles the multi-answer queries properly.
        row_scores = {
            "recall@1": recall_at_k(docs, relevant, 1),
            f"recall@{k}": recall_at_k(docs, relevant, k),
            "mrr": reciprocal_rank(docs, relevant),
            f"ndcg@{k}": ndcg_at_k(docs, relevant, k),
            f"hit@{k}": hit_rate_at_k(docs, relevant, k),
            "p@k": precision_at_k(docs, relevant, k),
        }
        scores.append(row_scores)
        rows.append(
            {
                "id": q["id"],
                "query": q["query"],
                "style": q.get("style", "unknown"),
                "retrieved": docs,
                "relevant": q["relevant"],
                **row_scores,
            }
        )

    return aggregate(scores), rows


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("─" * len(line))
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


def report_single(dataset: dict[str, Any], k: int, by_style: bool) -> dict[str, float]:
    rag = build_pipeline(dataset, ABLATIONS[-1].config)
    summary, rows = evaluate(rag, dataset, k=k)

    print(
        f"\ndataset: {dataset['name']}  "
        f"({len(dataset['documents'])} docs, {len(dataset['queries'])} queries)"
    )
    print(f"config:  hybrid + rerank    provider: {rag.llm.name}/{rag.embedder.name}\n")

    metrics = list(summary)
    _print_table(["metric", "score"], [[m, f"{summary[m]:.3f}"] for m in metrics])

    if by_style:
        print()
        styles = sorted({r["style"] for r in rows})
        table = []
        for style in styles:
            subset = [r for r in rows if r["style"] == style]
            agg = aggregate([{m: r[m] for m in metrics} for r in subset])
            table.append([style, str(len(subset))] + [f"{agg[m]:.3f}" for m in metrics])
        _print_table(["style", "n", *metrics], table)

    failures = [r for r in rows if r["mrr"] == 0.0]
    if failures:
        print(f"\n{len(failures)} queries retrieved nothing relevant:")
        for r in failures:
            print(f"  {r['id']}  {r['query']!r}")
            print(f"       expected {r['relevant']}, got {r['retrieved'][:3]}")

    return summary


def report_compare(dataset: dict[str, Any], k: int) -> dict[str, float]:
    print(
        f"\ndataset: {dataset['name']}  "
        f"({len(dataset['documents'])} docs, {len(dataset['queries'])} queries)\n"
    )

    metrics: list[str] = []
    table: list[list[str]] = []
    last: dict[str, float] = {}

    for ablation in ABLATIONS:
        rag = build_pipeline(dataset, ablation.config)
        summary, _ = evaluate(rag, dataset, k=k)
        metrics = metrics or list(summary)
        table.append([ablation.label] + [f"{summary[m]:.3f}" for m in metrics])
        last = summary

    _print_table(["config", *metrics], table)
    return last


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ragkit retrieval evaluation")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("-k", type=int, default=5, help="retrieval depth")
    parser.add_argument("--compare", action="store_true", help="run the ablation ladder")
    parser.add_argument(
        "--by-style", action="store_true", help="break down by query style"
    )
    parser.add_argument("--fail-under", type=float, default=None)
    parser.add_argument("--metric", default="recall@5", help="metric for --fail-under")
    parser.add_argument("--json", type=Path, default=None, help="write summary as JSON")
    args = parser.parse_args(argv)

    if not args.dataset.exists():
        print(f"dataset not found: {args.dataset}", file=sys.stderr)
        return 2

    dataset = load_dataset(args.dataset)
    summary = (
        report_compare(dataset, args.k)
        if args.compare
        else report_single(dataset, args.k, args.by_style)
    )

    if args.json:
        args.json.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\nwrote {args.json}")

    if args.fail_under is not None:
        if args.metric not in summary:
            print(
                f"\nunknown metric {args.metric!r}; have {list(summary)}",
                file=sys.stderr,
            )
            return 2
        score = summary[args.metric]
        if score < args.fail_under:
            print(
                f"\nFAIL  {args.metric} = {score:.3f} < threshold {args.fail_under:.3f}",
                file=sys.stderr,
            )
            return 1
        print(f"\nPASS  {args.metric} = {score:.3f} >= {args.fail_under:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
