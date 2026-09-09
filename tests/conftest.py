from __future__ import annotations

import pytest

from ragkit import Document, RAGConfig, RAGPipeline

CORPUS = [
    Document(
        id="pg-vacuum",
        text=(
            "## Autovacuum in PostgreSQL\n"
            "Autovacuum reclaims storage occupied by dead tuples. Dead tuples "
            "accumulate because PostgreSQL uses multiversion concurrency control, "
            "so an UPDATE writes a new row version and leaves the old one behind. "
            "When the dead tuple count exceeds the autovacuum threshold the daemon "
            "starts a worker for that table. Tuning autovacuum_vacuum_scale_factor "
            "lower makes vacuuming more frequent on large tables."
        ),
        metadata={"source": "postgres", "kind": "ops"},
    ),
    Document(
        id="pg-index",
        text=(
            "## Index types\n"
            "PostgreSQL supports B-tree, hash, GiST, SP-GiST, GIN and BRIN indexes. "
            "B-tree is the default and handles equality and range queries. GIN is "
            "designed for composite values such as arrays and full-text documents, "
            "and is what powers tsvector search. BRIN indexes are tiny and suit "
            "naturally ordered append-only tables such as time series."
        ),
        metadata={"source": "postgres", "kind": "reference"},
    ),
    Document(
        id="kafka-rebalance",
        text=(
            "## Consumer group rebalancing\n"
            "A Kafka consumer group rebalances when a member joins or leaves, or "
            "when the broker stops receiving heartbeats within session.timeout.ms. "
            "During a rebalance partitions are reassigned and consumption pauses. "
            "Cooperative sticky assignment reduces this pause because it only moves "
            "the partitions that actually need to change owner."
        ),
        metadata={"source": "kafka", "kind": "ops"},
    ),
    Document(
        id="redis-eviction",
        text=(
            "## Eviction policies\n"
            "Redis evicts keys when memory reaches maxmemory. The allkeys-lru policy "
            "removes the least recently used key regardless of TTL, while "
            "volatile-lru only considers keys that have an expiry set. Choosing "
            "noeviction makes writes fail once the limit is reached instead of "
            "silently discarding data."
        ),
        metadata={"source": "redis", "kind": "ops"},
    ),
]


@pytest.fixture
def corpus() -> list[Document]:
    return list(CORPUS)


@pytest.fixture
def rag(corpus: list[Document]) -> RAGPipeline:
    """Offline pipeline over the fixture corpus.

    Rewrites and grading are disabled by default because the mock LLM is
    extractive and returns no useful rewrite lines; the tests that exercise
    those paths enable them explicitly.
    """
    pipeline = RAGPipeline(config=RAGConfig(rewrites=0, grade=False))
    pipeline.index(corpus)
    return pipeline
