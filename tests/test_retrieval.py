from __future__ import annotations

import pytest

from ragkit.providers.mock import MockEmbedder
from ragkit.retrieval.bm25 import BM25, tokenize
from ragkit.retrieval.hybrid import reciprocal_rank_fusion
from ragkit.retrieval.vector import VectorStore


class TestBM25:
    def test_ranks_the_relevant_document_first(self):
        idx = BM25()
        idx.add("a", "autovacuum reclaims dead tuples in postgresql")
        idx.add("b", "kafka consumer groups rebalance on heartbeat timeout")
        idx.add("c", "redis evicts keys using an lru policy")

        results = idx.search("dead tuples vacuum")
        assert results[0][0] == "a"

    def test_stopwords_do_not_dominate(self):
        idx = BM25()
        idx.add("a", "the quick brown fox")
        idx.add("b", "the lazy dog sleeps")
        # "the" is in both and is a stopword; only "fox" should decide.
        results = idx.search("the fox")
        assert [doc for doc, _ in results] == ["a"]

    def test_empty_index_returns_nothing(self):
        assert BM25().search("anything") == []

    def test_unknown_terms_return_nothing(self):
        idx = BM25()
        idx.add("a", "postgres indexes")
        assert idx.search("kubernetes helm chart") == []

    def test_idf_is_never_negative(self):
        """A term in every document must not push scores below zero."""
        idx = BM25()
        for i in range(5):
            idx.add(str(i), "common term here")
        assert all(score >= 0.0 for _, score in idx.search("common"))

    def test_tokenize_strips_punctuation_and_case(self):
        assert tokenize("Hello, WORLD! 42.") == ["hello", "world", "42"]


class TestVectorStore:
    def test_finds_nearest_neighbour(self):
        emb = MockEmbedder(dim=128)
        store = VectorStore(dim=128)
        texts = {
            "pg": "postgresql autovacuum dead tuples storage",
            "kafka": "kafka partition rebalance consumer group",
        }
        store.add(list(texts), emb.embed(list(texts.values())))

        query = emb.embed(["autovacuum dead tuples"])[0]
        assert store.search(query, k=1)[0][0] == "pg"

    def test_metadata_filter_excludes_non_matching(self):
        emb = MockEmbedder(dim=64)
        store = VectorStore(dim=64)
        store.add(
            ["a", "b"],
            emb.embed(["postgres vacuum", "postgres vacuum"]),
            [{"env": "prod"}, {"env": "dev"}],
        )
        hits = store.search(
            emb.embed(["postgres vacuum"])[0], k=5, where=lambda m: m["env"] == "prod"
        )
        assert [h[0] for h in hits] == ["a"]

    def test_filter_matching_nothing_returns_empty(self):
        emb = MockEmbedder(dim=64)
        store = VectorStore(dim=64)
        store.add(["a"], emb.embed(["text"]), [{"env": "prod"}])
        assert (
            store.search(emb.embed(["text"])[0], where=lambda m: m["env"] == "nope") == []
        )

    def test_dimension_mismatch_is_rejected(self):
        store = VectorStore(dim=8)
        with pytest.raises(ValueError, match="expected vectors of shape"):
            store.add(["a"], [[0.1, 0.2]])

    def test_k_larger_than_corpus_is_clamped(self):
        emb = MockEmbedder(dim=32)
        store = VectorStore(dim=32)
        store.add(["a", "b"], emb.embed(["one", "two"]))
        assert len(store.search(emb.embed(["one"])[0], k=99)) == 2

    def test_empty_store_returns_nothing(self):
        assert VectorStore(dim=8).search([0.0] * 8) == []


class TestFusion:
    def test_document_ranked_by_both_retrievers_wins(self):
        fused = reciprocal_rank_fusion(
            {
                "bm25": [("a", 9.0), ("b", 8.0), ("c", 1.0)],
                "vector": [("c", 0.9), ("a", 0.8), ("d", 0.1)],
            },
            top_k=4,
        )
        # "a" is 1st and 2nd; "c" is 3rd and 1st. "a" should lead.
        assert fused[0][0] == "a"

    def test_raw_scores_are_preserved_per_retriever(self):
        fused = reciprocal_rank_fusion(
            {"bm25": [("a", 9.0)], "vector": [("a", 0.42)]}, top_k=1
        )
        assert fused[0][2] == {"bm25": 9.0, "vector": 0.42}

    def test_weights_shift_the_ordering(self):
        lists = {"bm25": [("a", 1.0)], "vector": [("b", 1.0)]}
        assert reciprocal_rank_fusion(lists, weights={"bm25": 5.0}, top_k=1)[0][0] == "a"
        assert (
            reciprocal_rank_fusion(lists, weights={"vector": 5.0}, top_k=1)[0][0] == "b"
        )

    def test_handles_empty_input(self):
        assert reciprocal_rank_fusion({}) == []
