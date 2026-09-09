from __future__ import annotations

from ragkit import Document, RAGConfig, RAGPipeline


class TestIndexing:
    def test_index_reports_chunk_count(self, corpus):
        rag = RAGPipeline(config=RAGConfig(rewrites=0, grade=False))
        n = rag.index(corpus)
        assert n == len(rag.chunks) > 0

    def test_indexing_populates_both_retrievers(self, rag):
        assert len(rag.bm25.doc_ids) == len(rag.chunks)
        assert len(rag.vectors) == len(rag.chunks)

    def test_indexing_nothing_is_a_noop(self):
        assert RAGPipeline().index([]) == 0


class TestRetrieval:
    def test_retrieves_the_topically_correct_document(self, rag):
        hits = rag.retrieve("why do dead tuples accumulate?")
        assert hits[0].chunk.doc_id == "pg-vacuum"

    def test_hits_expose_per_retriever_scores(self, rag):
        hit = rag.retrieve("consumer group rebalance")[0]
        assert "rerank" in hit.scores
        assert {"bm25", "vector"} & set(hit.scores)

    def test_ranks_are_sequential_from_one(self, rag):
        hits = rag.retrieve("index types")
        assert [h.rank for h in hits] == list(range(1, len(hits) + 1))

    def test_metadata_filter_restricts_results(self, rag):
        hits = rag.retrieve(
            "what happens on rebalance", where=lambda m: m["source"] == "redis"
        )
        assert hits
        assert all(h.chunk.metadata["source"] == "redis" for h in hits)

    def test_top_k_is_respected(self, rag):
        assert len(rag.retrieve("postgres", top_k=2)) <= 2


class TestAnswering:
    def test_answer_is_grounded_and_cited(self, rag):
        result = rag.answer("what does autovacuum reclaim?")
        assert result.citations
        assert all(c.marker >= 1 for c in result.citations)

    def test_citation_markers_resolve_to_real_chunks(self, rag):
        result = rag.answer("which index type suits time series?")
        for citation in result.citations:
            assert citation.chunk_id in rag.chunks
            assert citation.text == rag.chunks[citation.chunk_id].text

    def test_abstains_on_an_empty_index(self):
        empty = RAGPipeline(config=RAGConfig(rewrites=0, grade=False))
        assert "don't have enough context" in empty.answer("anything?").text

    def test_trace_records_each_step(self, rag):
        steps = [entry["step"] for entry in rag.answer("what is BRIN for?").trace]
        assert "retrieve" in steps
        assert "verify" in steps

    def test_token_usage_is_accounted(self, rag):
        result = rag.answer("how does GIN work?")
        assert result.prompt_tokens > 0
        assert result.completion_tokens > 0

    def test_streaming_yields_the_same_content(self, rag):
        streamed = "".join(rag.stream_answer("what does autovacuum reclaim?"))
        assert streamed.strip()

    def test_streaming_abstains_on_empty_index(self):
        empty = RAGPipeline(config=RAGConfig(rewrites=0, grade=False))
        assert "don't have enough context" in "".join(empty.stream_answer("hi"))


class TestGradingLoop:
    def test_grading_step_is_traced_when_enabled(self, corpus):
        rag = RAGPipeline(config=RAGConfig(rewrites=0, grade=True, max_retries=1))
        rag.index(corpus)
        steps = [e["step"] for e in rag.answer("what does autovacuum reclaim?").trace]
        assert "grade" in steps

    def test_retry_count_is_bounded(self, corpus):
        """An unanswerable question must not loop past max_retries."""
        rag = RAGPipeline(config=RAGConfig(rewrites=0, grade=True, max_retries=2))
        rag.index(corpus)
        attempts = [
            e
            for e in rag.answer("what is the airspeed of a swallow?").trace
            if e["step"] == "retrieve"
        ]
        assert len(attempts) <= 3  # initial + 2 retries


class TestConfig:
    def test_reranking_can_be_disabled(self, corpus):
        rag = RAGPipeline(config=RAGConfig(rewrites=0, grade=False, rerank=False))
        rag.index(corpus)
        assert "rerank" not in rag.retrieve("postgres indexes")[0].scores

    def test_lexical_weighting_changes_results(self, corpus):
        lexical = RAGPipeline(
            config=RAGConfig(rewrites=0, grade=False, vector_weight=0.0)
        )
        lexical.index(corpus)
        assert lexical.retrieve("BRIN")[0].chunk.doc_id == "pg-index"

    def test_chunk_size_config_is_applied(self, corpus):
        rag = RAGPipeline(config=RAGConfig(chunk_chars=150, chunk_overlap=0))
        rag.index(corpus)
        assert all(len(c.text) <= 150 for c in rag.chunks.values())


def test_documents_can_be_added_incrementally(rag):
    before = len(rag.chunks)
    rag.index([Document(id="new", text="Elasticsearch shards split an index.")])
    assert len(rag.chunks) > before
    assert rag.retrieve("elasticsearch shards")[0].chunk.doc_id == "new"
