from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from ragkit.api import app as app_module


@pytest.fixture
def client(monkeypatch):
    """Fresh pipeline per test so index state never leaks between cases."""
    monkeypatch.setattr(app_module, "_pipeline", None)
    return TestClient(app_module.app)


@pytest.fixture
def seeded(client):
    client.post(
        "/ingest",
        json={
            "documents": [
                {
                    "id": "pg",
                    "text": (
                        "Autovacuum reclaims storage occupied by dead tuples. "
                        "Dead tuples accumulate under multiversion concurrency control."
                    ),
                    "metadata": {"source": "postgres"},
                },
                {
                    "id": "kafka",
                    "text": (
                        "A consumer group rebalances when a member joins or leaves. "
                        "During a rebalance consumption stops entirely."
                    ),
                    "metadata": {"source": "kafka"},
                },
            ]
        },
    )
    return client


class TestHealth:
    def test_reports_provider_and_index_size(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["llm"] == "mock-llm"
        assert body["chunks"] == 0


class TestIngest:
    def test_returns_chunk_counts(self, client):
        r = client.post(
            "/ingest", json={"documents": [{"id": "a", "text": "hello world"}]}
        )
        assert r.status_code == 200
        body = r.json()
        assert body["documents"] == 1
        assert body["chunks_indexed"] >= 1
        assert body["chunks_total"] == body["chunks_indexed"]

    def test_ingest_is_cumulative(self, client):
        client.post("/ingest", json={"documents": [{"id": "a", "text": "first doc"}]})
        second = client.post(
            "/ingest", json={"documents": [{"id": "b", "text": "second doc"}]}
        ).json()
        assert second["chunks_total"] > second["chunks_indexed"]

    def test_empty_document_list_is_rejected(self, client):
        assert client.post("/ingest", json={"documents": []}).status_code == 422

    def test_blank_text_is_rejected(self, client):
        r = client.post("/ingest", json={"documents": [{"id": "a", "text": ""}]})
        assert r.status_code == 422


class TestQuery:
    def test_returns_answer_with_citations(self, seeded):
        r = seeded.post("/query", json={"query": "what does autovacuum reclaim?"})
        assert r.status_code == 200
        body = r.json()
        assert body["answer"]
        assert body["citations"]
        assert body["citations"][0]["doc_id"] in {"pg", "kafka"}

    def test_hits_carry_score_breakdown(self, seeded):
        body = seeded.post("/query", json={"query": "dead tuples"}).json()
        assert body["hits"]
        assert body["hits"][0]["scores"]

    def test_trace_is_returned(self, seeded):
        body = seeded.post("/query", json={"query": "rebalance"}).json()
        assert [s["step"] for s in body["trace"]]

    def test_metadata_filter_is_applied(self, seeded):
        body = seeded.post(
            "/query", json={"query": "what happens?", "filter": {"source": "kafka"}}
        ).json()
        assert body["hits"]
        assert all(h["doc_id"] == "kafka" for h in body["hits"])

    def test_top_k_is_respected(self, seeded):
        body = seeded.post("/query", json={"query": "tuples", "top_k": 1}).json()
        assert len(body["hits"]) <= 1

    def test_querying_an_empty_index_is_a_conflict(self, client):
        r = client.post("/query", json={"query": "anything"})
        assert r.status_code == 409
        assert "ingest" in r.json()["detail"]

    def test_blank_query_is_rejected(self, seeded):
        assert seeded.post("/query", json={"query": ""}).status_code == 422

    def test_out_of_range_top_k_is_rejected(self, seeded):
        assert seeded.post("/query", json={"query": "x", "top_k": 0}).status_code == 422
        assert seeded.post("/query", json={"query": "x", "top_k": 99}).status_code == 422


class TestStreaming:
    def test_streams_sse_and_terminates(self, seeded):
        with seeded.stream(
            "POST", "/query/stream", json={"query": "what does autovacuum reclaim?"}
        ) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            body = "".join(r.iter_text())

        assert body.endswith("data: [DONE]\n\n")
        payloads = [
            json.loads(line[len("data: ") :])
            for line in body.splitlines()
            if line.startswith("data: ") and not line.endswith("[DONE]")
        ]
        assert "".join(p["token"] for p in payloads).strip()

    def test_streaming_empty_index_is_a_conflict(self, client):
        assert client.post("/query/stream", json={"query": "x"}).status_code == 409
