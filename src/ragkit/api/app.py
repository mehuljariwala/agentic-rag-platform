"""FastAPI surface: ingest, query, streaming query, health.

The pipeline is process-global and guarded by a lock. Indexing mutates shared
numpy state, so concurrent writes would corrupt it; reads are safe once a write
completes. For a multi-replica deployment the store moves to pgvector -- see
docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..pipeline import RAGConfig, RAGPipeline
from ..types import Document

app = FastAPI(
    title="ragkit",
    version="0.1.0",
    description="Agentic RAG: hybrid retrieval, reranking, grounded citations.",
)

_lock = threading.Lock()
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        with _lock:
            if _pipeline is None:
                _pipeline = RAGPipeline(config=RAGConfig())
    return _pipeline


class DocumentIn(BaseModel):
    id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[DocumentIn] = Field(..., min_length=1)


class IngestResponse(BaseModel):
    documents: int
    chunks_indexed: int
    chunks_total: int


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)
    # Exact-match metadata filter, e.g. {"source": "handbook"}
    filter: dict[str, Any] = Field(default_factory=dict)


class CitationOut(BaseModel):
    marker: int
    chunk_id: str
    doc_id: str
    text: str


class HitOut(BaseModel):
    chunk_id: str
    doc_id: str
    rank: int
    score: float
    scores: dict[str, float]
    text: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    hits: list[HitOut]
    trace: list[dict[str, Any]]
    prompt_tokens: int
    completion_tokens: int


def _predicate(filter_: dict[str, Any]):
    if not filter_:
        return None

    def where(meta: dict[str, Any]) -> bool:
        return all(meta.get(k) == v for k, v in filter_.items())

    return where


@app.get("/health")
def health() -> dict[str, Any]:
    rag = get_pipeline()
    return {
        "status": "ok",
        "provider": os.getenv("RAGKIT_PROVIDER", "mock"),
        "llm": rag.llm.name,
        "embedder": rag.embedder.name,
        "chunks": len(rag.chunks),
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    rag = get_pipeline()
    docs = [Document(id=d.id, text=d.text, metadata=d.metadata) for d in req.documents]
    with _lock:
        indexed = rag.index(docs)
    return IngestResponse(
        documents=len(docs), chunks_indexed=indexed, chunks_total=len(rag.chunks)
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    rag = get_pipeline()
    if not rag.chunks:
        raise HTTPException(status_code=409, detail="index is empty; POST /ingest first")

    if req.top_k is not None:
        rag.config.top_k = req.top_k

    result = rag.answer(req.query, where=_predicate(req.filter))
    return QueryResponse(
        answer=result.text,
        citations=[CitationOut(**vars(c)) for c in result.citations],
        hits=[
            HitOut(
                chunk_id=h.chunk.id,
                doc_id=h.chunk.doc_id,
                rank=h.rank,
                score=h.score,
                scores=h.scores,
                text=h.chunk.text,
            )
            for h in result.hits
        ],
        trace=result.trace,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )


@app.post("/query/stream")
def query_stream(req: QueryRequest) -> StreamingResponse:
    rag = get_pipeline()
    if not rag.chunks:
        raise HTTPException(status_code=409, detail="index is empty; POST /ingest first")

    def events() -> Iterator[str]:
        for token in rag.stream_answer(req.query, where=_predicate(req.filter)):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
    )
