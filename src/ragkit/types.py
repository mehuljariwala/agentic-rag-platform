"""Core data types shared across ingest, retrieval and generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A source document before chunking."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrievable span of a document."""

    id: str
    doc_id: str
    text: str
    ordinal: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Hit:
    """A retrieved chunk with the scores that produced it.

    `scores` keeps the per-retriever contributions rather than collapsing to a
    single number, because that breakdown is what makes retrieval failures
    debuggable -- you can see whether lexical or dense matching found the chunk.
    """

    chunk: Chunk
    score: float
    scores: dict[str, float] = field(default_factory=dict)
    rank: int = 0


@dataclass
class Citation:
    marker: int
    chunk_id: str
    doc_id: str
    text: str


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    # Trace of what the agentic loop actually did -- rewrites, retries, grades.
    trace: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
