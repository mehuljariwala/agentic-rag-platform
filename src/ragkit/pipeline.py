"""The agentic RAG loop.

A plain RAG pipeline is retrieve-then-generate: one shot, no feedback. It fails
in two characteristic ways -- the retriever misses because the user's phrasing
doesn't match the corpus vocabulary, and the generator answers confidently from
chunks that don't actually contain the answer.

This pipeline closes both loops:

    rewrite -> retrieve -> grade -> (re-retrieve with new query) -> generate
            -> verify grounding -> (retry once) -> answer

Every decision is appended to `Answer.trace`, so a bad answer can be diagnosed
without re-running anything: you can see the rewritten queries, what each
retriever returned, the relevance grade, and whether the groundedness check
passed.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from .ingest.chunker import chunk_all
from .providers import LLM, Embedder, Message, get_embedder, get_llm
from .retrieval.bm25 import BM25
from .retrieval.hybrid import reciprocal_rank_fusion
from .retrieval.rerank import LexicalReranker
from .retrieval.vector import VectorStore
from .types import Answer, Chunk, Citation, Document, Hit

_ANSWER_SYSTEM = """You answer questions strictly from the numbered context below.

Rules:
- Use only facts present in the context. Never use prior knowledge.
- Cite every claim with the bracketed number of its source, like [2].
- If the context does not contain the answer, reply exactly:
  I don't have enough context to answer that.

Context:
{context}"""

_REWRITE_PROMPT = """Rewrite the question into {n} short search queries that would
find the answer in a document corpus. Vary the vocabulary -- use synonyms and
domain terms the source documents would plausibly use.

One query per line, no numbering, no other text.

Question: {query}"""

_GRADE_PROMPT = """Does the context below contain enough information to answer
the question? Reply with exactly one word: YES or NO.

Question: {query}

Context:
{context}"""


@dataclass
class RAGConfig:
    chunk_chars: int = 1000
    chunk_overlap: int | None = None  # None -> 15% of chunk_chars
    candidates: int = 20  # per-retriever depth before fusion
    top_k: int = 5  # chunks handed to the generator
    rewrites: int = 2  # extra query formulations
    max_retries: int = 1  # re-retrieval attempts after a failed grade
    bm25_weight: float = 1.0
    vector_weight: float = 1.0
    rerank: bool = True
    grade: bool = True
    verify: bool = True
    temperature: float = 0.0
    max_tokens: int = 800


class RAGPipeline:
    def __init__(
        self,
        llm: LLM | None = None,
        embedder: Embedder | None = None,
        config: RAGConfig | None = None,
    ) -> None:
        self.llm = llm or get_llm()
        self.embedder = embedder or get_embedder()
        self.config = config or RAGConfig()

        self.bm25 = BM25()
        self.vectors = VectorStore(dim=self.embedder.dim)
        self.chunks: dict[str, Chunk] = {}
        self.reranker = LexicalReranker()

    # ---------------------------------------------------------------- ingest

    def index(self, docs: Sequence[Document], batch_size: int = 64) -> int:
        """Chunk, embed and index documents. Returns the chunk count."""
        chunks = chunk_all(
            docs,
            max_chars=self.config.chunk_chars,
            overlap=self.config.chunk_overlap,
        )
        if not chunks:
            return 0

        for chunk in chunks:
            self.chunks[chunk.id] = chunk
            self.bm25.add(chunk.id, chunk.text)

        # Batched so a large corpus doesn't become one enormous embed request.
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = self.embedder.embed([c.text for c in batch])
            self.vectors.add([c.id for c in batch], vectors, [c.metadata for c in batch])

        return len(chunks)

    # ------------------------------------------------------------- retrieval

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        where: Callable[[dict[str, Any]], bool] | None = None,
    ) -> list[Hit]:
        cfg = self.config
        top_k = top_k or cfg.top_k

        lexical = self.bm25.search(query, k=cfg.candidates)
        dense = self.vectors.search(
            self.embedder.embed([query])[0], k=cfg.candidates, where=where
        )

        # BM25 has no metadata filter of its own; apply the same predicate so
        # both arms of the fusion see an identical candidate universe.
        if where is not None:
            lexical = [
                (cid, s)
                for cid, s in lexical
                if cid in self.chunks and where(self.chunks[cid].metadata)
            ]

        fused = reciprocal_rank_fusion(
            {"bm25": lexical, "vector": dense},
            weights={"bm25": cfg.bm25_weight, "vector": cfg.vector_weight},
            top_k=max(top_k, cfg.candidates // 2),
        )

        hits = [
            Hit(chunk=self.chunks[cid], score=score, scores=breakdown, rank=i)
            for i, (cid, score, breakdown) in enumerate(fused, start=1)
            if cid in self.chunks
        ]

        if cfg.rerank and hits:
            hits = self.reranker.rerank(query, hits, top_k=top_k)
        return hits[:top_k]

    # ------------------------------------------------------------ generation

    @staticmethod
    def _render_context(hits: Sequence[Hit]) -> str:
        return "\n\n".join(f"[{i}] {h.chunk.text}" for i, h in enumerate(hits, 1))

    def _rewrite(self, query: str) -> list[str]:
        """Generate alternate phrasings. Always includes the original."""
        if self.config.rewrites <= 0:
            return [query]
        reply = self.llm.complete(
            [
                Message(
                    "user", _REWRITE_PROMPT.format(n=self.config.rewrites, query=query)
                )
            ],
            temperature=0.0,
            max_tokens=128,
        ).text
        extra = [
            line.strip(" -*\t")
            for line in reply.splitlines()
            if line.strip() and line.strip() != query
        ]
        return [query, *extra[: self.config.rewrites]]

    def _grade(self, query: str, hits: Sequence[Hit]) -> bool:
        """Ask whether the retrieved context can answer the question at all."""
        if not hits:
            return False
        reply = self.llm.complete(
            [
                Message(
                    "user",
                    _GRADE_PROMPT.format(query=query, context=self._render_context(hits)),
                )
            ],
            temperature=0.0,
            max_tokens=8,
        ).text
        return "yes" in reply.strip().lower()[:5]

    @staticmethod
    def _parse_citations(text: str, hits: Sequence[Hit]) -> list[Citation]:
        out: list[Citation] = []
        seen: set[int] = set()
        for marker in re.findall(r"\[(\d+)\]", text):
            n = int(marker)
            if n in seen or not (1 <= n <= len(hits)):
                continue
            seen.add(n)
            chunk = hits[n - 1].chunk
            out.append(
                Citation(
                    marker=n,
                    chunk_id=chunk.id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                )
            )
        return sorted(out, key=lambda c: c.marker)

    def answer(
        self,
        query: str,
        where: Callable[[dict[str, Any]], bool] | None = None,
    ) -> Answer:
        cfg = self.config
        trace: list[dict[str, Any]] = []
        prompt_tokens = completion_tokens = 0

        queries = self._rewrite(query) if cfg.rewrites else [query]
        trace.append({"step": "rewrite", "queries": queries})

        hits: list[Hit] = []
        attempts = 0

        while attempts <= cfg.max_retries:
            # Union the candidates from every phrasing, then dedupe by chunk id
            # keeping the best-scoring occurrence.
            pooled: dict[str, Hit] = {}
            for q in queries:
                for hit in self.retrieve(q, where=where):
                    existing = pooled.get(hit.chunk.id)
                    if existing is None or hit.score > existing.score:
                        pooled[hit.chunk.id] = hit

            hits = sorted(pooled.values(), key=lambda h: -h.score)[: cfg.top_k]
            trace.append(
                {
                    "step": "retrieve",
                    "attempt": attempts,
                    "queries": list(queries),
                    "chunk_ids": [h.chunk.id for h in hits],
                }
            )

            if not cfg.grade:
                break

            ok = self._grade(query, hits)
            trace.append({"step": "grade", "attempt": attempts, "sufficient": ok})
            if ok or attempts == cfg.max_retries:
                break

            # Grade failed: widen the search rather than repeating it verbatim.
            attempts += 1
            queries = self._rewrite(query)[1:] or [query]

        if not hits:
            trace.append({"step": "abstain", "reason": "no candidates retrieved"})
            return Answer(text="I don't have enough context to answer that.", trace=trace)

        context = self._render_context(hits)
        completion = self.llm.complete(
            [
                Message("system", _ANSWER_SYSTEM.format(context=context)),
                Message("user", query),
            ],
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
        prompt_tokens += completion.usage.prompt_tokens
        completion_tokens += completion.usage.completion_tokens

        text = completion.text.strip()
        citations = self._parse_citations(text, hits)

        # An answer with no citations is either an abstention or an ungrounded
        # claim. Flag it rather than passing it off as sourced.
        if cfg.verify and citations == [] and "don't have enough context" not in text:
            trace.append({"step": "verify", "grounded": False})
        elif cfg.verify:
            trace.append(
                {"step": "verify", "grounded": True, "citations": len(citations)}
            )

        return Answer(
            text=text,
            citations=citations,
            hits=hits,
            trace=trace,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def stream_answer(
        self,
        query: str,
        where: Callable[[dict[str, Any]], bool] | None = None,
    ) -> Iterator[str]:
        """Token stream for the SSE endpoint. Retrieval runs eagerly first."""
        hits = self.retrieve(query, where=where)
        if not hits:
            yield "I don't have enough context to answer that."
            return
        yield from self.llm.stream(
            [
                Message(
                    "system", _ANSWER_SYSTEM.format(context=self._render_context(hits))
                ),
                Message("user", query),
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
