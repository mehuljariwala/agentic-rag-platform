# ragkit — Agentic RAG Platform

[![CI](https://github.com/mehuljariwala/agentic-rag-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/mehuljariwala/agentic-rag-platform/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Retrieval-augmented generation that **grades its own retrieval and cites its own answers** — hybrid lexical + dense search fused by reciprocal rank, a reranking second stage, and a self-correcting loop that re-queries when the retrieved context can't actually answer the question.

**Runs with zero API keys.** The default provider is a deterministic offline model, so `pytest` and the eval harness are green on a fresh clone with no configuration. Swap in Anthropic, OpenAI or Ollama with one environment variable.

```bash
git clone https://github.com/mehuljariwala/agentic-rag-platform && cd agentic-rag-platform
uv venv && uv pip install -e ".[dev]"
pytest                      # 48 tests, no network, no keys
python -m evals.run_eval    # retrieval quality report
```

---

## Why this exists

Most RAG implementations are `retrieve()` then `generate()`. That pipeline has two failure modes it cannot detect, let alone recover from:

1. **Vocabulary mismatch.** The user asks "why is my table bloating?" and the docs say "dead tuples accumulate under MVCC". Pure dense retrieval often catches this; pure lexical never does. Neither knows when it missed.
2. **Confident ungrounded answers.** The retriever returns five plausible-looking chunks, none of which contain the answer, and the model writes a fluent paragraph anyway.

ragkit addresses both structurally rather than by prompt-tweaking.

## Architecture

```
                    ┌──────────────┐
   query ──────────▶│   rewrite    │  n alternate phrasings
                    └──────┬───────┘
                           ▼
          ┌────────────────────────────────┐
          │  BM25 (lexical)  Vector (dense)│  candidates=20 each
          └────────────────┬───────────────┘
                           ▼
                  ┌─────────────────┐
                  │  RRF fusion     │  rank-based, scale-free
                  └────────┬────────┘
                           ▼
                  ┌─────────────────┐
                  │    rerank       │  recall@1: 0.819 → 0.861
                  └────────┬────────┘
                           ▼
                  ┌─────────────────┐        insufficient
                  │  relevance grade├──────────────┐
                  └────────┬────────┘              │
                           │ sufficient            ▼
                           ▼                  re-query with
                  ┌─────────────────┐         new phrasings
                  │  generate       │              │
                  │  + citations    │◀─────────────┘
                  └────────┬────────┘
                           ▼
                  ┌─────────────────┐
                  │ groundedness    │  flags uncited claims
                  └─────────────────┘
```

Full design notes and the pgvector migration path: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

### Why reciprocal rank fusion, not score blending

BM25 returns unbounded positive scores; cosine similarity returns `[-1, 1]`. Adding or averaging them is meaningless, and min-max normalising makes the result depend on whatever happens to be in the candidate set that day. RRF fuses on **rank** instead:

```
RRF(d) = Σ  weight_r / (k + rank_r(d))          k = 60
```

A document that both retrievers rank highly beats one that a single retriever loves — which is exactly the behaviour you want from hybrid search. `k=60` is from [Cormack et al. 2009](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf).

## Usage

```python
from ragkit import Document, RAGPipeline

rag = RAGPipeline()                       # offline provider by default
rag.index([
    Document(id="pg", text="Autovacuum reclaims storage from dead tuples...",
             metadata={"source": "postgres"}),
])

answer = rag.answer("what does autovacuum reclaim?")
print(answer.text)                        # "Autovacuum reclaims storage ... [1]"
print(answer.citations[0].doc_id)         # "pg"

for step in answer.trace:                 # every decision, replayable
    print(step)
```

### Metadata filtering

```python
rag.answer("how do I tune this?", where=lambda m: m["source"] == "postgres")
```

The predicate is applied to **both** retrieval arms, so lexical and dense search always see the same candidate universe.

### Choosing a provider

| `RAGKIT_PROVIDER` | LLM | Embeddings | Needs |
|---|---|---|---|
| `mock` *(default)* | extractive, deterministic | hashed bag-of-words | nothing |
| `ollama` | any local model | `nomic-embed-text` | Ollama running |
| `openai` | `gpt-4o-mini` | `text-embedding-3-small` | `OPENAI_API_KEY` |
| `anthropic` | `claude-sonnet-5` | falls back to local | `ANTHROPIC_API_KEY` |

```bash
RAGKIT_PROVIDER=ollama python -m evals.run_eval
```

No vendor SDK is a dependency — the remote providers use `urllib` against the public HTTP APIs, which keeps the container image small and the dependency tree auditable.

## HTTP API

```bash
uv pip install -e ".[api]"
uvicorn ragkit.api.app:app --reload
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | provider, model and index size |
| `POST /ingest` | chunk, embed and index documents |
| `POST /query` | answer with citations, hits and full trace |
| `POST /query/stream` | same, streamed over SSE |

```bash
curl -s localhost:8000/ingest -H 'content-type: application/json' -d '{
  "documents": [{"id":"pg","text":"Autovacuum reclaims storage from dead tuples."}]
}'

curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"query":"what does autovacuum reclaim?"}' | jq .answer
```

Interactive docs at `http://localhost:8000/docs`.

## Evaluation

Retrieval quality is measured, not asserted. `evals/` ships a 27-document / 48-query labelled dataset built to be hard — the corpus contains deliberate near-neighbours (five documents discuss "eviction", four discuss reclaiming space, three discuss retries), so lexical overlap alone doesn't identify the right document.

```bash
python -m evals.run_eval --compare
```

```
config               recall@1  recall@5  mrr    ndcg@5  hit@5
─────────────────────────────────────────────────────────────
bm25 only            0.861     0.983     0.963  0.957   1.000
vector only          0.694     0.913     0.835  0.844   0.938
hybrid (RRF, 1:1)    0.819     0.983     0.935  0.937   1.000
hybrid (RRF, 3:1)    0.819     0.983     0.942  0.944   1.000
hybrid 3:1 + rerank  0.861     0.983     0.959  0.956   1.000
```

**Read that table honestly: hybrid retrieval does not beat BM25 here.** The offline embedder is a hashed *subword* model — robust to morphology (`vacuum` ↔ `vacuuming`) but with no semantics, so it can't connect `bloat` to `dead tuples`. Fusing a much weaker arm at equal weight costs 4 points of recall@1. Down-weighting it 3:1 recovers most of the MRR and reranking closes the gap, but on this corpus fusion buys nothing over lexical search alone.

That result is left in the table rather than tuned away, because it's the whole point of having a harness. Hybrid earns its complexity only when the dense arm contributes signal the lexical arm can't — which needs a real embedding model:

```bash
RAGKIT_PROVIDER=ollama python -m evals.run_eval --compare --by-style
```

Watch the `paraphrase` slice: those queries are written to share no content words with their source document, so that's exactly where a semantic embedder should pull ahead and a subword one can't.

Numbers from the offline provider reproduce exactly on any machine, which is what makes `--fail-under` usable as a CI gate against retrieval regressions.

## Docker

```bash
docker compose up
```

Multi-stage build, non-root user, healthcheck on `/health`.

## Testing

```
pytest --cov=src/ragkit
```

48 tests covering BM25 ranking and IDF flooring, vector filtering and dimension validation, RRF ordering and weighting, chunk-boundary overlap, the grading retry bound, citation resolution, and the HTTP layer. No network, no mocks-of-mocks — the offline provider is a real implementation of the same protocol the vendors implement.

## Project layout

```
src/ragkit/
├── providers/       LLM + Embedder protocols, offline and remote impls
├── ingest/          structure-aware recursive chunking
├── retrieval/       bm25 · vector · hybrid fusion · rerank
├── pipeline.py      the agentic loop
└── api/             FastAPI + SSE
evals/               labelled dataset, metrics, regression gate
docs/ARCHITECTURE.md design decisions and trade-offs
```

## License

MIT — see [LICENSE](LICENSE).

---

Built by [Mehul Jariwala](https://github.com/mehuljariwala) · [LinkedIn](https://www.linkedin.com/in/mehul-jariwala-352a01132/)
