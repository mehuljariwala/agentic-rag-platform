# Architecture

Design decisions, the reasoning behind them, and the places where this system
would need to change to run at a different scale.

## Contents

- [The problem with retrieve-then-generate](#the-problem-with-retrieve-then-generate)
- [Provider abstraction](#provider-abstraction)
- [Chunking](#chunking)
- [Retrieval](#retrieval)
- [Fusion](#fusion)
- [Reranking](#reranking)
- [The agentic loop](#the-agentic-loop)
- [Evaluation](#evaluation)
- [Scaling: what breaks and what replaces it](#scaling-what-breaks-and-what-replaces-it)
- [Things deliberately left out](#things-deliberately-left-out)

---

## The problem with retrieve-then-generate

The baseline RAG pipeline is two calls:

```python
chunks = retrieve(query)
return generate(prompt(chunks, query))
```

It has no way to notice either of its characteristic failures.

**Retrieval miss from vocabulary mismatch.** The user asks "why is my table
bloating?"; the documentation says "dead tuples accumulate under MVCC". Lexical
search finds nothing because no content word overlaps. The pipeline proceeds
anyway with whatever five chunks scored least badly.

**Ungrounded generation.** Given five irrelevant chunks and a direct question,
an instruction-tuned model will usually still answer. It reads as confident and
it is wrong, and nothing in the pipeline distinguishes it from a correct answer.

Both are addressed structurally here rather than with prompt wording, because
prompt wording is not a control surface you can test or measure.

## Provider abstraction

Two protocols, `LLM` and `Embedder` (`providers/base.py`). Everything else in
the codebase depends only on those.

The consequential decision is that **the offline provider is a real
implementation, not a test double**. `MockLLM` is extractive — it selects
sentences from the supplied context and emits them with citation markers in the
same format the real providers are prompted to produce. `MockEmbedder` is a
hashed subword model that produces genuinely comparable vectors.

This buys three things:

1. `pytest` runs with no network and no keys, so a stranger can clone the repo
   and get a green suite in one command.
2. The eval harness produces **identical numbers on every machine**, which is
   what makes a CI regression gate possible at all.
3. There are no `unittest.mock.patch` calls threaded through the test suite, so
   the tests exercise real call paths rather than an imagined version of them.

The cost is that the offline provider cannot demonstrate semantic retrieval.
That limitation is stated explicitly rather than hidden — see
[Evaluation](#evaluation).

No vendor SDK is a dependency. The remote providers call the public HTTP APIs
through `urllib`, which keeps the image small, the dependency tree auditable,
and the failure modes visible instead of buried under a retry layer somebody
else configured.

## Chunking

`ingest/chunker.py` walks a separator hierarchy from coarse to fine —
`\n## `, `\n### `, `\n\n`, `\n`, `. `, ` ` — and only hard-cuts when a single
atom genuinely exceeds the budget.

Fixed-width chunking is the most common cause of bad retrieval in a first
implementation: it cuts through the middle of sentences, tables and code
blocks, producing chunks that are individually incoherent.

**Overlap defaults to a ratio, not a constant.** An earlier version defaulted
to `overlap=150`, which meant `chunk_document(doc, max_chars=120)` raised
`ValueError` — a footgun that surfaced immediately in testing. Defaulting to
15% of `max_chars` keeps the knob correct at any chunk size while an explicitly
contradictory pair still raises.

Overlap is what stops a fact spanning a boundary from becoming unretrievable.
The carried tail is trimmed to a word boundary so the prefix reads as language.

## Retrieval

Two arms, deliberately with different failure modes.

**BM25** (`retrieval/bm25.py`) — Okapi BM25 written out rather than imported.
`k1=1.5` controls term-frequency saturation, `b=0.75` controls length
normalisation. IDF is floored at zero: without the floor, a term appearing in
more than half the corpus contributes *negatively*, so adding a common word to
a query could push the correct document down the list.

**Dense** (`retrieval/vector.py`) — brute-force cosine over a normalised numpy
matrix. Vectors are normalised at write time, so query scoring is a single
matrix-vector product rather than a per-row division.

An exact scan is the right choice at this scale. For corpora up to roughly
10⁵ chunks it is faster than an ANN index in wall-clock terms once you count
index construction, and its recall is exactly 1.0 rather than "approximately,
depending on parameters you will not tune". HNSW earns its complexity somewhere
above that; see [Scaling](#scaling-what-breaks-and-what-replaces-it).

**Metadata filters apply to both arms.** BM25 has no native filter, so the
predicate is applied to its results explicitly. Filtering only the dense arm
would mean the two retrievers were ranking over different candidate universes
and the fusion below would be comparing incomparable lists.

## Fusion

Reciprocal Rank Fusion, `retrieval/hybrid.py`:

```
RRF(d) = Σ_r  weight_r / (k + rank_r(d))          k = 60
```

BM25 returns unbounded positive scores. Cosine similarity returns `[-1, 1]`.
There is no principled way to add or average them. Min-max normalising each
list before combining is the usual workaround, and it makes the fused result
depend on the spread of whatever happened to be in the candidate set — the same
document can move several places because an unrelated document entered or left
the pool.

RRF avoids the problem by discarding scores and fusing on **rank**, which is
the one thing both retrievers produce comparably. `k=60` is from
[Cormack et al. 2009](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf);
it damps the top rank enough that one retriever cannot unilaterally decide the
fused ordering.

The raw per-retriever scores are still carried through in `Hit.scores`. They
are useless for ranking but essential for debugging — they are how you tell
whether a chunk was found lexically, densely, or by both.

**Weights are not decorative.** Equal-weight RRF assumes the arms are of
comparable quality. When one is clearly weaker, equal weighting drags the fused
list *below* the stronger arm used alone. This is measurable in this repository
and is discussed in [Evaluation](#evaluation).

## Reranking

Fusion produces a good candidate set and a mediocre ordering: it knows about
ranks, never about whether a chunk answers the question.

`LexicalReranker` is the default because it is free and offline: query-term
coverage weighted by rarity *within the shortlist* (a term present in every
candidate discriminates nothing and is weighted to near zero), plus a proximity
bonus for candidates where the query terms cluster together.

`LLMReranker` asks the model to score each candidate 0–10. It is better and
costs one call per query. It also degrades safely: if the reply cannot be
parsed into scores, it returns the incoming fused order rather than an ordering
derived from partial garbage.

## The agentic loop

`pipeline.py`:

```
rewrite ──▶ retrieve ──▶ grade ──┬── sufficient ──▶ generate ──▶ verify
   ▲                             │
   └────── insufficient ─────────┘   (bounded by max_retries)
```

**Rewrite** produces alternate phrasings. Candidates from every phrasing are
pooled and deduplicated by chunk id, keeping the best-scoring occurrence. This
is the cheapest available defence against vocabulary mismatch.

**Grade** asks the model a single yes/no question: does this context contain
enough to answer? It is a separate call with an eight-token budget, kept
separate from generation on purpose — a model asked to both judge and answer in
one call will rationalise its way to answering.

**Retry** re-queries with *new* phrasings rather than repeating the same
search, and is bounded by `max_retries`. An unbounded self-correcting loop is
an unbounded bill.

**Verify** checks that the answer carries citations. An answer with none is
either an abstention or an ungrounded claim; both are flagged in the trace
rather than passed off as sourced.

Every step appends to `Answer.trace`. A bad answer can then be diagnosed
without re-running anything: the rewrites, the chunk ids each retrieval
returned, the grade, and the groundedness result are all in the response.

## Evaluation

`evals/` ships a 27-document / 48-query labelled dataset built to be hard. The
corpus contains deliberate near-neighbours — five documents discuss "eviction",
four discuss reclaiming space, three discuss retries — so lexical overlap alone
does not identify the right document. Queries are tagged `exact`,
`morphological`, `paraphrase` or `multi`, so failures can be attributed to a
kind of query rather than averaged into a single opaque number.

Scoring is at document level. Chunk-level labels would be more precise but they
would need relabelling every time the chunker changed, which in practice means
they rot.

**`precision@3` is deliberately not a headline metric.** With a dataset where
most queries have one correct answer it is capped at 0.333 for reasons that
have nothing to do with retrieval quality, so it cannot be compared across
configurations. `recall@1` measures the same "is the top hit right" question
without the artificial ceiling, and nDCG handles the multi-answer queries.

### What the numbers actually say

Measured on the offline provider, so these reproduce exactly:

```
config               recall@1  recall@5  mrr    ndcg@5  hit@5
─────────────────────────────────────────────────────────────
bm25 only            0.861     0.983     0.963  0.957   1.000
vector only          0.694     0.913     0.835  0.844   0.938
hybrid (RRF, 1:1)    0.819     0.983     0.935  0.937   1.000
hybrid (RRF, 3:1)    0.819     0.983     0.942  0.944   1.000
hybrid 3:1 + rerank  0.861     0.983     0.959  0.956   1.000
```

**Hybrid retrieval does not beat BM25 here, and the reason is the embedder.**
`MockEmbedder` is a hashed subword model. It is robust to morphology — it
connects `vacuum` to `vacuuming` where BM25 sees unrelated terms — but it has
no semantics: it cannot connect `bloat` to `dead tuples`. Fusing a much weaker
arm with a stronger one at equal weight costs 4 points of recall@1 (0.861 →
0.819). Down-weighting the weak arm 3:1 recovers most of the MRR, and adding
reranking closes the gap, but on this corpus with this embedder the honest
summary is that **fusion buys nothing over BM25 alone**.

That is the expected result and it is left in the table rather than tuned away.
Hybrid retrieval is worth its complexity when the dense arm contributes signal
the lexical arm cannot, which requires a real embedding model:

```bash
RAGKIT_PROVIDER=ollama python -m evals.run_eval --compare --by-style
```

The `paraphrase` queries are the ones to watch — they are written to share no
content words with their source document, so they are precisely where a
semantic embedder should pull ahead and a subword one cannot. Whether it
actually does on your corpus is an empirical question, which is the entire
argument for shipping the harness alongside the pipeline.

The known weak spot is `multi` queries: recall@1 of 0.476, because when several
documents are correct the top slot can only hold one of them. nDCG@5 of 0.911
on that slice is the more meaningful reading.

## Scaling: what breaks and what replaces it

| Limit | Symptom | Change |
|---|---|---|
| ~10⁵ chunks | brute-force scan latency becomes visible | pgvector with an HNSW index, or Qdrant |
| Multiple replicas | in-process store is per-process; ingest on one replica is invisible to the others | move both indexes out: pgvector for dense, Postgres FTS or OpenSearch for lexical |
| Corpus updates | no delete or update path; `index()` only appends | content-hash chunk ids and upsert |
| Long documents | whole document re-chunked on every ingest | hash per document, skip unchanged |
| High QPS | one embed call per query on the hot path | LRU cache on query embeddings; they repeat far more than you would expect |

The `VectorStore` interface is small on purpose — `add`, `search`, `__len__` —
so a pgvector implementation is a drop-in. The pipeline never touches numpy
directly.

Process-global pipeline state in `api/app.py` is guarded by a lock because
indexing mutates a shared numpy matrix. That is correct for a single process
and is the first thing to go when this moves behind more than one replica.

## Things deliberately left out

- **A vector database.** It would add an operational dependency and a network
  hop to demonstrate an interface that is already abstracted.
- **LangChain / LlamaIndex.** The retrieval mechanics are the point of the
  project. Delegating them would leave nothing to look at.
- **A UI.** The trace in the API response is the debugging surface; a chat
  window would show less.
- **Streaming through the full agentic loop.** `stream_answer` runs retrieval
  eagerly then streams generation. Streaming across the grade-and-retry loop
  means emitting tokens you may need to retract, which needs a protocol for
  retraction that a demo does not justify.
