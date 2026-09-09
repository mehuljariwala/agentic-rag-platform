from __future__ import annotations

import pytest

from ragkit.ingest.chunker import chunk_all, chunk_document, resolve_overlap
from ragkit.types import Document


def test_short_document_stays_one_chunk():
    doc = Document(id="d", text="A single short sentence.")
    chunks = chunk_document(doc, max_chars=1000)
    assert len(chunks) == 1
    assert chunks[0].text == "A single short sentence."


def test_long_document_is_split_within_budget():
    doc = Document(id="d", text=" ".join(f"sentence number {i}." for i in range(400)))
    chunks = chunk_document(doc, max_chars=200, overlap=20)
    assert len(chunks) > 1
    # Overlap is prepended, so allow for it in the ceiling.
    assert all(len(c.text) <= 200 + 20 for c in chunks)


def test_chunk_ids_are_stable_and_ordered():
    doc = Document(id="doc7", text="alpha. " * 200)
    chunks = chunk_document(doc, max_chars=100, overlap=0)
    assert [c.id for c in chunks] == [f"doc7::{i}" for i in range(len(chunks))]
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert all(c.doc_id == "doc7" for c in chunks)


def test_metadata_propagates_to_every_chunk():
    doc = Document(id="d", text="word " * 300, metadata={"source": "handbook"})
    for chunk in chunk_document(doc, max_chars=120):
        assert chunk.metadata["source"] == "handbook"


def test_metadata_is_copied_not_shared():
    """Mutating one chunk's metadata must not affect its siblings."""
    doc = Document(id="d", text="word " * 300, metadata={"source": "a"})
    chunks = chunk_document(doc, max_chars=120)
    chunks[0].metadata["source"] = "mutated"
    assert chunks[1].metadata["source"] == "a"


def test_overlap_carries_context_across_the_boundary():
    doc = Document(id="d", text="alpha beta gamma delta. " * 40)
    with_overlap = chunk_document(doc, max_chars=150, overlap=40)
    without = chunk_document(doc, max_chars=150, overlap=0)
    assert sum(len(c.text) for c in with_overlap) > sum(len(c.text) for c in without)


def test_markdown_headings_start_new_chunks():
    text = "## First\n" + ("a" * 300) + "\n## Second\n" + ("b" * 300)
    chunks = chunk_document(Document(id="d", text=text), max_chars=400, overlap=0)
    assert any(c.text.startswith("## Second") for c in chunks)


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError, match="overlap must be smaller"):
        chunk_document(Document(id="d", text="x"), max_chars=100, overlap=100)


def test_negative_overlap_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        chunk_document(Document(id="d", text="x"), max_chars=100, overlap=-1)


def test_default_overlap_scales_with_chunk_size():
    """A small max_chars must not inherit an oversized default overlap."""
    assert resolve_overlap(1000, None) == 150
    assert resolve_overlap(120, None) == 18
    # The pair that used to raise now just works.
    assert chunk_document(Document(id="d", text="word " * 300), max_chars=120)


def test_whitespace_only_document_yields_nothing():
    assert chunk_document(Document(id="d", text="   \n\n  ")) == []


def test_oversized_atom_is_hard_cut():
    """A single unbroken token longer than the budget still gets split."""
    chunks = chunk_document(Document(id="d", text="x" * 500), max_chars=100, overlap=0)
    assert len(chunks) >= 5
    assert all(len(c.text) <= 100 for c in chunks)


def test_chunk_all_spans_documents():
    docs = [Document(id=f"d{i}", text="content here. " * 30) for i in range(3)]
    chunks = chunk_all(docs, max_chars=100)
    assert {c.doc_id for c in chunks} == {"d0", "d1", "d2"}
