"""Structure-aware recursive chunking.

Fixed-width chunking cuts through the middle of sentences and tables, which is
the single most common cause of bad retrieval in a first-pass RAG system. This
splitter walks a hierarchy of separators from coarse to fine and only falls back
to a hard character cut when a single atom genuinely exceeds the budget.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from ..types import Chunk, Document

# Coarse to fine. Markdown headings first so a section stays whole where it can.
_SEPARATORS: tuple[str, ...] = (
    "\n## ",
    "\n### ",
    "\n\n",
    "\n",
    ". ",
    " ",
)


def _split_on(text: str, sep: str) -> list[str]:
    if not sep:
        return list(text)
    parts = text.split(sep)
    # Re-attach the separator so headings and sentence punctuation survive.
    return [parts[0]] + [sep + p for p in parts[1:]]


def _recursive_split(text: str, max_chars: int, seps: tuple[str, ...]) -> list[str]:
    if len(text) <= max_chars:
        return [text] if text.strip() else []

    if not seps:
        # No separator worked; hard-cut as a last resort.
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

    sep, rest = seps[0], seps[1:]
    out: list[str] = []
    buffer = ""

    for piece in _split_on(text, sep):
        if len(buffer) + len(piece) <= max_chars:
            buffer += piece
            continue
        if buffer.strip():
            out.append(buffer)
        buffer = ""
        if len(piece) > max_chars:
            out.extend(_recursive_split(piece, max_chars, rest))
        else:
            buffer = piece

    if buffer.strip():
        out.append(buffer)
    return out


def _apply_overlap(pieces: list[str], overlap: int) -> list[str]:
    """Prefix each chunk with the tail of the previous one.

    Overlap is what stops a fact that straddles a boundary from becoming
    unretrievable. The tail is trimmed to a word boundary so the prefix reads
    as language rather than a severed token.
    """
    if overlap <= 0 or len(pieces) < 2:
        return pieces

    out = [pieces[0]]
    # strict=False is intentional: the sliced tail is one shorter than `pieces`,
    # which is exactly the pairwise window we want.
    for prev, cur in zip(pieces, pieces[1:], strict=False):
        tail = prev[-overlap:]
        cut = tail.find(" ")
        if cut != -1:
            tail = tail[cut + 1 :]
        out.append(f"{tail}{cur}" if tail else cur)
    return out


#: Fraction of `max_chars` used for overlap when the caller doesn't specify one.
DEFAULT_OVERLAP_RATIO = 0.15


def resolve_overlap(max_chars: int, overlap: int | None) -> int:
    """Scale the default overlap to the chunk size.

    A fixed default is a footgun: `chunk_document(doc, max_chars=120)` would
    inherit a 150-char overlap and fail. Defaulting to a ratio means the knob
    stays correct at any chunk size, while an explicitly bad pair still raises.
    """
    if overlap is None:
        return int(max_chars * DEFAULT_OVERLAP_RATIO)
    if overlap >= max_chars:
        raise ValueError(
            f"overlap must be smaller than max_chars (got {overlap} >= {max_chars})"
        )
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative (got {overlap})")
    return overlap


def chunk_document(
    doc: Document, max_chars: int = 1000, overlap: int | None = None
) -> list[Chunk]:
    overlap = resolve_overlap(max_chars, overlap)

    normalized = re.sub(r"\n{3,}", "\n\n", doc.text.strip())
    pieces = _apply_overlap(_recursive_split(normalized, max_chars, _SEPARATORS), overlap)

    return [
        Chunk(
            id=f"{doc.id}::{i}",
            doc_id=doc.id,
            text=piece.strip(),
            ordinal=i,
            metadata=dict(doc.metadata),
        )
        for i, piece in enumerate(pieces)
        if piece.strip()
    ]


def chunk_all(
    docs: Iterable[Document], max_chars: int = 1000, overlap: int | None = None
) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, max_chars=max_chars, overlap=overlap))
    return out
