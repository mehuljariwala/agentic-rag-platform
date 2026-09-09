"""In-process vector store with optional metadata filtering.

Brute-force cosine over a numpy matrix. For the corpus sizes this project
targets (up to ~1e5 chunks) an exact scan is both faster and more predictable
than an ANN index, and it removes a heavyweight dependency. `VectorStore` is
deliberately swappable -- see docs/ARCHITECTURE.md for the pgvector notes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class VectorStore:
    dim: int
    ids: list[str] = field(default_factory=list)
    metadata: list[dict[str, Any]] = field(default_factory=list)
    _matrix: np.ndarray | None = None

    def add(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        metadata: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        if not ids:
            return
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise ValueError(
                f"expected vectors of shape (n, {self.dim}), got {arr.shape}"
            )

        # Normalise once at write time so query-time scoring is a plain dot
        # product rather than a division per row.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        arr = arr / norms

        self.ids.extend(ids)
        self.metadata.extend(metadata or [{} for _ in ids])
        self._matrix = arr if self._matrix is None else np.vstack([self._matrix, arr])

    def search(
        self,
        query: Sequence[float],
        k: int = 10,
        where: Callable[[dict[str, Any]], bool] | None = None,
    ) -> list[tuple[str, float]]:
        if self._matrix is None or not self.ids:
            return []

        q = np.asarray(query, dtype=np.float32)
        norm = float(np.linalg.norm(q))
        if norm == 0.0:
            return []
        q = q / norm

        scores = self._matrix @ q

        if where is not None:
            mask = np.array([where(m) for m in self.metadata], dtype=bool)
            if not mask.any():
                return []
            # -inf so filtered rows can never enter the top-k
            scores = np.where(mask, scores, -np.inf)

        k = min(k, len(self.ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.ids[i], float(scores[i])) for i in top if np.isfinite(scores[i])]

    def __len__(self) -> int:
        return len(self.ids)
