"""Cosine similarity and nearest-neighbor search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from pnl2vec.tokenizer.vocabulary import Vocabulary


@dataclass(frozen=True)
class Neighbor:
    token: str
    similarity: float
    frequency: int
    category: str


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return matrix / norms


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


class EmbeddingIndex:
    def __init__(
        self,
        embeddings: np.ndarray,
        vocabulary: Vocabulary,
        *,
        normalize: bool = True,
    ) -> None:
        self.vocabulary = vocabulary
        self.raw = embeddings
        self.matrix = l2_normalize(embeddings) if normalize else embeddings

    def vector(self, token: str) -> np.ndarray:
        tid = self.vocabulary.token_to_id(token)
        return self.matrix[tid]

    def nearest_neighbors(
        self,
        token: str,
        *,
        top_k: int = 10,
        category: str | None = None,
    ) -> list[Neighbor]:
        if token not in self.vocabulary.token_to_id_ and self.vocabulary.token_to_id(token) == self.vocabulary.unk_id:
            # still allow query via UNK row, but prefer missing
            pass
        q = self.vector(token)
        scores = self.matrix @ q
        query_id = self.vocabulary.token_to_id(token)
        scores[query_id] = -np.inf  # exclude self
        order = np.argsort(-scores)
        out: list[Neighbor] = []
        for idx in order:
            if len(out) >= top_k:
                break
            if not np.isfinite(scores[idx]):
                continue
            tok = self.vocabulary.id_to_token(int(idx))
            meta = self.vocabulary.metadata.get(tok)
            cat = meta.musical_category if meta else "other"
            if category is not None and cat != category and not tok.startswith(category.upper()):
                # also allow prefix filter like pitch -> PITCH_
                if category.lower() not in cat and not tok.upper().startswith(category.upper()):
                    continue
            freq = meta.frequency if meta else 0
            out.append(
                Neighbor(
                    token=tok,
                    similarity=float(scores[idx]),
                    frequency=freq,
                    category=cat,
                )
            )
        return out

    def similarity(self, a: str, b: str) -> float:
        return cosine_similarity(self.vector(a), self.vector(b))
