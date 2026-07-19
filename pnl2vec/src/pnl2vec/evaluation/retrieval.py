"""Phrase retrieval index and evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from pnl2vec.evaluation.similarity import cosine_similarity, l2_normalize


class PhraseEmbedder(Protocol):
    def embed_pnl(self, text: str, pooling: str = "mean") -> np.ndarray: ...


@dataclass
class PhraseRecord:
    phrase_id: str
    source_path: str
    pnl_text: str
    summary: str
    family: str | None = None


@dataclass
class SearchHit:
    phrase_id: str
    score: float
    pnl_text: str
    summary: str
    source_path: str


class PhraseIndex:
    """NumPy brute-force index (FAISS-ready interface)."""

    def __init__(self) -> None:
        self.records: list[PhraseRecord] = []
        self.matrix: np.ndarray | None = None

    def add(self, vectors: np.ndarray, records: Sequence[PhraseRecord]) -> None:
        vectors = l2_normalize(np.asarray(vectors, dtype=np.float64))
        if self.matrix is None:
            self.matrix = vectors
            self.records = list(records)
        else:
            self.matrix = np.vstack([self.matrix, vectors])
            self.records.extend(records)

    def search(self, query: np.ndarray, *, top_k: int = 10) -> list[SearchHit]:
        if self.matrix is None or not self.records:
            return []
        q = l2_normalize(query.reshape(1, -1))[0]
        scores = self.matrix @ q
        order = np.argsort(-scores)[:top_k]
        hits = []
        for i in order:
            r = self.records[int(i)]
            hits.append(
                SearchHit(
                    phrase_id=r.phrase_id,
                    score=float(scores[i]),
                    pnl_text=r.pnl_text,
                    summary=r.summary,
                    source_path=r.source_path,
                )
            )
        return hits

    def save(self, directory: Path | str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self.matrix is not None:
            np.save(directory / "vectors.npy", self.matrix)
        payload = [
            {
                "phrase_id": r.phrase_id,
                "source_path": r.source_path,
                "pnl_text": r.pnl_text,
                "summary": r.summary,
                "family": r.family,
            }
            for r in self.records
        ]
        (directory / "records.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path | str) -> PhraseIndex:
        directory = Path(directory)
        idx = cls()
        records = json.loads((directory / "records.json").read_text(encoding="utf-8"))
        idx.records = [PhraseRecord(**r) for r in records]
        vec_path = directory / "vectors.npy"
        if vec_path.exists():
            idx.matrix = np.load(vec_path)
        return idx


def summarize_pnl(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    notes = [ln for ln in lines if ln.startswith("note ")]
    return f"{len(notes)} note lines; " + "; ".join(notes[:4])


def evaluate_family_retrieval(
    index: PhraseIndex,
    embedder: PhraseEmbedder,
    queries: Sequence[tuple[str, str]],
    *,
    top_k: int = 5,
) -> dict[str, float]:
    """queries: list of (pnl_text, family)."""
    recalls = []
    mrrs = []
    for text, family in queries:
        vec = embedder.embed_pnl(text)
        hits = index.search(vec, top_k=top_k)
        ranks = [i + 1 for i, h in enumerate(hits) if _family_of(index, h.phrase_id) == family]
        if ranks:
            recalls.append(1.0)
            mrrs.append(1.0 / ranks[0])
        else:
            recalls.append(0.0)
            mrrs.append(0.0)
    return {
        "recall@k": float(np.mean(recalls)) if recalls else 0.0,
        "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
        "n": len(queries),
    }


def _family_of(index: PhraseIndex, phrase_id: str) -> str | None:
    for r in index.records:
        if r.phrase_id == phrase_id:
            return r.family
    return None
