"""Public API for trained PNL/2 embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import yaml

from pnl2vec.evaluation.retrieval import PhraseIndex, PhraseRecord, SearchHit, summarize_pnl
from pnl2vec.evaluation.similarity import EmbeddingIndex, Neighbor, cosine_similarity, l2_normalize
from pnl2vec.pnl import parse_pnl
from pnl2vec.tokenizer import Tokenizer, TokenizerConfig, Vocabulary
from pnl2vec.tokenizer.serialization import load_tokenizer_config
from pnl2vec.tokenizer.token import Token, TokenKind

Pooling = Literal["mean", "freq", "inv_freq", "sif", "mean_remove_pc"]


@dataclass
class EmbedResult:
    vector: np.ndarray
    token_ids: list[int]
    tokens: list[str]


class PNL2Vec:
    def __init__(
        self,
        vocabulary: Vocabulary,
        embeddings: np.ndarray,
        tokenizer: Tokenizer,
        *,
        frequencies: dict[str, int] | None = None,
        artifacts_dir: Path | None = None,
    ) -> None:
        self.vocabulary = vocabulary
        self.embeddings = embeddings
        self.tokenizer = tokenizer
        self.frequencies = frequencies or {
            k: e.frequency for k, e in vocabulary.metadata.items()
        }
        self.artifacts_dir = artifacts_dir
        self.index = EmbeddingIndex(embeddings, vocabulary)
        self._pc1: np.ndarray | None = None

    @classmethod
    def load(
        cls,
        artifacts: Path | str,
        *,
        which: Literal["combined", "input", "output"] = "combined",
    ) -> PNL2Vec:
        artifacts = Path(artifacts)
        vocab = Vocabulary.load(artifacts / "tokenizer" / "vocabulary.json")
        cfg = load_tokenizer_config(artifacts / "tokenizer" / "config.yaml")
        tokenizer = Tokenizer(cfg)
        name = {
            "combined": "combined_embeddings.npy",
            "input": "input_embeddings.npy",
            "output": "output_embeddings.npy",
        }[which]
        emb = np.load(artifacts / "embeddings" / name)
        return cls(vocab, emb, tokenizer, artifacts_dir=artifacts)

    def tokenize(self, pnl_text: str) -> list[Token]:
        return self.tokenizer.tokenize_text(pnl_text)

    def encode(self, pnl_text: str) -> list[int]:
        tokens = self.tokenize(pnl_text)
        strings = self.tokenizer.to_canonical_strings(tokens)
        return self.vocabulary.encode_strings(strings)

    def embedding_for_token(self, token: str) -> np.ndarray:
        return self.embeddings[self.vocabulary.token_to_id(token)].copy()

    def similarity(self, a: str, b: str) -> float:
        return self.index.similarity(a, b)

    def nearest_neighbors(
        self,
        token: str,
        *,
        top_k: int = 10,
        category: str | None = None,
    ) -> list[Neighbor]:
        return self.index.nearest_neighbors(token, top_k=top_k, category=category)

    def embed_tokens(
        self,
        ids: Sequence[int],
        *,
        pooling: Pooling = "mean",
        ignore_structural: bool = True,
    ) -> np.ndarray:
        filtered = []
        for i in ids:
            tok = self.vocabulary.id_to_token(i)
            meta = self.vocabulary.metadata.get(tok)
            if ignore_structural and meta and meta.musical_category in {"special", "structural"}:
                continue
            if ignore_structural and tok.startswith("<"):
                continue
            filtered.append(i)
        if not filtered:
            filtered = list(ids)
        vectors = self.embeddings[np.array(filtered, dtype=np.int64)]
        if pooling == "mean":
            return vectors.mean(axis=0)
        if pooling == "freq":
            weights = np.array(
                [max(self.frequencies.get(self.vocabulary.id_to_token(i), 1), 1) for i in filtered],
                dtype=np.float64,
            )
            weights = weights / weights.sum()
            return (vectors * weights[:, None]).sum(axis=0)
        if pooling == "inv_freq":
            weights = np.array(
                [1.0 / max(self.frequencies.get(self.vocabulary.id_to_token(i), 1), 1) for i in filtered],
                dtype=np.float64,
            )
            weights = weights / weights.sum()
            return (vectors * weights[:, None]).sum(axis=0)
        if pooling == "sif":
            a = 1e-3
            weights = np.array(
                [
                    a / (a + self.frequencies.get(self.vocabulary.id_to_token(i), 1))
                    for i in filtered
                ],
                dtype=np.float64,
            )
            weights = weights / weights.sum()
            vec = (vectors * weights[:, None]).sum(axis=0)
            return self._remove_pc(vec.reshape(1, -1))[0]
        if pooling == "mean_remove_pc":
            vec = vectors.mean(axis=0)
            return self._remove_pc(vec.reshape(1, -1))[0]
        raise ValueError(pooling)

    def embed_pnl(self, text: str, pooling: Pooling = "mean") -> np.ndarray:
        ids = self.encode(text)
        return self.embed_tokens(ids, pooling=pooling)

    def embed_events(self, events_tokens: Sequence[Sequence[int]], pooling: Pooling = "mean") -> np.ndarray:
        vecs = [self.embed_tokens(e, pooling=pooling) for e in events_tokens]
        return np.stack(vecs, axis=0).mean(axis=0)

    def _remove_pc(self, matrix: np.ndarray) -> np.ndarray:
        if self._pc1 is None:
            # estimate from embedding matrix
            X = self.embeddings - self.embeddings.mean(axis=0, keepdims=True)
            _, _, vt = np.linalg.svd(X, full_matrices=False)
            self._pc1 = vt[0]
        pc = self._pc1
        return matrix - np.outer(matrix @ pc, pc)

    def search_similar_phrases(
        self,
        pnl_text: str,
        *,
        index: PhraseIndex | Path | str | None = None,
        top_k: int = 10,
        pooling: Pooling = "mean",
    ) -> list[SearchHit]:
        if index is None:
            if self.artifacts_dir is None:
                raise ValueError("no phrase index provided")
            index = self.artifacts_dir / "phrase_index"
        if not isinstance(index, PhraseIndex):
            index = PhraseIndex.load(index)
        q = self.embed_pnl(pnl_text, pooling=pooling)
        return index.search(q, top_k=top_k)

    def build_phrase_index(
        self,
        corpus_dir: Path | str,
        output: Path | str,
        *,
        pooling: Pooling = "mean",
    ) -> PhraseIndex:
        from pnl2vec.corpus import load_corpus

        corpus_dir = Path(corpus_dir)
        docs = load_corpus(corpus_dir)
        records = []
        vectors = []
        for d in docs:
            if d.parse_error:
                continue
            family = None
            meta = d.document.score.meta
            if isinstance(meta, dict):
                family = meta.get("family")
            vec = self.embed_pnl(d.text, pooling=pooling)
            vectors.append(vec)
            records.append(
                PhraseRecord(
                    phrase_id=d.doc_id,
                    source_path=str(d.path),
                    pnl_text=d.text,
                    summary=summarize_pnl(d.text),
                    family=str(family) if family is not None else None,
                )
            )
        idx = PhraseIndex()
        if vectors:
            idx.add(np.stack(vectors), records)
        idx.save(output)
        return idx
