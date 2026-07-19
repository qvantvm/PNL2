"""Negative sampling distribution."""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch


class NegativeSampler:
    def __init__(
        self,
        frequencies: Counter[str] | dict[int, int],
        vocab_size: int,
        *,
        power: float = 0.75,
        exclude_ids: set[int] | None = None,
        seed: int = 42,
    ) -> None:
        self.exclude_ids = exclude_ids or set()
        weights = np.zeros(vocab_size, dtype=np.float64)
        if frequencies and isinstance(next(iter(frequencies.keys())), str):
            # string keyed — cannot map here; expect id frequencies
            raise TypeError("pass id->frequency mapping")
        for tid, freq in frequencies.items():  # type: ignore[union-attr]
            tid = int(tid)
            if tid in self.exclude_ids:
                continue
            weights[tid] = float(freq) ** power
        total = weights.sum()
        if total <= 0:
            weights = np.ones(vocab_size, dtype=np.float64)
            for eid in self.exclude_ids:
                if 0 <= eid < vocab_size:
                    weights[eid] = 0.0
            total = weights.sum()
        self.probs = weights / total
        self.vocab_size = vocab_size
        self._rng = np.random.default_rng(seed)

    def sample(self, n: int, *, batch_size: int = 1) -> torch.Tensor:
        ids = self._rng.choice(self.vocab_size, size=(batch_size, n), p=self.probs)
        return torch.as_tensor(ids, dtype=torch.long)

    @classmethod
    def from_vocabulary(
        cls,
        vocabulary,
        *,
        power: float = 0.75,
        seed: int = 42,
    ) -> NegativeSampler:
        freq: dict[int, int] = {}
        exclude = set()
        for canon, entry in vocabulary.metadata.items():
            if entry.is_special or canon in {
                "<PAD>",
                "<UNK>",
                "<BOS>",
                "<EOS>",
                "<MASK>",
                "<DOC_SEP>",
                "<MEASURE_SEP>",
            }:
                exclude.add(entry.id)
            freq[entry.id] = max(entry.frequency, 1)
        # also exclude structural DOC boundaries by name pattern
        for canon, entry in vocabulary.metadata.items():
            if canon.startswith("STRUCT:DOC_") or canon.startswith("STRUCT:PART_"):
                exclude.add(entry.id)
        return cls(freq, len(vocabulary), power=power, exclude_ids=exclude, seed=seed)
