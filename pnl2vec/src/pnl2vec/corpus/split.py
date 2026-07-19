"""Document-level corpus splits."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1, got {total}")


@dataclass
class CorpusSplit:
    train: list
    val: list
    test: list


def split_documents(
    items: Sequence[T],
    *,
    seed: int = 42,
    ratios: SplitRatios | None = None,
) -> CorpusSplit:
    """Split by complete documents (caller passes one item per document)."""
    ratios = ratios or SplitRatios()
    items = list(items)
    rng = random.Random(seed)
    rng.shuffle(items)
    n = len(items)
    if n == 0:
        return CorpusSplit([], [], [])
    n_train = max(1, int(n * ratios.train)) if n >= 3 else max(1, n - 2)
    n_val = int(n * ratios.val) if n >= 3 else (1 if n >= 2 else 0)
    # Ensure test gets remainder; keep at least one train
    if n_train + n_val >= n and n > 1:
        n_val = max(0, n - n_train - 1)
    n_train = min(n_train, n - n_val)
    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val :]
    if not test and n >= 3:
        test = val[-1:]
        val = val[:-1]
    return CorpusSplit(train=train, val=val, test=test)
