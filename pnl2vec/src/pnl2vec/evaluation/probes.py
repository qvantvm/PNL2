"""Linear probes on frozen embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from pnl2vec.tokenizer.vocabulary import Vocabulary


@dataclass
class ProbeResult:
    name: str
    accuracy: float
    macro_f1: float
    n_classes: int
    note: str = ""


def _labels_from_vocab(vocabulary: Vocabulary, kind_prefix: str) -> tuple[np.ndarray, np.ndarray]:
    ids = []
    labels = []
    for canon, entry in vocabulary.metadata.items():
        if entry.is_special:
            continue
        if canon.startswith(kind_prefix):
            ids.append(entry.id)
            labels.append(canon.split(":", 1)[1] if ":" in canon else canon)
    return np.array(ids), np.array(labels)


def run_probe(
    embeddings: np.ndarray,
    token_ids: np.ndarray,
    labels: np.ndarray,
    *,
    name: str,
    note: str = "",
    seed: int = 42,
) -> ProbeResult:
    if len(np.unique(labels)) < 2 or len(labels) < 8:
        return ProbeResult(name=name, accuracy=0.0, macro_f1=0.0, n_classes=len(np.unique(labels)), note="insufficient data")
    X = embeddings[token_ids]
    # Drop classes with a single member so stratification/split stays valid
    unique, counts = np.unique(labels, return_counts=True)
    keep = set(unique[counts >= 2])
    mask = np.array([lab in keep for lab in labels])
    X, labels = X[mask], labels[mask]
    if len(np.unique(labels)) < 2 or len(labels) < 8:
        return ProbeResult(name=name, accuracy=0.0, macro_f1=0.0, n_classes=len(np.unique(labels)), note="insufficient data after filtering")
    _, counts2 = np.unique(labels, return_counts=True)
    can_stratify = bool(len(labels) > 20 and counts2.min() >= 2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.25, random_state=seed, stratify=labels if can_stratify else None
    )
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    return ProbeResult(
        name=name,
        accuracy=float(accuracy_score(y_test, pred)),
        macro_f1=float(f1_score(y_test, pred, average="macro")),
        n_classes=len(np.unique(labels)),
        note=note,
    )


def category_probe(embeddings: np.ndarray, vocabulary: Vocabulary) -> ProbeResult:
    ids = []
    labels = []
    for canon, entry in vocabulary.metadata.items():
        if entry.is_special:
            continue
        ids.append(entry.id)
        labels.append(entry.musical_category)
    return run_probe(
        embeddings,
        np.array(ids),
        np.array(labels),
        name="token_category",
        note="Label is category metadata; partly encoded in token namespace (semi-trivial).",
    )


def run_standard_probes(embeddings: np.ndarray, vocabulary: Vocabulary) -> list[ProbeResult]:
    results = [category_probe(embeddings, vocabulary)]
    for prefix, name in [
        ("PITCH_CLASS:", "pitch_class"),
        ("OCTAVE:", "octave"),
        ("DURATION:", "duration"),
        ("HAND:", "hand"),
        ("ARTICULATION:", "articulation"),
        ("DYNAMIC:", "dynamic"),
    ]:
        ids, labels = _labels_from_vocab(vocabulary, prefix)
        results.append(
            run_probe(
                embeddings,
                ids,
                labels,
                name=name,
                note="Trivial if predicting the token's own namespace value from its one-hot-like identity; useful as sanity check.",
            )
        )
    return results
