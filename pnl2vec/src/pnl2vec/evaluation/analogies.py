"""Vector arithmetic analogy tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pnl2vec.evaluation.similarity import EmbeddingIndex
from pnl2vec.tokenizer.vocabulary import Vocabulary


@dataclass
class AnalogyResult:
    query: str
    a: str
    b: str
    c: str
    neighbors: list[tuple[str, float]]


def analogy(
    index: EmbeddingIndex,
    a: str,
    b: str,
    c: str,
    *,
    top_k: int = 10,
) -> AnalogyResult:
    """a - b + c ≈ ?  (or commonly a:b :: c:?) — we use c - a + b style? 

    Prompt examples: PITCH:C4 - OCTAVE:4 + OCTAVE:5 ≈ PITCH:C5
    So result ≈ a - b + c with a=PITCH:C4, b=OCTAVE:4, c=OCTAVE:5.
    """
    va = index.vector(a)
    vb = index.vector(b)
    vc = index.vector(c)
    target = va - vb + vc
    target = target / max(np.linalg.norm(target), 1e-12)
    scores = index.matrix @ target
    for tok in (a, b, c):
        scores[index.vocabulary.token_to_id(tok)] = -np.inf
    order = np.argsort(-scores)[:top_k]
    neighbors = [(index.vocabulary.id_to_token(int(i)), float(scores[i])) for i in order]
    return AnalogyResult(query=f"{a} - {b} + {c}", a=a, b=b, c=c, neighbors=neighbors)


def default_analogy_suite(vocabulary: Vocabulary) -> list[tuple[str, str, str]]:
    suite: list[tuple[str, str, str]] = []
    # Duration adjacency-style
    if "DURATION:1/4" in vocabulary.token_to_id_ and "DURATION:1/8" in vocabulary.token_to_id_:
        if "DURATION:1/2" in vocabulary.token_to_id_:
            suite.append(("DURATION:1/4", "DURATION:1/8", "DURATION:1/2"))
    # Dynamics
    if all(t in vocabulary.token_to_id_ for t in ("DYNAMIC:MF", "DYNAMIC:F", "DYNAMIC:P")):
        suite.append(("DYNAMIC:MF", "DYNAMIC:F", "DYNAMIC:P"))
    # Octave shift via atomic tokens
    if all(t in vocabulary.token_to_id_ for t in ("OCTAVE:4", "OCTAVE:5", "PITCH_CLASS:C")):
        suite.append(("PITCH_CLASS:C", "OCTAVE:4", "OCTAVE:5"))
    return suite
