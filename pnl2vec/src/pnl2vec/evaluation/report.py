"""Evaluation report generation with baselines."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from pnl2vec.evaluation.analogies import analogy, default_analogy_suite
from pnl2vec.evaluation.probes import ProbeResult, run_standard_probes
from pnl2vec.evaluation.similarity import EmbeddingIndex, Neighbor
from pnl2vec.tokenizer.vocabulary import Vocabulary


@dataclass
class IntrinsicTest:
    name: str
    query: str
    expected: list[str]
    neighbors: list[str]
    precision_at_k: float
    mrr: float


@dataclass
class EvaluationReport:
    intrinsic: list[IntrinsicTest] = field(default_factory=list)
    analogies: list[dict] = field(default_factory=list)
    probes: list[dict] = field(default_factory=list)
    baselines: dict[str, dict] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _precision_mrr(neighbors: list[str], expected: set[str]) -> tuple[float, float]:
    if not expected:
        return 0.0, 0.0
    hits = sum(1 for n in neighbors if n in expected)
    precision = hits / max(1, len(neighbors))
    mrr = 0.0
    for i, n in enumerate(neighbors, start=1):
        if n in expected:
            mrr = 1.0 / i
            break
    return precision, mrr


def expected_neighbor_sets(vocabulary: Vocabulary) -> list[tuple[str, list[str]]]:
    tests: list[tuple[str, list[str]]] = []
    durs = sorted([t for t in vocabulary.token_to_id_ if t.startswith("DURATION:")])
    if "DURATION:1/4" in vocabulary.token_to_id_:
        expected = [d for d in durs if d != "DURATION:1/4"][:4]
        tests.append(("DURATION:1/4", expected))
    if "PITCH_CLASS:C" in vocabulary.token_to_id_:
        expected = [t for t in vocabulary.token_to_id_ if t.startswith("PITCH_CLASS:") and t != "PITCH_CLASS:C"]
        tests.append(("PITCH_CLASS:C", expected[:6]))
    if "PEDAL:DOWN" in vocabulary.token_to_id_ and "PEDAL:UP" in vocabulary.token_to_id_:
        tests.append(("PEDAL:DOWN", ["PEDAL:UP"]))
    if "DYNAMIC:MF" in vocabulary.token_to_id_:
        dyns = [t for t in vocabulary.token_to_id_ if t.startswith("DYNAMIC:") and t != "DYNAMIC:MF"]
        tests.append(("DYNAMIC:MF", dyns))
    if "ARTICULATION:STACCATO" in vocabulary.token_to_id_:
        arts = [t for t in vocabulary.token_to_id_ if t.startswith("ARTICULATION:") and t != "ARTICULATION:STACCATO"]
        tests.append(("ARTICULATION:STACCATO", arts))
    return tests


def feature_baseline_matrix(vocabulary: Vocabulary, dim: int) -> np.ndarray:
    """Hand-crafted musical feature vectors (hashed into dim)."""
    mat = np.zeros((len(vocabulary), dim), dtype=np.float64)
    for canon, entry in vocabulary.metadata.items():
        v = np.zeros(dim)
        # simple feature hashing
        for feat in (entry.kind, entry.musical_category, str(entry.value)):
            h = hash(feat) % dim
            v[h] += 1.0
        if canon.startswith("PITCH_CLASS:"):
            pc = "CDEFGAB".find(str(entry.value)[0]) if entry.value else -1
            if pc >= 0:
                v[pc % dim] += 2.0
        if canon.startswith("OCTAVE:") and entry.value is not None:
            v[int(entry.value) % dim] += 1.5
        n = np.linalg.norm(v)
        mat[entry.id] = v / n if n > 0 else v
    return mat


def run_evaluation(
    embeddings: np.ndarray,
    vocabulary: Vocabulary,
    *,
    untrained: np.ndarray | None = None,
    top_k: int = 10,
) -> EvaluationReport:
    report = EvaluationReport()
    index = EmbeddingIndex(embeddings, vocabulary)
    for query, expected in expected_neighbor_sets(vocabulary):
        neigh = index.nearest_neighbors(query, top_k=top_k)
        names = [n.token for n in neigh]
        prec, mrr = _precision_mrr(names, set(expected))
        report.intrinsic.append(
            IntrinsicTest(
                name=f"nn:{query}",
                query=query,
                expected=expected,
                neighbors=names,
                precision_at_k=prec,
                mrr=mrr,
            )
        )

    for a, b, c in default_analogy_suite(vocabulary):
        res = analogy(index, a, b, c, top_k=top_k)
        report.analogies.append(
            {"query": res.query, "neighbors": res.neighbors}
        )

    probes = run_standard_probes(embeddings, vocabulary)
    report.probes = [asdict(p) for p in probes]

    # Baselines: random, untrained, feature
    rng = np.random.default_rng(0)
    random_emb = rng.normal(size=embeddings.shape)
    feature_emb = feature_baseline_matrix(vocabulary, embeddings.shape[1])
    baseline_mats = {
        "learned": embeddings,
        "random": random_emb,
        "feature": feature_emb,
    }
    if untrained is not None:
        baseline_mats["untrained"] = untrained

    for name, mat in baseline_mats.items():
        idx = EmbeddingIndex(mat, vocabulary)
        scores = []
        for query, expected in expected_neighbor_sets(vocabulary):
            neigh = [n.token for n in idx.nearest_neighbors(query, top_k=top_k)]
            prec, mrr = _precision_mrr(neigh, set(expected))
            scores.append({"query": query, "precision_at_k": prec, "mrr": mrr})
        report.baselines[name] = {
            "mean_precision_at_k": float(np.mean([s["precision_at_k"] for s in scores])) if scores else 0.0,
            "mean_mrr": float(np.mean([s["mrr"] for s in scores])) if scores else 0.0,
            "details": scores,
        }

    report.notes.append(
        "Intrinsic expected sets are soft musical heuristics; not all relationships must emerge from every corpus."
    )
    report.notes.append(
        "Namespace probes can be partly trivial because labels are encoded in token identity."
    )
    return report


def save_evaluation_report(report: EvaluationReport, directory: Path | str) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "intrinsic": [asdict(x) for x in report.intrinsic],
        "analogies": report.analogies,
        "probes": report.probes,
        "baselines": report.baselines,
        "notes": report.notes,
    }
    (directory / "evaluation_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# Evaluation Report", "", "## Baselines"]
    for name, stats in report.baselines.items():
        lines.append(
            f"- **{name}**: mean P@{len(report.intrinsic) and 10}={stats['mean_precision_at_k']:.3f}, "
            f"MRR={stats['mean_mrr']:.3f}"
        )
    lines.append("")
    lines.append("## Intrinsic nearest neighbors")
    for t in report.intrinsic:
        lines.append(f"- {t.query}: P@k={t.precision_at_k:.3f} MRR={t.mrr:.3f} → {t.neighbors[:5]}")
    lines.append("")
    lines.append("## Notes")
    for n in report.notes:
        lines.append(f"- {n}")
    (directory / "evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
