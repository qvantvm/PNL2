"""Static Matplotlib visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pnl2vec.tokenizer.vocabulary import Vocabulary
from pnl2vec.visualization.projection import project

# Colorblind-friendly qualitative palette
COLORS = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#F0E442",
    "#000000",
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def _category_of(tok: str, vocabulary: Vocabulary) -> str:
    meta = vocabulary.metadata.get(tok)
    return meta.musical_category if meta else "other"


def plot_projection(
    embeddings: np.ndarray,
    vocabulary: Vocabulary,
    out_path: Path,
    *,
    title: str,
    method: str = "pca",
    filter_prefix: str | None = None,
    filter_category: str | None = None,
    max_points: int = 800,
    seed: int = 42,
) -> Path:
    ids = []
    for i in range(len(vocabulary)):
        tok = vocabulary.id_to_token(i)
        if vocabulary.metadata.get(tok, None) and vocabulary.metadata[tok].is_special:
            continue
        if filter_prefix and not tok.startswith(filter_prefix):
            continue
        if filter_category and _category_of(tok, vocabulary) != filter_category:
            continue
        ids.append(i)
    if not ids:
        # empty plot
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_title(title + " (no points)")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return out_path

    indices = np.array(ids)
    use_method = method
    if method in {"tsne", "umap"} and len(indices) < 10:
        use_method = "pca"
    coords, used = project(
        embeddings,
        method=use_method,  # type: ignore[arg-type]
        max_points=max_points,
        indices=indices,
        seed=seed,
    )
    cats = [_category_of(vocabulary.id_to_token(int(i)), vocabulary) for i in used]
    uniq = sorted(set(cats))
    cat_to_style = {
        c: (COLORS[i % len(COLORS)], MARKERS[i % len(MARKERS)]) for i, c in enumerate(uniq)
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    for c in uniq:
        mask = [cat == c for cat in cats]
        pts = coords[np.array(mask)]
        color, marker = cat_to_style[c]
        ax.scatter(pts[:, 0], pts[:, 1], c=color, marker=marker, s=28, label=c, alpha=0.85, edgecolors="none")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_before_after(
    before: np.ndarray,
    after: np.ndarray,
    vocabulary: Vocabulary,
    out_path: Path,
    *,
    seed: int = 42,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, emb, title in zip(axes, [before, after], ["Before training", "After training"]):
        coords, used = project(emb, method="pca", max_points=600, seed=seed)
        cats = [_category_of(vocabulary.id_to_token(int(i)), vocabulary) for i in used]
        uniq = sorted(set(cats))
        for i, c in enumerate(uniq):
            mask = np.array([cat == c for cat in cats])
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=COLORS[i % len(COLORS)],
                marker=MARKERS[i % len(MARKERS)],
                s=20,
                label=c,
                alpha=0.8,
            )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[1].legend(loc="best", fontsize=7, frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_static_suite(
    embeddings: np.ndarray,
    vocabulary: Vocabulary,
    directory: Path,
    *,
    before: np.ndarray | None = None,
    method: str = "pca",
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    jobs = [
        ("all_categories", None, None),
        ("pitch_class", "PITCH_CLASS:", "pitch"),
        ("octave_pitch", None, "pitch"),
        ("duration", "DURATION:", "duration"),
        ("dynamics", "DYNAMIC:", "dynamic"),
        ("articulations", "ARTICULATION:", "articulation"),
        ("fingering", "FINGER:", "fingering"),
        ("pedal", "PEDAL:", "pedal"),
        ("structural", "STRUCT:", "structural"),
        ("chord", "CHORD:", "chord"),
    ]
    for name, prefix, cat in jobs:
        path = directory / f"{method}_{name}.png"
        outputs.append(
            plot_projection(
                embeddings,
                vocabulary,
                path,
                title=f"{method.upper()}: {name}",
                method=method,
                filter_prefix=prefix,
                filter_category=cat if prefix is None else None,
            )
        )
    if before is not None:
        outputs.append(plot_before_after(before, embeddings, vocabulary, directory / "before_after_pca.png"))
    return outputs
