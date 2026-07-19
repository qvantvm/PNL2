"""Dimensionality reduction helpers."""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

Method = Literal["pca", "tsne", "umap"]


def project(
    embeddings: np.ndarray,
    *,
    method: Method = "pca",
    n_components: int = 2,
    seed: int = 42,
    max_points: int = 1500,
    indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (coords, indices_used)."""
    if indices is None:
        n = embeddings.shape[0]
        if n > max_points:
            rng = np.random.default_rng(seed)
            indices = np.sort(rng.choice(n, size=max_points, replace=False))
        else:
            indices = np.arange(n)
    X = embeddings[indices]
    if method == "pca":
        coords = PCA(n_components=n_components, random_state=seed).fit_transform(X)
    elif method == "tsne":
        # PCA first for stability (cap by samples and features)
        n_pca = max(2, min(50, X.shape[0] - 1, X.shape[1]))
        X2 = PCA(n_components=n_pca, random_state=seed).fit_transform(X)
        perplexity = min(30, max(2, (len(X) - 1) // 3))
        coords = TSNE(
            n_components=n_components,
            random_state=seed,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
        ).fit_transform(X2)
    elif method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise ImportError("Install optional dependency: pip install -e '.[umap]'") from exc
        coords = umap.UMAP(n_components=n_components, random_state=seed).fit_transform(X)
    else:
        raise ValueError(method)
    return coords, indices
