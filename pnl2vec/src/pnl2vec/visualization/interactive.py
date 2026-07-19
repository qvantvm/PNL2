"""Interactive Plotly visualizations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pnl2vec.evaluation.similarity import EmbeddingIndex
from pnl2vec.tokenizer.vocabulary import Vocabulary
from pnl2vec.visualization.projection import project


def interactive_scatter(
    embeddings: np.ndarray,
    vocabulary: Vocabulary,
    out_path: Path,
    *,
    method: str = "pca",
    max_points: int = 1000,
    top_neighbors: int = 5,
) -> Path:
    coords, used = project(embeddings, method=method, max_points=max_points)  # type: ignore[arg-type]
    index = EmbeddingIndex(embeddings, vocabulary)
    tokens = [vocabulary.id_to_token(int(i)) for i in used]
    cats = []
    freqs = []
    neighbor_text = []
    for tok in tokens:
        meta = vocabulary.metadata.get(tok)
        cats.append(meta.musical_category if meta else "other")
        freqs.append(meta.frequency if meta else 0)
        try:
            nn = index.nearest_neighbors(tok, top_k=top_neighbors)
            neighbor_text.append(", ".join(f"{n.token}:{n.similarity:.2f}" for n in nn))
        except Exception:
            neighbor_text.append("")

    ids = [vocabulary.token_to_id(t) for t in tokens]
    fig = px.scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        color=cats,
        symbol=cats,
        hover_name=tokens,
        custom_data=[ids, cats, freqs, neighbor_text],
        title=f"Interactive {method.upper()} embedding space",
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "id=%{customdata[0]}<br>"
            "category=%{customdata[1]}<br>"
            "frequency=%{customdata[2]}<br>"
            "neighbors=%{customdata[3]}<extra></extra>"
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))
    return out_path


def neighbor_graph_html(
    embeddings: np.ndarray,
    vocabulary: Vocabulary,
    out_path: Path,
    *,
    top_tokens: int = 40,
    top_k: int = 3,
) -> Path:
    """High-frequency nearest-neighbor graph as a simple Plotly figure."""
    index = EmbeddingIndex(embeddings, vocabulary)
    items = sorted(
        [(e.frequency, e.canonical) for e in vocabulary.metadata.values() if not e.is_special],
        reverse=True,
    )[:top_tokens]
    tokens = [t for _, t in items]
    # layout in a circle
    angles = np.linspace(0, 2 * np.pi, len(tokens), endpoint=False)
    xs = np.cos(angles)
    ys = np.sin(angles)
    pos = {t: (xs[i], ys[i]) for i, t in enumerate(tokens)}
    edge_x = []
    edge_y = []
    for t in tokens:
        for n in index.nearest_neighbors(t, top_k=top_k):
            if n.token not in pos:
                continue
            x0, y0 = pos[t]
            x1, y1 = pos[n.token]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.5, color="#888"), hoverinfo="none"))
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            text=tokens,
            textposition="top center",
            marker=dict(size=9, color="#0072B2"),
        )
    )
    fig.update_layout(title="High-frequency nearest-neighbor graph", showlegend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))
    return out_path
