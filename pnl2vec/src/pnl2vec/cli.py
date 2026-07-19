"""Unified CLI for pnl2vec."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import typer

app = typer.Typer(
    name="pnl2vec",
    help="PNL/2 tokenization and embedding toolkit",
    add_completion=False,
    no_args_is_help=True,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("pnl2vec")


def _die(msg: str, code: int = 1) -> None:
    typer.secho(msg, fg=typer.colors.RED, err=True)
    raise typer.Exit(code)


def _ensure_writable(path: Path, force: bool) -> None:
    if path.exists() and not force:
        _die(f"{path} exists; pass --force to overwrite")


@app.command("validate")
def validate_cmd(
    path: Path = typer.Argument(..., exists=True, readable=True),
) -> None:
    """Validate a PNL/2 file or directory."""
    from pnl2vec.pnl import PNLParseError, parse_pnl, validate_pnl

    paths = [path] if path.is_file() else sorted(path.glob("**/*.pnl"))
    errors = 0
    for p in paths:
        text = p.read_text(encoding="utf-8")
        try:
            doc = parse_pnl(text, filename=p)
        except PNLParseError as exc:
            typer.echo(f"FAIL {p}: {exc.issue.message}")
            errors += 1
            continue
        issues = validate_pnl(doc, filename=p)
        if issues:
            for i in issues:
                typer.echo(f"WARN {p}: {i.message}")
        else:
            typer.echo(f"OK {p}")
    if errors:
        raise typer.Exit(1)


@app.command("generate-synthetic")
def generate_synthetic(
    size: str = typer.Option("tiny", "--size", help="tiny|small|medium"),
    output: Path = typer.Option(Path("data/raw"), "--output"),
    seed: int = typer.Option(42, "--seed"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Generate a synthetic PNL/2 corpus."""
    from pnl2vec.corpus import generate_corpus

    if output.exists() and any(output.glob("*.pnl")) and not force:
        _die(f"{output} already contains .pnl files; pass --force")
    paths = generate_corpus(size, seed=seed, output_dir=output, force=force)
    typer.echo(f"Wrote {len(paths)} documents to {output}")


@app.command("inspect-corpus")
def inspect_corpus(
    path: Path = typer.Argument(Path("data/raw")),
    output: Path = typer.Option(Path("artifacts/reports"), "--output"),
) -> None:
    """Compute corpus statistics."""
    from pnl2vec.corpus import build_corpus_report, load_corpus, save_report
    from pnl2vec.tokenizer import Tokenizer

    corpus = load_corpus(path)
    if not corpus:
        _die(f"no documents in {path}")
    tok = Tokenizer()
    report = build_corpus_report(corpus, tok)
    save_report(report, output)
    typer.echo(report.to_markdown())
    typer.echo(f"Saved report to {output}")


@app.command("build-vocab")
def build_vocab(
    path: Path = typer.Argument(Path("data/raw")),
    output: Path = typer.Option(Path("artifacts/tokenizer"), "--output"),
    config: Optional[Path] = typer.Option(None, "--config"),
    seed: int = typer.Option(42, "--seed"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Build vocabulary from the training split only."""
    from pnl2vec.corpus import load_corpus, split_documents
    from pnl2vec.tokenizer.serialization import load_tokenizer_config, save_tokenizer_artifacts
    from pnl2vec.tokenizer.tokenizer import Tokenizer
    from pnl2vec.training.trainer import build_vocabulary

    if (output / "vocabulary.json").exists() and not force:
        _die(f"{output}/vocabulary.json exists; pass --force")
    corpus = load_corpus(path)
    split = split_documents(corpus, seed=seed)
    cfg = load_tokenizer_config(config)
    tokenizer = Tokenizer(cfg)
    vocab = build_vocabulary(split.train, tokenizer)
    save_tokenizer_artifacts(vocab, cfg, output)
    typer.echo(f"Vocabulary size={len(vocab)} written to {output}")


@app.command("train")
def train_cmd(
    config: Path = typer.Option(Path("configs/train_skipgram.yaml"), "--config"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Train skip-gram or CBOW embeddings."""
    from pnl2vec.training import train_from_config

    if not config.exists():
        _die(f"config not found: {config}")
    try:
        result = train_from_config(config, force=force)
    except Exception as exc:
        _die(f"training failed: {exc}")
    typer.echo(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
    if result.get("history"):
        typer.echo(f"epochs={len(result['history'])} best_val={result['best_val_loss']:.4f}")


@app.command("evaluate")
def evaluate_cmd(
    checkpoint: Path = typer.Option(Path("artifacts/checkpoints/best.pt"), "--checkpoint"),
    artifacts: Path = typer.Option(Path("artifacts"), "--artifacts"),
) -> None:
    """Run intrinsic evaluation and baselines."""
    from pnl2vec.evaluation import run_evaluation, save_evaluation_report
    from pnl2vec.tokenizer import Vocabulary

    emb_path = artifacts / "embeddings" / "combined_embeddings.npy"
    vocab_path = artifacts / "tokenizer" / "vocabulary.json"
    if not emb_path.exists() or not vocab_path.exists():
        _die("missing embeddings or vocabulary; train first")
    emb = np.load(emb_path)
    vocab = Vocabulary.load(vocab_path)
    # untrained baseline from checkpoint init if present
    untrained = None
    if checkpoint.exists():
        import torch

        from pnl2vec.training.checkpoint import load_checkpoint

        try:
            ckpt = load_checkpoint(checkpoint, map_location="cpu")
            # use random of same shape as proxy if we don't store init
            untrained = np.random.default_rng(0).normal(size=emb.shape).astype(np.float32)
            _ = ckpt
        except Exception:
            untrained = None
    report = run_evaluation(emb, vocab, untrained=untrained)
    save_evaluation_report(report, artifacts / "reports")
    typer.echo((artifacts / "reports" / "evaluation_report.md").read_text(encoding="utf-8"))


@app.command("visualize")
def visualize_cmd(
    checkpoint: Path = typer.Option(Path("artifacts/checkpoints/best.pt"), "--checkpoint"),
    artifacts: Path = typer.Option(Path("artifacts"), "--artifacts"),
    config: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Create static and interactive visualizations."""
    from pnl2vec.tokenizer import Vocabulary
    from pnl2vec.visualization import generate_static_suite, interactive_scatter, neighbor_graph_html

    emb = np.load(artifacts / "embeddings" / "combined_embeddings.npy")
    vocab = Vocabulary.load(artifacts / "tokenizer" / "vocabulary.json")
    out = artifacts / "visualizations"
    before = np.random.default_rng(0).normal(size=emb.shape).astype(np.float32)
    paths = generate_static_suite(emb, vocab, out, before=before, method="pca")
    try:
        paths.append(generate_static_suite(emb, vocab, out, method="tsne")[0])
    except Exception as exc:
        logger.warning("t-SNE skipped: %s", exc)
    html = interactive_scatter(emb, vocab, out / "interactive_pca.html", method="pca")
    graph = neighbor_graph_html(emb, vocab, out / "neighbor_graph.html")
    typer.echo(f"Wrote {len(paths)} static plots + {html.name} + {graph.name} to {out}")


@app.command("neighbors")
def neighbors_cmd(
    token: str = typer.Argument(...),
    top_k: int = typer.Option(10, "--top-k"),
    category: Optional[str] = typer.Option(None, "--category"),
    artifacts: Path = typer.Option(Path("artifacts"), "--artifacts"),
) -> None:
    """Nearest neighbors by cosine similarity."""
    from pnl2vec import PNL2Vec

    model = PNL2Vec.load(artifacts)
    results = model.nearest_neighbors(token, top_k=top_k, category=category)
    for n in results:
        typer.echo(f"{n.similarity:.4f}\t{n.token}\tfreq={n.frequency}\t{n.category}")


@app.command("analogy")
def analogy_cmd(
    token_a: str = typer.Argument(...),
    token_b: str = typer.Argument(...),
    token_c: str = typer.Argument(...),
    top_k: int = typer.Option(10, "--top-k"),
    artifacts: Path = typer.Option(Path("artifacts"), "--artifacts"),
) -> None:
    """Vector analogy: A - B + C."""
    from pnl2vec import PNL2Vec
    from pnl2vec.evaluation.analogies import analogy

    model = PNL2Vec.load(artifacts)
    res = analogy(model.index, token_a, token_b, token_c, top_k=top_k)
    typer.echo(res.query)
    for tok, score in res.neighbors:
        typer.echo(f"{score:.4f}\t{tok}")


@app.command("embed")
def embed_cmd(
    input_path: Path = typer.Argument(..., exists=True),
    output: Path = typer.Option(Path("vector.npy"), "--output"),
    artifacts: Path = typer.Option(Path("artifacts"), "--artifacts"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Embed a PNL file to a vector."""
    from pnl2vec import PNL2Vec

    if output.exists() and not force:
        _die(f"{output} exists; pass --force")
    model = PNL2Vec.load(artifacts)
    text = input_path.read_text(encoding="utf-8")
    vec = model.embed_pnl(text)
    np.save(output, vec)
    typer.echo(f"Wrote {output} shape={vec.shape}")


@app.command("index")
def index_cmd(
    corpus_path: Path = typer.Argument(...),
    output: Path = typer.Option(Path("artifacts/phrase_index"), "--output"),
    artifacts: Path = typer.Option(Path("artifacts"), "--artifacts"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Build a phrase retrieval index."""
    from pnl2vec import PNL2Vec

    if output.exists() and not force:
        _die(f"{output} exists; pass --force")
    model = PNL2Vec.load(artifacts)
    model.build_phrase_index(corpus_path, output)
    typer.echo(f"Indexed phrases → {output}")


@app.command("search")
def search_cmd(
    query: Path = typer.Argument(..., exists=True),
    index: Path = typer.Option(Path("artifacts/phrase_index"), "--index"),
    top_k: int = typer.Option(10, "--top-k"),
    artifacts: Path = typer.Option(Path("artifacts"), "--artifacts"),
) -> None:
    """Search similar phrases."""
    from pnl2vec import PNL2Vec

    model = PNL2Vec.load(artifacts)
    text = query.read_text(encoding="utf-8")
    hits = model.search_similar_phrases(text, index=index, top_k=top_k)
    for h in hits:
        typer.echo(f"{h.score:.4f}\t{h.phrase_id}\t{h.summary}")


@app.command("demo-classifier")
def demo_classifier(
    artifacts: Path = typer.Option(Path("artifacts"), "--artifacts"),
    raw_dir: Path = typer.Option(Path("data/raw"), "--raw-dir"),
) -> None:
    """Downstream family classifier demonstration."""
    from pnl2vec.demo_classifier import run_classifier_demo

    report_path = run_classifier_demo(artifacts, raw_dir)
    typer.echo(report_path.read_text(encoding="utf-8"))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
