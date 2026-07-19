"""Embedding trainer for skip-gram and CBOW."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml
from tqdm import tqdm

from pnl2vec.models import CBOWNS, NegativeSampler, SkipGramNS
from pnl2vec.tokenizer import Tokenizer, TokenizerConfig, Vocabulary
from pnl2vec.tokenizer.serialization import save_tokenizer_artifacts
from pnl2vec.training.checkpoint import load_checkpoint, save_checkpoint
from pnl2vec.training.dataset import ContextConfig, ContextPolicy, PairStream
from pnl2vec.training.seed import select_device, set_seed

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    seed: int = 42
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    artifacts_dir: Path = Path("artifacts")
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    objective: Literal["skipgram", "cbow"] = "skipgram"
    embedding_dim: int = 128
    sparse_embeddings: bool = False
    context: ContextConfig = field(default_factory=ContextConfig)
    epochs: int = 20
    batch_size: int = 1024
    learning_rate: float = 0.003
    negative_samples: int = 10
    optimizer_name: str = "adamw"
    weight_decay: float = 0.0001
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 4
    device: str = "auto"
    num_workers: int = 0
    log_every_steps: int = 100
    save_every_epochs: int = 1


def load_train_config(path: Path | str) -> TrainConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    tok = data.get("tokenizer", {})
    ctx = data.get("context", {})
    model = data.get("model", {})
    training = data.get("training", {})
    logging_cfg = data.get("logging", {})
    data_cfg = data.get("data", {})
    policy = ctx.get("policy", "hybrid")
    return TrainConfig(
        seed=int(data.get("seed", 42)),
        raw_dir=Path(data_cfg.get("raw_dir", "data/raw")),
        processed_dir=Path(data_cfg.get("processed_dir", "data/processed")),
        artifacts_dir=Path(data.get("artifacts_dir", "artifacts")),
        tokenizer=TokenizerConfig(
            mode=tok.get("mode", "atomic"),
            compound_min_frequency=int(tok.get("compound_min_frequency", 20)),
            normalize_enharmonics=bool(tok.get("normalize_enharmonics", False)),
            preserve_pitch_spelling=bool(tok.get("preserve_pitch_spelling", True)),
            preserve_source_spans=bool(tok.get("preserve_source_spans", True)),
        ),
        objective=model.get("objective", "skipgram"),
        embedding_dim=int(model.get("embedding_dim", 128)),
        sparse_embeddings=bool(model.get("sparse_embeddings", False)),
        context=ContextConfig(
            policy=ContextPolicy(policy),
            min_window=int(ctx.get("min_window", 1)),
            max_window=int(ctx.get("max_window", 4)),
            include_same_event=bool(ctx.get("include_same_event", True)),
            same_event_weight=float(ctx.get("same_event_weight", 0.25)),
            sequential_event_weight=float(ctx.get("sequential_event_weight", 1.0)),
            simultaneous_event_weight=float(ctx.get("simultaneous_event_weight", 1.0)),
            same_measure_weight=float(ctx.get("same_measure_weight", 0.5)),
        ),
        epochs=int(training.get("epochs", 20)),
        batch_size=int(training.get("batch_size", 1024)),
        learning_rate=float(training.get("learning_rate", 0.003)),
        negative_samples=int(training.get("negative_samples", 10)),
        optimizer_name=training.get("optimizer", "adamw"),
        weight_decay=float(training.get("weight_decay", 0.0001)),
        gradient_clip_norm=float(training.get("gradient_clip_norm", 1.0)),
        early_stopping_patience=int(training.get("early_stopping_patience", 4)),
        device=training.get("device", "auto"),
        num_workers=int(training.get("num_workers", 0)),
        log_every_steps=int(logging_cfg.get("log_every_steps", 100)),
        save_every_epochs=int(logging_cfg.get("save_every_epochs", 1)),
    )


def build_vocabulary(train_docs, tokenizer: Tokenizer) -> Vocabulary:
    from collections import Counter

    from pnl2vec.tokenizer.token import Token

    if tokenizer.config.mode == "compound":
        tokenizer.fit_compound_frequencies([d.document for d in train_docs])
    freq: Counter[str] = Counter()
    lookup: dict[str, Token] = {}
    compounds: set[str] = set()
    for d in train_docs:
        ann = tokenizer.tokenize_annotated(d.document)
        for a in ann:
            t = a.token
            canon = f"<{t.value}>" if t.kind.value == "SPECIAL" else t.canonical()
            freq[canon] += 1
            lookup[canon] = t
            if a.is_compound:
                compounds.add(canon)
    vocab = Vocabulary()
    vocab.build_from_frequencies(freq, lookup, compound_keys=compounds)
    return vocab


def export_embeddings(
    model: torch.nn.Module,
    vocabulary: Vocabulary,
    directory: Path,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    inp = model.input_embeddings.weight.detach().cpu().numpy()
    out = model.output_embeddings.weight.detach().cpu().numpy()
    combined = (inp + out) / 2.0
    # L2-normalize combined for default inspection copy (raw also saved)
    norms = np.linalg.norm(combined, axis=1, keepdims=True).clip(min=1e-12)
    combined_normed = combined / norms
    np.save(directory / "input_embeddings.npy", inp)
    np.save(directory / "output_embeddings.npy", out)
    np.save(directory / "combined_embeddings.npy", combined)
    np.save(directory / "combined_embeddings_normalized.npy", combined_normed)
    token_ids = {vocabulary.id_to_token(i): i for i in range(len(vocabulary))}
    (directory / "token_ids.json").write_text(json.dumps(token_ids, indent=2), encoding="utf-8")
    meta = {
        "vocab_size": len(vocabulary),
        "embedding_dim": int(inp.shape[1]),
        "default_for_inspection": "combined",
    }
    (directory / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


class Trainer:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        set_seed(config.seed)
        self.device = select_device(config.device)

    def train(self, train_docs, val_docs=None, *, force: bool = False) -> dict[str, Any]:
        cfg = self.config
        artifacts = cfg.artifacts_dir
        ckpt_dir = artifacts / "checkpoints"
        emb_dir = artifacts / "embeddings"
        report_dir = artifacts / "reports"
        tok_dir = artifacts / "tokenizer"
        for d in (ckpt_dir, emb_dir, report_dir, tok_dir):
            d.mkdir(parents=True, exist_ok=True)

        tokenizer = Tokenizer(cfg.tokenizer)
        vocabulary = build_vocabulary(train_docs, tokenizer)
        save_tokenizer_artifacts(vocabulary, cfg.tokenizer, tok_dir)

        model: torch.nn.Module
        if cfg.objective == "cbow":
            model = CBOWNS(len(vocabulary), cfg.embedding_dim, sparse=cfg.sparse_embeddings)
        else:
            model = SkipGramNS(len(vocabulary), cfg.embedding_dim, sparse=cfg.sparse_embeddings)
        model.to(self.device)

        if cfg.optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

        sampler = NegativeSampler.from_vocabulary(vocabulary, seed=cfg.seed)
        history: list[dict[str, Any]] = []
        best_val = float("inf")
        patience = 0

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()
            model.train()
            stream = PairStream(
                train_docs,
                tokenizer,
                vocabulary,
                cfg.context,
                seed=cfg.seed + epoch,
            )
            total_loss = 0.0
            n_batches = 0
            pos_score_sum = 0.0
            neg_score_sum = 0.0
            batch_centers: list[int] = []
            batch_contexts: list[int] = []

            def flush() -> None:
                nonlocal total_loss, n_batches, pos_score_sum, neg_score_sum, batch_centers, batch_contexts
                if not batch_centers:
                    return
                centers = torch.tensor(batch_centers, dtype=torch.long, device=self.device)
                contexts = torch.tensor(batch_contexts, dtype=torch.long, device=self.device)
                negatives = sampler.sample(cfg.negative_samples, batch_size=len(batch_centers)).to(
                    self.device
                )
                optimizer.zero_grad(set_to_none=True)
                if cfg.objective == "cbow":
                    # Use single context token as degenerate CBOW context window
                    ctx = contexts.unsqueeze(1)
                    loss = model(ctx, centers, negatives)
                else:
                    loss = model(centers, contexts, negatives)
                loss.backward()
                if cfg.gradient_clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
                optimizer.step()
                with torch.no_grad():
                    if cfg.objective == "skipgram":
                        ps = model.scores(centers, contexts).mean().item()
                        ns = model.scores(centers, negatives[:, 0]).mean().item()
                    else:
                        ps = 0.0
                        ns = 0.0
                total_loss += float(loss.item())
                pos_score_sum += ps
                neg_score_sum += ns
                n_batches += 1
                batch_centers, batch_contexts = [], []

            pbar = tqdm(stream.iter_pairs(), desc=f"epoch {epoch}", leave=False)
            for pair in pbar:
                batch_centers.append(pair.center_id)
                batch_contexts.append(pair.context_id)
                if len(batch_centers) >= cfg.batch_size:
                    flush()
                    if n_batches % max(1, cfg.log_every_steps) == 0:
                        pbar.set_postfix(loss=total_loss / max(1, n_batches))
            flush()

            train_loss = total_loss / max(1, n_batches)
            val_loss = self._eval_loss(model, sampler, val_docs or train_docs[: max(1, len(train_docs)//10)], tokenizer, vocabulary)
            duration = time.time() - t0
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "positive_pair_score": pos_score_sum / max(1, n_batches),
                "negative_pair_score": neg_score_sum / max(1, n_batches),
                "learning_rate": cfg.learning_rate,
                "epoch_duration": duration,
            }
            history.append(row)
            logger.info("epoch %s train=%.4f val=%.4f", epoch, train_loss, val_loss)

            save_checkpoint(
                ckpt_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=row,
                config={"objective": cfg.objective, "embedding_dim": cfg.embedding_dim},
            )
            if val_loss < best_val:
                best_val = val_loss
                patience = 0
                save_checkpoint(
                    ckpt_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=row,
                    config={"objective": cfg.objective, "embedding_dim": cfg.embedding_dim},
                )
                export_embeddings(model, vocabulary, emb_dir)
            else:
                patience += 1
                if patience >= cfg.early_stopping_patience:
                    logger.info("early stopping at epoch %s", epoch)
                    break

        # Ensure embeddings exist
        if not (emb_dir / "combined_embeddings.npy").exists():
            export_embeddings(model, vocabulary, emb_dir)

        (report_dir / "training_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        md = ["# Training Report", ""]
        for row in history:
            md.append(
                f"- Epoch {row['epoch']}: train={row['train_loss']:.4f} val={row['val_loss']:.4f}"
            )
        (report_dir / "training_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        return {"history": history, "best_val_loss": best_val, "vocab_size": len(vocabulary)}

    def _eval_loss(self, model, sampler, docs, tokenizer, vocabulary) -> float:
        if not docs:
            return 0.0
        model.eval()
        cfg = self.config
        stream = PairStream(docs, tokenizer, vocabulary, cfg.context, seed=cfg.seed)
        total = 0.0
        n = 0
        centers: list[int] = []
        contexts: list[int] = []
        with torch.no_grad():
            for pair in stream.iter_pairs():
                centers.append(pair.center_id)
                contexts.append(pair.context_id)
                if len(centers) >= cfg.batch_size:
                    c = torch.tensor(centers, device=self.device)
                    x = torch.tensor(contexts, device=self.device)
                    neg = sampler.sample(cfg.negative_samples, batch_size=len(centers)).to(self.device)
                    if cfg.objective == "cbow":
                        loss = model(x.unsqueeze(1), c, neg)
                    else:
                        loss = model(c, x, neg)
                    total += float(loss.item())
                    n += 1
                    centers, contexts = [], []
                    if n >= 20:
                        break
        model.train()
        return total / max(1, n)


def train_from_config(config_path: Path | str, *, force: bool = False) -> dict[str, Any]:
    from pnl2vec.corpus import load_corpus, split_documents

    cfg = load_train_config(config_path)
    corpus = load_corpus(cfg.raw_dir)
    if not corpus:
        raise FileNotFoundError(f"no .pnl files in {cfg.raw_dir}")
    split = split_documents(corpus, seed=cfg.seed)
    trainer = Trainer(cfg)
    return trainer.train(split.train, split.val, force=force)
