"""Optional downstream classifier demonstration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from pnl2vec.api import PNL2Vec
from pnl2vec.corpus import load_corpus


class MeanPoolClassifier(nn.Module):
    def __init__(self, emb: np.ndarray, n_classes: int, *, freeze: bool = True) -> None:
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(emb, dtype=torch.float32), freeze=freeze
        )
        self.fc = nn.Linear(emb.shape[1], n_classes)

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)
        mask = mask.unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.fc(pooled)


def run_classifier_demo(artifacts: Path | str, raw_dir: Path | str) -> Path:
    artifacts = Path(artifacts)
    raw_dir = Path(raw_dir)
    model = PNL2Vec.load(artifacts)
    docs = load_corpus(raw_dir)
    X_ids: list[list[int]] = []
    y: list[str] = []
    for d in docs:
        if d.parse_error:
            continue
        fam = d.document.score.meta.get("family") if isinstance(d.document.score.meta, dict) else None
        if not fam:
            continue
        X_ids.append(model.encode(d.text))
        y.append(str(fam))
    if len(set(y)) < 2 or len(y) < 20:
        out = artifacts / "reports" / "classifier_demo.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("# Classifier demo\n\nInsufficient labeled family data.\n", encoding="utf-8")
        return out

    le = LabelEncoder()
    labels = le.fit_transform(y)
    max_len = min(256, max(len(x) for x in X_ids))

    def pad(seqs):
        arr = np.zeros((len(seqs), max_len), dtype=np.int64)
        mask = np.zeros((len(seqs), max_len), dtype=np.int64)
        for i, s in enumerate(seqs):
            s = s[:max_len]
            arr[i, : len(s)] = s
            mask[i, : len(s)] = 1
        return arr, mask

    idx = np.arange(len(X_ids))
    tr, te = train_test_split(idx, test_size=0.25, random_state=42, stratify=labels)
    results = {}
    for name, emb, freeze, trainable_note in [
        ("frozen_random", np.random.default_rng(0).normal(size=model.embeddings.shape).astype(np.float32), True, "random"),
        ("frozen_learned", model.embeddings.astype(np.float32), True, "learned-frozen"),
        ("trainable_learned", model.embeddings.astype(np.float32), False, "learned-trainable"),
    ]:
        clf = MeanPoolClassifier(emb, len(le.classes_), freeze=freeze)
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, clf.parameters()), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        Xtr, Mtr = pad([X_ids[i] for i in tr])
        ytr = labels[tr]
        Xte, Mte = pad([X_ids[i] for i in te])
        yte = labels[te]
        for _ in range(8):
            clf.train()
            logits = clf(torch.tensor(Xtr), torch.tensor(Mtr))
            loss = loss_fn(logits, torch.tensor(ytr))
            opt.zero_grad()
            loss.backward()
            opt.step()
        clf.eval()
        with torch.no_grad():
            pred = clf(torch.tensor(Xte), torch.tensor(Mte)).argmax(dim=-1).numpy()
        acc = float((pred == yte).mean())
        results[name] = {"accuracy": acc, "note": trainable_note}

    lines = [
        "# Downstream classifier demonstration",
        "",
        "Task: predict synthetic document `family` (scale/chordal/alberti/...).",
        "",
        "These are pooled baselines, not contextual sequence models.",
        "",
    ]
    for k, v in results.items():
        lines.append(f"- **{k}**: accuracy={v['accuracy']:.3f} ({v['note']})")
    out = artifacts / "reports" / "classifier_demo.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
