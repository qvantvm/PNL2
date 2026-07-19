# pnl2vec

Token embeddings for **PNL/2** (Piano Notation Language). This project learns static token vectors with skip-gram / CBOW so you can validate tokenization and musical geometry *before* training a transformer.

It reuses the canonical `pnl2` parser in the parent repository — it does not invent a second dialect.

## Conceptual distinctions

| Concept | Meaning |
| --- | --- |
| **Tokenizer** | Deterministic map from PNL/2 source → discrete token identities |
| **Vocabulary** | Map between canonical token strings and integer IDs |
| **Embedding layer** | Trainable matrix `vocab_size × dim`; an ID selects a row |
| **Training objective** | Why useful geometry appears (skip-gram / CBOW + negative sampling) |
| **Token embedding** | Static vector for one vocabulary item |
| **Event embedding** | Combination of atomic tokens for one musical event |
| **Phrase embedding** | Pooled tokens/events (baseline, not contextual) |
| **Contextual embedding** | Depends on surrounding events — **not** trained in v1 |

### What an embedding is

An embedding is a dense vector. Early in training the rows of `nn.Embedding` are essentially random. They become meaningful only because a training objective repeatedly pulls related tokens together and pushes unrelated ones apart.

### Tokenization and embeddings

If pitch and key share a surface string, the model cannot tell them apart. pnl2vec uses namespaced tokens (`PITCH_CLASS:C` vs `KEY:C_MAJOR`, `FINGER:3` vs `OCTAVE:3`).

**Atomic mode** decomposes events (`EVENT:NOTE`, `PITCH_CLASS:C`, `DURATION:1/4`, …) — small vocab, compositional.

**Compound mode** keeps frequent whole events (`NOTE:C#4:DUR_1/8`) and falls back to atomic for rares.

### Skip-gram vs CBOW

- **Skip-gram:** center token → predict context tokens.
- **CBOW:** context tokens → predict center.

Music-aware context (default **hybrid**) samples same-voice neighbors, simultaneous events, and measure-local pairs — not only a raw linear window. Same-event pairs are down-weighted so the model does not only learn that `OCTAVE:4` sits next to `PITCH_CLASS:C` inside one note.

### Token vs contextual embeddings

These vectors are **static**: `PITCH_CLASS:C` has one vector regardless of key or surrounding harmony. A future PNL transformer would produce contextual embeddings.

## Install

Python 3.11+. From this directory:

```bash
pip install -e ..
pip install -e ".[dev]"
# optional:
pip install -e ".[umap]"
```

## Five-minute experiment

```bash
pip install -e ..
pip install -e ".[dev]"
pnl2vec generate-synthetic --size tiny --force
pnl2vec train --config configs/train_skipgram.yaml
pnl2vec evaluate --checkpoint artifacts/checkpoints/best.pt
pnl2vec visualize --checkpoint artifacts/checkpoints/best.pt
pnl2vec neighbors "PITCH_CLASS:C" --top-k 10
```

## CLI

```text
pnl2vec validate <path>
pnl2vec generate-synthetic --size tiny
pnl2vec inspect-corpus <path>
pnl2vec build-vocab <path>
pnl2vec train --config configs/train_skipgram.yaml
pnl2vec evaluate --checkpoint artifacts/checkpoints/best.pt
pnl2vec visualize --checkpoint artifacts/checkpoints/best.pt
pnl2vec neighbors TOKEN --top-k 10
pnl2vec analogy TOKEN_A TOKEN_B TOKEN_C --top-k 10
pnl2vec embed input.pnl --output vector.npy
pnl2vec index <corpus-path> --output <index-path>
pnl2vec search query.pnl --index <index-path>
pnl2vec demo-classifier
```

Artifacts are not overwritten unless `--force` is passed (where applicable).

## Python API

```python
from pnl2vec import PNL2Vec

model = PNL2Vec.load("artifacts")
tokens = model.tokenize(pnl_text)
ids = model.encode(pnl_text)
phrase = model.embed_pnl(pnl_text)
neighbors = model.nearest_neighbors("PITCH_CLASS:C", top_k=10, category="pitch")
```

Phrase pooling options: `mean`, `freq`, `inv_freq`, `sif`, `mean_remove_pc`. Structural/special tokens are ignored by default.

## Artifacts

| Path | Contents |
| --- | --- |
| `artifacts/tokenizer/` | vocabulary, metadata, config |
| `artifacts/checkpoints/` | `best.pt`, `latest.pt` |
| `artifacts/embeddings/` | input / output / combined `.npy` |
| `artifacts/visualizations/` | PCA/t-SNE plots + interactive HTML |
| `artifacts/reports/` | training, evaluation, classifier demo |

## Interpreting results cautiously

- Nearest neighbors in 2D plots can look structured even when geometry is weak — always compare against **random** and **untrained** baselines in `evaluation_report.md`.
- Synthetic corpora create controllable patterns; relationships may not transfer to real scores.
- Namespace probes (predict pitch class from a `PITCH_CLASS:*` token) are partly trivial.
- Honest limitation: v1 embeddings are not contextual and are only as good as the corpus + objective.

## Sample results (tiny synthetic run)

After `generate-synthetic --size tiny` and skip-gram training (5 epochs, dim=64):

- Training loss decreased (~2.76 → ~2.56).
- Intrinsic NN baselines (mean P@10 / MRR): **learned** 0.14 / 0.60, **random** 0.02 / 0.02, **feature** 0.36 / 0.87. Hand-crafted features still beat early skip-gram on these soft neighbor sets — expected on a tiny corpus.
- Example neighbors for `PITCH_CLASS:C`: `PITCH_CLASS:E`, `PITCH_CLASS:D`, then fingering/octave tokens (mixed musical signal + co-occurrence).
- Downstream family classifier (mean-pool): frozen random can outperform frozen learned on this tiny setup — do not overclaim musical structure yet.
- Visualizations: `artifacts/visualizations/pca_*.png`, `interactive_pca.html`, `neighbor_graph.html`.

## Known limitations

- No transformer / attention.
- Phrase vectors are pooled baselines.
- Retrieval uses NumPy brute force (FAISS-ready interface).
- Medium synthetic corpus is not generated in tests.
- Tiny synthetic training does not guarantee strong music-theory geometry; compare baselines before trusting plots.

## Evolution path

Next step toward a contextual PNL/2 model: reuse this tokenizer + vocabulary, replace skip-gram with a small transformer language model (or masked event model) trained on the same document-bounded sequences, and keep the evaluation / neighbor tooling.

## Tests

```bash
pytest
```

No GPU required.
