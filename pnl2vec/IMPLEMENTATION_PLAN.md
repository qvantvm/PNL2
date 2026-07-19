# pnl2vec Implementation Plan

## Goal

Learn and validate PNL/2 token embeddings (skip-gram / CBOW) before introducing a transformer. Parse structurally, tokenize without ambiguity, train from musical context, evaluate, visualize, and expose a reusable API/CLI.

## Canonical PNL/2 (reuse, do not fork)

| Asset | Location |
| --- | --- |
| EBNF grammar | `../grammar/pnl2.ebnf` |
| Public API | `../src/pnl2/__init__.py` — `parse`, `serialize`, `validate` |
| AST | `../src/pnl2/ast.py` — `Document`, `Score`, `Node`, `Pitch` |
| Lexer / parser | `../src/pnl2/lexer.py`, `../src/pnl2/parser.py` |
| Validator / serializer | `../src/pnl2/validator.py`, `../src/pnl2/serializer.py` |
| Examples | `../examples/*.pnl` |

`pnl2vec.pnl` is a thin facade over `pnl2`. No second dialect. Grammar is not modified.

## Assumptions

1. **Staff → hand.** Staff ids `RH` / `LH` (and aliases `right` / `left`) map to `HAND:RIGHT` / `HAND:LEFT`. Other staff ids emit `HAND:UNKNOWN` plus a staff attribute.
2. **Hierarchy.** Tokenization walks `part → measure → staff → voice → events`, inserting structural boundary tokens.
3. **Relations.** Ties, slurs, and pedals become explicit tokens (`TIE:START`/`END`, `SLUR:START`/`END`, `PEDAL:DOWN`/`UP`) placed near related events when ids resolve; otherwise at part level after measures.
4. **Pitch spelling.** Default preserves written spelling (`PITCH_CLASS` + `ACCIDENTAL` + `OCTAVE`). Optional enharmonic normalization collapses to a canonical sounding class when configured.
5. **Durations.** Written `dur` and `augment` (dots) are separate tokens (`DURATION:…`, `DOT_COUNT:…`).
6. **Synthetic corpus.** Generated documents set `meta.synthetic=true` and are validated with `pnl2` before write.
7. **Vocabulary leakage.** Vocabulary is built only from the training document split.
8. **Device.** CUDA → MPS → CPU. Embeddings are dense by default (MPS-safe).

## Risks

| Risk | Mitigation |
| --- | --- |
| Same-event atomic adjacency learns trivial co-occurrence | Configurable same-event weight; default hybrid policy down-weights same-event pairs |
| Tiny synthetic corpus yields weak musical geometry | Honest evaluation vs baselines; document limitations |
| Parent parse errors lack rich spans | Facade wraps messages into `ValidationIssue`; best-effort line/col from exception text |
| Large pair materialization | Streaming / chunked pair generation |

## Non-goals (v1)

- Transformer / contextual sequence model
- Editing the PNL/2 grammar or parent parser
- Auto-generating the medium corpus in tests/CI
- FAISS (NumPy brute-force index with an abstract interface)

## Module map

See repository layout in `prompt.md` §2. Training default: hybrid event-aware context + skip-gram with negative sampling.

## Acceptance

`pip install -e ".[dev]"`, `pytest`, and the five-minute experiment in the README must succeed. Results are reported honestly.
