"""Corpus statistics and reports."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from pnl2.ast import Document

from pnl2vec.corpus.loader import CorpusDocument
from pnl2vec.tokenizer import Tokenizer, Vocabulary


@dataclass
class CorpusReport:
    num_documents: int = 0
    num_measures: int = 0
    num_events: int = 0
    num_tokens: int = 0
    vocabulary_size: int = 0
    oov_rate: float = 0.0
    average_sequence_length: float = 0.0
    token_frequency: dict[str, int] = field(default_factory=dict)
    pitch_class_distribution: dict[str, int] = field(default_factory=dict)
    duration_distribution: dict[str, int] = field(default_factory=dict)
    articulation_frequency: dict[str, int] = field(default_factory=dict)
    dynamic_frequency: dict[str, int] = field(default_factory=dict)
    pedal_event_frequency: dict[str, int] = field(default_factory=dict)
    fingering_frequency: dict[str, int] = field(default_factory=dict)
    parse_failures: int = 0
    validation_warnings: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def to_markdown(self) -> str:
        lines = [
            "# Corpus Report",
            "",
            f"- Documents: {self.num_documents}",
            f"- Measures: {self.num_measures}",
            f"- Events: {self.num_events}",
            f"- Tokens: {self.num_tokens}",
            f"- Vocabulary size: {self.vocabulary_size}",
            f"- OOV rate: {self.oov_rate:.4f}",
            f"- Average sequence length: {self.average_sequence_length:.2f}",
            f"- Parse failures: {self.parse_failures}",
            f"- Validation warnings: {self.validation_warnings}",
            "",
            "## Pitch-class distribution",
        ]
        for k, v in sorted(self.pitch_class_distribution.items()):
            lines.append(f"- {k}: {v}")
        lines.append("")
        lines.append("## Duration distribution")
        for k, v in sorted(self.duration_distribution.items()):
            lines.append(f"- {k}: {v}")
        return "\n".join(lines) + "\n"


def _count_measures_events(doc: Document) -> tuple[int, int]:
    measures = 0
    events = 0
    for part in doc.score.parts:
        for child in part.children:
            if child.kind != "measure":
                continue
            measures += 1
            for staff in child.children:
                if staff.kind != "staff":
                    continue
                for voice in staff.children:
                    if voice.kind != "voice":
                        continue
                    for ev in voice.children:
                        if ev.kind in {"note", "rest", "chord"}:
                            events += 1
    return measures, events


def build_corpus_report(
    corpus: Sequence[CorpusDocument],
    tokenizer: Tokenizer,
    vocabulary: Vocabulary | None = None,
    *,
    eval_docs: Sequence[CorpusDocument] | None = None,
) -> CorpusReport:
    report = CorpusReport()
    report.num_documents = len(corpus)
    report.parse_failures = sum(1 for d in corpus if d.parse_error)
    report.validation_warnings = sum(len(d.validation_warnings) for d in corpus)

    freq: Counter[str] = Counter()
    seq_lens: list[int] = []
    for item in corpus:
        if item.parse_error:
            continue
        m, e = _count_measures_events(item.document)
        report.num_measures += m
        report.num_events += e
        strings = tokenizer.to_canonical_strings(tokenizer.tokenize(item.document))
        freq.update(strings)
        seq_lens.append(len(strings))

    report.num_tokens = sum(freq.values())
    report.average_sequence_length = sum(seq_lens) / len(seq_lens) if seq_lens else 0.0
    report.token_frequency = dict(freq.most_common(200))
    report.pitch_class_distribution = {
        k: v for k, v in freq.items() if k.startswith("PITCH_CLASS:")
    }
    report.duration_distribution = {k: v for k, v in freq.items() if k.startswith("DURATION:")}
    report.articulation_frequency = {
        k: v for k, v in freq.items() if k.startswith("ARTICULATION:")
    }
    report.dynamic_frequency = {k: v for k, v in freq.items() if k.startswith("DYNAMIC:")}
    report.pedal_event_frequency = {k: v for k, v in freq.items() if k.startswith("PEDAL:")}
    report.fingering_frequency = {k: v for k, v in freq.items() if k.startswith("FINGER:")}

    if vocabulary is not None:
        report.vocabulary_size = len(vocabulary)
        oov = 0
        total = 0
        check = eval_docs if eval_docs is not None else corpus
        for item in check:
            if item.parse_error:
                continue
            strings = tokenizer.to_canonical_strings(tokenizer.tokenize(item.document))
            for s in strings:
                total += 1
                if s not in vocabulary.token_to_id_ and s not in {
                    "<PAD>",
                    "<UNK>",
                    "<BOS>",
                    "<EOS>",
                    "<MASK>",
                    "<DOC_SEP>",
                    "<MEASURE_SEP>",
                }:
                    # reserved are in vocab; true OOV maps to UNK
                    if vocabulary.token_to_id(s) == vocabulary.unk_id and s != "<UNK>":
                        oov += 1
        report.oov_rate = oov / total if total else 0.0
    else:
        report.vocabulary_size = len(freq)

    return report


def save_report(report: CorpusReport, directory: Path | str, stem: str = "corpus_report") -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.json").write_text(report.to_json(), encoding="utf-8")
    (directory / f"{stem}.md").write_text(report.to_markdown(), encoding="utf-8")
