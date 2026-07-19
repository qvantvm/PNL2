"""Vocabulary mapping between canonical token strings and integer IDs."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .token import SPECIAL_TOKENS, Token, TokenKind


RESERVED = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
    "<MASK>",
    "<DOC_SEP>",
    "<MEASURE_SEP>",
]


@dataclass
class VocabEntry:
    id: int
    canonical: str
    kind: str
    value: str | int | float | None
    musical_category: str
    frequency: int
    is_special: bool
    is_atomic: bool
    is_compound: bool


@dataclass
class Vocabulary:
    token_to_id_: dict[str, int] = field(default_factory=dict)
    id_to_token_: dict[int, str] = field(default_factory=dict)
    metadata: dict[str, VocabEntry] = field(default_factory=dict)
    frequencies: Counter[str] = field(default_factory=Counter)

    def __post_init__(self) -> None:
        if not self.token_to_id_:
            for i, name in enumerate(RESERVED):
                self.token_to_id_[name] = i
                self.id_to_token_[i] = name
                tok = SPECIAL_TOKENS[name]
                self.metadata[name] = VocabEntry(
                    id=i,
                    canonical=name,
                    kind=tok.kind.value,
                    value=tok.value,
                    musical_category="special",
                    frequency=0,
                    is_special=True,
                    is_atomic=True,
                    is_compound=False,
                )

    @property
    def pad_id(self) -> int:
        return self.token_to_id_["<PAD>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id_["<UNK>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id_["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id_["<EOS>"]

    def __len__(self) -> int:
        return len(self.token_to_id_)

    def token_to_id(self, token: str) -> int:
        return self.token_to_id_.get(token, self.unk_id)

    def id_to_token(self, token_id: int) -> str:
        return self.id_to_token_.get(token_id, "<UNK>")

    def encode(self, tokens: Sequence[Token]) -> list[int]:
        return [self.token_to_id(t.canonical() if t.kind != TokenKind.SPECIAL else self._special_key(t)) for t in tokens]

    def encode_strings(self, tokens: Sequence[str]) -> list[int]:
        return [self.token_to_id(t) for t in tokens]

    def decode(self, ids: Sequence[int]) -> list[Token]:
        out: list[Token] = []
        for i in ids:
            s = self.id_to_token(i)
            if s in SPECIAL_TOKENS:
                out.append(SPECIAL_TOKENS[s])
            else:
                out.append(Token.from_canonical(s))
        return out

    @staticmethod
    def _special_key(t: Token) -> str:
        if t.value is None:
            return "<UNK>"
        key = f"<{t.value}>"
        return key if key in SPECIAL_TOKENS else t.canonical()

    def add(self, canonical: str, token: Token, *, frequency: int = 0, is_compound: bool = False) -> int:
        if canonical in self.token_to_id_:
            entry = self.metadata[canonical]
            entry.frequency += frequency
            return entry.id
        idx = len(self.token_to_id_)
        self.token_to_id_[canonical] = idx
        self.id_to_token_[idx] = canonical
        self.metadata[canonical] = VocabEntry(
            id=idx,
            canonical=canonical,
            kind=token.kind.value,
            value=token.value,
            musical_category=token.musical_category(),
            frequency=frequency,
            is_special=False,
            is_atomic=not is_compound,
            is_compound=is_compound,
        )
        return idx

    def build_from_frequencies(
        self,
        frequencies: Counter[str],
        token_lookup: dict[str, Token],
        *,
        compound_keys: set[str] | None = None,
        min_frequency: int = 1,
    ) -> None:
        compound_keys = compound_keys or set()
        self.frequencies = Counter(frequencies)
        for canon, freq in frequencies.most_common():
            if canon in self.token_to_id_:
                self.metadata[canon].frequency = freq
                continue
            if freq < min_frequency:
                continue
            tok = token_lookup.get(canon) or Token.from_canonical(canon)
            self.add(canon, tok, frequency=freq, is_compound=canon in compound_keys)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "token_to_id": self.token_to_id_,
            "metadata": {k: asdict(v) for k, v in self.metadata.items()},
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> Vocabulary:
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        vocab = cls.__new__(cls)
        vocab.token_to_id_ = {k: int(v) for k, v in data["token_to_id"].items()}
        vocab.id_to_token_ = {int(v): k for k, v in vocab.token_to_id_.items()}
        vocab.metadata = {}
        for k, v in data["metadata"].items():
            vocab.metadata[k] = VocabEntry(**v)
        vocab.frequencies = Counter({k: e.frequency for k, e in vocab.metadata.items()})
        return vocab

    def save_metadata(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        items = [asdict(e) for e in sorted(self.metadata.values(), key=lambda e: e.id)]
        path.write_text(json.dumps(items, indent=2), encoding="utf-8")


def count_token_strings(sequences: Iterable[Sequence[str]]) -> Counter[str]:
    c: Counter[str] = Counter()
    for seq in sequences:
        c.update(seq)
    return c
