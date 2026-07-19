"""Token representation for pnl2vec."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class TokenKind(str, Enum):
    SPECIAL = "SPECIAL"
    STRUCT = "STRUCT"
    HAND = "HAND"
    PITCH_CLASS = "PITCH_CLASS"
    ACCIDENTAL = "ACCIDENTAL"
    OCTAVE = "OCTAVE"
    DURATION = "DURATION"
    DOT_COUNT = "DOT_COUNT"
    ARTICULATION = "ARTICULATION"
    DYNAMIC = "DYNAMIC"
    SLUR = "SLUR"
    TIE = "TIE"
    PEDAL = "PEDAL"
    FINGER = "FINGER"
    REST = "REST"
    BARLINE = "BARLINE"
    KEY = "KEY"
    TIME_SIGNATURE = "TIME_SIGNATURE"
    TEMPO_BPM = "TEMPO_BPM"
    EVENT = "EVENT"
    NOTE = "NOTE"
    CHORD = "CHORD"
    META = "META"
    OTHER = "OTHER"


@dataclass(frozen=True)
class SourceSpan:
    start_line: int | None = None
    start_col: int | None = None
    end_line: int | None = None
    end_col: int | None = None
    filename: str | None = None


@dataclass(frozen=True)
class Token:
    kind: TokenKind
    value: str | int | float | None = None
    attributes: Mapping[str, str | int | float | bool] = field(default_factory=dict)
    source_span: SourceSpan | None = None
    # Grouping for context policies
    event_id: str | None = None
    measure: int | None = None
    voice: str | None = None
    hand: str | None = None
    offset: str | None = None  # rational string for simultaneity

    def canonical(self) -> str:
        """Unique canonical serialized form."""
        if self.kind == TokenKind.REST and self.value is None:
            return "REST"
        if self.value is None or self.value == "":
            return self.kind.value
        return f"{self.kind.value}:{self.value}"

    def musical_category(self) -> str:
        mapping = {
            TokenKind.SPECIAL: "special",
            TokenKind.STRUCT: "structural",
            TokenKind.HAND: "hand",
            TokenKind.PITCH_CLASS: "pitch",
            TokenKind.ACCIDENTAL: "pitch",
            TokenKind.OCTAVE: "pitch",
            TokenKind.DURATION: "duration",
            TokenKind.DOT_COUNT: "duration",
            TokenKind.ARTICULATION: "articulation",
            TokenKind.DYNAMIC: "dynamic",
            TokenKind.SLUR: "slur",
            TokenKind.TIE: "tie",
            TokenKind.PEDAL: "pedal",
            TokenKind.FINGER: "fingering",
            TokenKind.REST: "rest",
            TokenKind.BARLINE: "structural",
            TokenKind.KEY: "key",
            TokenKind.TIME_SIGNATURE: "meter",
            TokenKind.TEMPO_BPM: "tempo",
            TokenKind.EVENT: "event",
            TokenKind.NOTE: "compound",
            TokenKind.CHORD: "chord",
            TokenKind.META: "meta",
            TokenKind.OTHER: "other",
        }
        return mapping.get(self.kind, "other")

    @staticmethod
    def from_canonical(text: str) -> Token:
        if text == "REST":
            return Token(kind=TokenKind.REST)
        if ":" not in text:
            try:
                kind = TokenKind(text)
            except ValueError:
                kind = TokenKind.OTHER
            return Token(kind=kind, value=None)
        kind_s, _, value = text.partition(":")
        try:
            kind = TokenKind(kind_s)
        except ValueError:
            kind = TokenKind.OTHER
            value = text
        # Preserve numeric-looking values as strings for stability
        return Token(kind=kind, value=value)


SPECIAL_TOKENS = {
    "<PAD>": Token(kind=TokenKind.SPECIAL, value="PAD"),
    "<UNK>": Token(kind=TokenKind.SPECIAL, value="UNK"),
    "<BOS>": Token(kind=TokenKind.SPECIAL, value="BOS"),
    "<EOS>": Token(kind=TokenKind.SPECIAL, value="EOS"),
    "<MASK>": Token(kind=TokenKind.SPECIAL, value="MASK"),
    "<DOC_SEP>": Token(kind=TokenKind.SPECIAL, value="DOC_SEP"),
    "<MEASURE_SEP>": Token(kind=TokenKind.SPECIAL, value="MEASURE_SEP"),
}

# Canonical strings used in sequences
SPECIAL_CANONICAL = {
    "SPECIAL:PAD": "<PAD>",
    "SPECIAL:UNK": "<UNK>",
    "SPECIAL:BOS": "<BOS>",
    "SPECIAL:EOS": "<EOS>",
    "SPECIAL:MASK": "<MASK>",
    "SPECIAL:DOC_SEP": "<DOC_SEP>",
    "SPECIAL:MEASURE_SEP": "<MEASURE_SEP>",
}


def make_special(name: str) -> Token:
    """name without angle brackets, e.g. BOS."""
    return Token(kind=TokenKind.SPECIAL, value=name)


def token_attrs(**kwargs: Any) -> dict[str, str | int | float | bool]:
    return {k: v for k, v in kwargs.items() if v is not None}
