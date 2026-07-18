"""Abstract syntax tree for PNL/2 documents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .rational import Rational


@dataclass(frozen=True)
class Pitch:
    letter: str
    accidental: str  # "", "b", "bb", "#", "##"
    octave: int

    def __str__(self) -> str:
        return f"{self.letter}{self.accidental}{self.octave}"

    def sounding_midi(self) -> int:
        """Equal-tempered MIDI note number (C4 = 60)."""
        semis = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[self.letter]
        alter = {"": 0, "b": -1, "bb": -2, "#": 1, "##": 2}[self.accidental]
        return (self.octave + 1) * 12 + semis + alter

    def enharmonic_equal(self, other: "Pitch") -> bool:
        return self.sounding_midi() == other.sounding_midi()


@dataclass(frozen=True)
class Position:
    measure: int
    offset: Rational

    def __str__(self) -> str:
        return f"{self.measure}:{self.offset}"


@dataclass(frozen=True)
class Ratio:
    """Generic a:b ratio (tuplet, finger-change, etc.)."""

    left: int
    right: int

    def __str__(self) -> str:
        return f"{self.left}:{self.right}"


@dataclass
class Node:
    kind: str
    id: str | None = None
    props: dict[str, Any] = field(default_factory=dict)
    children: list["Node"] = field(default_factory=list)
    number: int | None = None  # for measure blocks

    def get(self, key: str, default: Any = None) -> Any:
        return self.props.get(key, default)


@dataclass
class Score:
    meta: dict[str, Any] = field(default_factory=dict)
    parts: list[Node] = field(default_factory=list)
    statements: list[Node] = field(default_factory=list)  # top-level extras


@dataclass
class Document:
    version: str
    score: Score
