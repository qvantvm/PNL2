"""Exact rational numbers for PNL/2 durations and offsets."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import Union

NumberLike = Union[int, str, "Rational", Fraction]


@dataclass(frozen=True)
class Rational:
    """Immutable reduced rational number."""

    numerator: int
    denominator: int = 1

    def __post_init__(self) -> None:
        if self.denominator == 0:
            raise ValueError("rational denominator must be nonzero")
        num, den = self.numerator, self.denominator
        if den < 0:
            num, den = -num, -den
        g = gcd(num, den) or 1
        object.__setattr__(self, "numerator", num // g)
        object.__setattr__(self, "denominator", den // g)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Rational):
            return self.numerator == other.numerator and self.denominator == other.denominator
        if isinstance(other, (int, Fraction)):
            return self.to_fraction() == Fraction(other)
        return NotImplemented

    def __lt__(self, other: NumberLike) -> bool:
        return self.to_fraction() < Rational.from_value(other).to_fraction()

    def __le__(self, other: NumberLike) -> bool:
        return self.to_fraction() <= Rational.from_value(other).to_fraction()

    def __gt__(self, other: NumberLike) -> bool:
        return self.to_fraction() > Rational.from_value(other).to_fraction()

    def __ge__(self, other: NumberLike) -> bool:
        return self.to_fraction() >= Rational.from_value(other).to_fraction()

    def __hash__(self) -> int:
        return hash((self.numerator, self.denominator))

    @classmethod
    def from_value(cls, value: NumberLike) -> "Rational":
        if isinstance(value, Rational):
            return value
        if isinstance(value, Fraction):
            return cls(value.numerator, value.denominator)
        if isinstance(value, int):
            return cls(value, 1)
        if isinstance(value, str):
            text = value.strip()
            if "/" in text:
                a, b = text.split("/", 1)
                return cls(int(a), int(b))
            return cls(int(text), 1)
        raise TypeError(f"cannot convert {type(value)!r} to Rational")

    def to_fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    def __add__(self, other: NumberLike) -> "Rational":
        o = Rational.from_value(other).to_fraction()
        r = self.to_fraction() + o
        return Rational(r.numerator, r.denominator)

    def __sub__(self, other: NumberLike) -> "Rational":
        o = Rational.from_value(other).to_fraction()
        r = self.to_fraction() - o
        return Rational(r.numerator, r.denominator)

    def __mul__(self, other: NumberLike) -> "Rational":
        o = Rational.from_value(other).to_fraction()
        r = self.to_fraction() * o
        return Rational(r.numerator, r.denominator)

    def __truediv__(self, other: NumberLike) -> "Rational":
        o = Rational.from_value(other).to_fraction()
        r = self.to_fraction() / o
        return Rational(r.numerator, r.denominator)

    def __bool__(self) -> bool:
        return self.numerator != 0

    def __int__(self) -> int:
        if self.denominator != 1:
            raise ValueError("rational is not an integer")
        return self.numerator

    def __float__(self) -> float:
        return self.numerator / self.denominator

    def __str__(self) -> str:
        if self.denominator == 1:
            return str(self.numerator)
        return f"{self.numerator}/{self.denominator}"

    def __repr__(self) -> str:
        return f"Rational({self.numerator}, {self.denominator})"

    def is_reduced(self) -> bool:
        return gcd(abs(self.numerator), self.denominator) == 1


def effective_duration(
    dur: NumberLike,
    augment: int = 0,
    tuplet: tuple[int, int] | None = None,
) -> Rational:
    """Compute effective duration from written dur, dots, and tuplet."""
    d = Rational.from_value(dur)
    if augment < 0:
        raise ValueError("augment must be non-negative")
    if augment:
        # dur × (2 - 1/2^augment)
        factor = Rational(2) - Rational(1, 2**augment)
        d = d * factor
    if tuplet is not None:
        actual, normal = tuplet
        if actual <= 0 or normal <= 0:
            raise ValueError("tuplet ratios must be positive")
        d = d * Rational(normal, actual)
    return d


def parse_position(text: str) -> tuple[int, Rational]:
    """Parse measure:offset into (measure, offset)."""
    if ":" not in text:
        raise ValueError(f"invalid position: {text!r}")
    measure_s, offset_s = text.split(":", 1)
    return int(measure_s), Rational.from_value(offset_s)
