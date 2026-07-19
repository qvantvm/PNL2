"""Validation facade over pnl2 with structured issues."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pnl2.ast import Document
from pnl2.validator import validate as _validate


@dataclass(frozen=True)
class ValidationIssue:
    message: str
    filename: str | None = None
    line: int | None = None
    column: int | None = None
    unexpected: str | None = None
    expected: str | None = None
    excerpt: str | None = None
    severity: str = "error"


_LOC_RE = re.compile(r"[Ll]ine\s+(\d+)(?::(\d+))?")


def validate_pnl(
    document: Document,
    *,
    filename: str | Path | None = None,
) -> list[ValidationIssue]:
    """Validate a document. Returns structured issues (empty if ok)."""
    name = str(filename) if filename is not None else None
    raw = _validate(document)
    issues: list[ValidationIssue] = []
    for msg in raw:
        line = column = None
        m = _LOC_RE.search(msg)
        if m:
            line = int(m.group(1))
            if m.group(2):
                column = int(m.group(2))
        issues.append(
            ValidationIssue(
                message=msg,
                filename=name,
                line=line,
                column=column,
            )
        )
    return issues


__all__ = ["ValidationIssue", "validate_pnl"]
