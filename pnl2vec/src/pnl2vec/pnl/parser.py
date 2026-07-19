"""Parser facade over pnl2 with structured diagnostics."""

from __future__ import annotations

import re
from pathlib import Path

from pnl2.ast import Document
from pnl2.parser import ParseError, parse as _parse

from .validation import ValidationIssue

_LOC_RE = re.compile(r"[Ll]ine\s+(\d+)(?::(\d+))?")


class PNLParseError(Exception):
    """Raised when PNL/2 source cannot be parsed."""

    def __init__(self, issue: ValidationIssue) -> None:
        super().__init__(issue.message)
        self.issue = issue


def parse_pnl(
    text: str,
    *,
    filename: str | Path | None = None,
) -> Document:
    """Parse PNL/2 text into a Document.

    Raises PNLParseError with filename, line, column, and excerpt when available.
    """
    name = str(filename) if filename is not None else None
    try:
        return _parse(text)
    except ParseError as exc:
        msg = str(exc)
        line = column = None
        m = _LOC_RE.search(msg)
        if m:
            line = int(m.group(1))
            if m.group(2):
                column = int(m.group(2))
        excerpt = None
        if line is not None:
            lines = text.splitlines()
            if 1 <= line <= len(lines):
                excerpt = lines[line - 1].strip()
        raise PNLParseError(
            ValidationIssue(
                message=msg,
                filename=name,
                line=line,
                column=column,
                unexpected=_extract_unexpected(msg),
                expected=_extract_expected(msg),
                excerpt=excerpt,
            )
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise PNLParseError(
            ValidationIssue(
                message=str(exc),
                filename=name,
            )
        ) from exc


def _extract_unexpected(msg: str) -> str | None:
    m = re.search(r"unexpected\s+(\S+)", msg, re.I)
    return m.group(1) if m else None


def _extract_expected(msg: str) -> str | None:
    m = re.search(r"expected\s+(.+?)(?:\s+at|\s*$)", msg, re.I)
    return m.group(1).strip() if m else None


__all__ = ["PNLParseError", "parse_pnl"]
