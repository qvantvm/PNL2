"""Serializer facade over pnl2."""

from __future__ import annotations

from pnl2.ast import Document
from pnl2.serializer import serialize as _serialize


def serialize_pnl(document: Document, *, indent: str = "    ") -> str:
    return _serialize(document, indent=indent)


__all__ = ["serialize_pnl"]
