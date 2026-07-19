"""Load PNL/2 documents from disk."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from pnl2.ast import Document, Score

from pnl2vec.pnl import PNLParseError, parse_pnl, validate_pnl

logger = logging.getLogger(__name__)


@dataclass
class CorpusDocument:
    path: Path
    text: str
    document: Document
    doc_id: str
    parse_error: str | None = None
    validation_warnings: list[str] = field(default_factory=list)


def load_corpus(
    root: Path | str,
    *,
    pattern: str = "**/*.pnl",
    skip_invalid: bool = False,
) -> list[CorpusDocument]:
    root = Path(root)
    paths = sorted(root.glob(pattern))
    docs: list[CorpusDocument] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        try:
            document = parse_pnl(text, filename=path)
        except PNLParseError as exc:
            logger.warning("parse failed %s: %s", path, exc)
            if skip_invalid:
                docs.append(
                    CorpusDocument(
                        path=path,
                        text=text,
                        document=Document(version="pnl/2", score=Score()),
                        doc_id=path.stem,
                        parse_error=str(exc),
                    )
                )
                continue
            raise
        issues = validate_pnl(document, filename=path)
        warnings = [i.message for i in issues]
        docs.append(
            CorpusDocument(
                path=path,
                text=text,
                document=document,
                doc_id=path.stem,
                validation_warnings=warnings,
            )
        )
    return docs


def load_valid_documents(root: Path | str) -> list[Document]:
    return [d.document for d in load_corpus(root) if d.parse_error is None]
