"""Corpus loading, splitting, statistics, and synthetic generation."""

from .examples import SIZE_COUNTS, generate_corpus
from .loader import CorpusDocument, load_corpus
from .split import CorpusSplit, split_documents
from .statistics import CorpusReport, build_corpus_report, save_report

__all__ = [
    "SIZE_COUNTS",
    "CorpusDocument",
    "CorpusReport",
    "CorpusSplit",
    "build_corpus_report",
    "generate_corpus",
    "load_corpus",
    "save_report",
    "split_documents",
]
