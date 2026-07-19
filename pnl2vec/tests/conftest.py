"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PARENT_SRC = ROOT.parent / "src"
SRC = ROOT / "src"

for p in (SRC, PARENT_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

EXAMPLES = ROOT / "data" / "examples"


@pytest.fixture
def examples_dir() -> Path:
    return EXAMPLES


@pytest.fixture
def tiny_scale_text() -> str:
    return (EXAMPLES / "tiny_scale.pnl").read_text(encoding="utf-8")


@pytest.fixture
def articulation_text() -> str:
    return (EXAMPLES / "articulation.pnl").read_text(encoding="utf-8")
