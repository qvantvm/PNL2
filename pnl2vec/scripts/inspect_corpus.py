#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pnl2vec.cli import inspect_corpus


if __name__ == "__main__":
    from typer import Context
    import typer

    typer.run(inspect_corpus)
