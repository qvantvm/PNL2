#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pnl2vec.cli import evaluate_cmd
import typer

if __name__ == "__main__":
    typer.run(evaluate_cmd)
