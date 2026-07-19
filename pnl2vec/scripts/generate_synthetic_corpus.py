#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pnl2vec.corpus import generate_corpus


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--size", default="tiny")
    p.add_argument("--output", default="data/raw")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    paths = generate_corpus(args.size, seed=args.seed, output_dir=args.output, force=args.force)
    print(f"wrote {len(paths)} files")


if __name__ == "__main__":
    main()
