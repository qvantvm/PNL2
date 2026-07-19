#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pnl2vec.training import train_from_config


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train_skipgram.yaml")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    print(train_from_config(args.config, force=args.force))


if __name__ == "__main__":
    main()
