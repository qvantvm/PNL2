"""Tokenizer config and artifact serialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .tokenizer import Tokenizer, TokenizerConfig
from .vocabulary import Vocabulary


def load_tokenizer_config(path: Path | str | dict[str, Any] | None) -> TokenizerConfig:
    if path is None:
        return TokenizerConfig()
    if isinstance(path, dict):
        data = path
    else:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    tok = data.get("tokenizer", data)
    return TokenizerConfig(
        mode=tok.get("mode", "atomic"),
        compound_min_frequency=int(tok.get("compound_min_frequency", 20)),
        normalize_enharmonics=bool(tok.get("normalize_enharmonics", False)),
        preserve_pitch_spelling=bool(tok.get("preserve_pitch_spelling", True)),
        preserve_source_spans=bool(tok.get("preserve_source_spans", True)),
    )


def save_tokenizer_config(config: TokenizerConfig, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tokenizer": {
            "mode": config.mode,
            "compound_min_frequency": config.compound_min_frequency,
            "normalize_enharmonics": config.normalize_enharmonics,
            "preserve_pitch_spelling": config.preserve_pitch_spelling,
            "preserve_source_spans": config.preserve_source_spans,
        }
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def save_tokenizer_artifacts(
    vocab: Vocabulary,
    config: TokenizerConfig,
    directory: Path | str,
) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    vocab.save(directory / "vocabulary.json")
    vocab.save_metadata(directory / "token_metadata.json")
    save_tokenizer_config(config, directory / "config.yaml")


def load_vocabulary(directory: Path | str) -> Vocabulary:
    return Vocabulary.load(Path(directory) / "vocabulary.json")
