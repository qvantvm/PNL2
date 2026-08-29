import json
from pathlib import Path

import pytest

from pnl2.studio.sample import (
    BLANK_PNL,
    Sample,
    load_sample,
    new_sample,
    save_sample,
    sidecar_path_for,
)


def test_new_sample_is_valid_pnl():
    from pnl2 import parse, validate

    sample = new_sample()
    doc = parse(sample.text)
    assert sample.pnl_path is None
    assert validate(doc) == []


def test_load_pnl_attaches_sibling_png(tmp_path: Path):
    pnl = tmp_path / "scale.pnl"
    png = tmp_path / "scale.png"
    pnl.write_text(BLANK_PNL, encoding="utf-8")
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    sample = load_sample(pnl)
    assert sample.text.startswith("pnl/2")
    assert sample.expected_path == png.resolve()
    assert sample.sidecar_path is None


def test_save_writes_sidecar_and_copies_reference(tmp_path: Path):
    src_png = tmp_path / "gold.png"
    src_png.write_bytes(b"\x89PNG\r\n\x1a\nref")
    sample = Sample(text=BLANK_PNL, expected_path=src_png)
    dest = tmp_path / "out" / "my-scale.pnl"
    saved = save_sample(sample, dest)
    assert saved.pnl_path == dest.resolve()
    assert dest.read_text(encoding="utf-8") == BLANK_PNL
    sidecar = sidecar_path_for(dest)
    assert sidecar.is_file()
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert data["pnl"] == "my-scale.pnl"
    assert data["expected"] == "my-scale.png"
    copied = dest.with_suffix(".png")
    assert copied.read_bytes() == src_png.read_bytes()
    assert saved.expected_path == copied.resolve()


def test_save_as_round_trip_via_sidecar(tmp_path: Path):
    png = tmp_path / "a.png"
    png.write_bytes(b"png-bytes")
    first = save_sample(Sample(text=BLANK_PNL, expected_path=png), tmp_path / "a.pnl")
    other = tmp_path / "nested" / "b.pnl"
    second = save_sample(first, other)
    assert (tmp_path / "nested" / "b.png").is_file()
    reloaded = load_sample(second.sidecar_path)
    assert reloaded.text == BLANK_PNL
    assert reloaded.expected_path == (tmp_path / "nested" / "b.png").resolve()


def test_load_sidecar_without_expected(tmp_path: Path):
    pnl = tmp_path / "solo.pnl"
    pnl.write_text(BLANK_PNL, encoding="utf-8")
    sidecar = sidecar_path_for(pnl)
    sidecar.write_text(
        json.dumps({"version": 1, "pnl": "solo.pnl", "expected": None}),
        encoding="utf-8",
    )
    sample = load_sample(sidecar)
    assert sample.pnl_path == pnl.resolve()
    assert sample.expected_path is None


def test_load_rejects_unknown_type(tmp_path: Path):
    path = tmp_path / "notes.txt"
    path.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError, match="Open a .pnl"):
        load_sample(path)


def test_load_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_sample(tmp_path / "missing.pnl")


def test_save_requires_destination():
    with pytest.raises(ValueError, match="destination"):
        save_sample(Sample(text=BLANK_PNL))
