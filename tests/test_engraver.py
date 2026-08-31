import re
import sys
import types
from pathlib import Path

import pytest

from pnl2 import parse
from pnl2.cli import main
from pnl2.engraver import (
    EngraverError,
    combine_svgs,
    engrave,
    engrave_svg,
    flatten_nested_svgs,
    infer_format,
    outline_smufl_text,
    reset_verovio_toolkit,
    wrap_html,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

TINY_PNL = """pnl/2
score {
    meta { title="Tiny" }
    part p instrument=piano staves=1 {
        measure 1 {
            staff RH {
                voice v {
                    note n1 pitch=C4 dur=1/4
                }
            }
        }
    }
}
"""

PAGE_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 50" '
    'width="100" height="50">'
    '<g class="staff"><g class="note">n</g></g>'
    "</svg>"
)


class FakeToolkit:
    def __init__(self, pages: list[str] | None = None, load_ok: bool = True):
        self.options: dict = {}
        self.data: str | None = None
        self.pages = pages if pages is not None else [PAGE_SVG]
        self.load_ok = load_ok

    def setOptions(self, options):
        self.options.update(options)

    def loadData(self, data):
        self.data = data
        return self.load_ok

    def getPageCount(self):
        return len(self.pages)

    def renderToSVG(self, page, xmlDeclaration=False):
        return self.pages[page - 1]


def _install_fake_verovio(monkeypatch, toolkit_factory=None):
    reset_verovio_toolkit()
    factory = toolkit_factory or FakeToolkit
    fake = types.ModuleType("verovio")
    fake.toolkit = factory
    monkeypatch.setitem(sys.modules, "verovio", fake)
    return fake


def test_infer_format_defaults_to_svg():
    assert infer_format(None, None) == "svg"


def test_infer_format_from_suffix_and_override():
    assert infer_format(Path("score.png"), None) == "png"
    assert infer_format(Path("score.htm"), None) == "html"
    assert infer_format(Path("score.bin"), "svg") == "svg"
    with pytest.raises(EngraverError, match="Cannot infer"):
        infer_format(Path("score.bin"), None)
    with pytest.raises(EngraverError, match="Unsupported"):
        infer_format(None, "midi")


def test_combine_svgs_single_page_unchanged():
    assert combine_svgs([PAGE_SVG]) == PAGE_SVG


def test_combine_svgs_stacks_pages():
    combined = combine_svgs([PAGE_SVG, PAGE_SVG], gap=10)
    assert combined.count('class="staff"') == 2
    assert 'viewBox="0 0 100 110"' in combined


def test_flatten_nested_svgs_unwraps_verovio_inner():
    nested = (
        '<svg width="210px" height="61px" xmlns="http://www.w3.org/2000/svg">'
        "<defs><g id='E050'/></defs>"
        '<svg class="definition-scale" viewBox="0 0 2100 610">'
        '<g class="staff"><path d="M0 0"/></g>'
        "</svg>"
        "</svg>"
    )
    flat = flatten_nested_svgs(nested)
    assert re.search(r"<svg[^>]*>.*<svg", flat, re.I | re.S) is None
    assert 'class="definition-scale"' in flat
    assert 'viewBox="0 0 2100 610"' in flat
    assert 'fill="white"' in flat
    assert 'class="staff"' in flat


def test_flatten_nested_svgs_adds_white_page_to_flat_svg():
    flat = flatten_nested_svgs(PAGE_SVG)
    assert 'fill="white"' in flat
    assert 'class="staff"' in flat


TEMPO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">'
    "<defs></defs>"
    '<g class="tempo">'
    '<text x="100" y="200" font-size="0px">'
    '<tspan font-family="Leipzig" font-size="1000px">\ueca5</tspan>'
    '<tspan font-size="400px"> = 120</tspan>'
    "</text>"
    "</g>"
    "</svg>"
)


def test_outline_smufl_text_replaces_leipzig_tempo():
    out = outline_smufl_text(TEMPO_SVG)
    assert "\ueca5" not in out
    assert 'xlink:href="#smufl-ECA5"' in out
    assert 'id="smufl-ECA5"' in out
    assert 'dx="302"' in out
    assert " = 120" in out


def test_wrap_html_escapes_title():
    doc = wrap_html([PAGE_SVG], title='A & B <score>')
    assert "A &amp; B &lt;score&gt;" in doc
    assert PAGE_SVG in doc
    assert 'class="page"' in doc


def _block_import(monkeypatch, name: str) -> None:
    import builtins

    monkeypatch.delitem(sys.modules, name, raising=False)
    real = builtins.__import__

    def blocked(mod, globals=None, locals=None, fromlist=(), level=0):
        if mod == name:
            raise ImportError("missing")
        return real(mod, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked)


def test_missing_verovio_message(monkeypatch):
    _block_import(monkeypatch, "verovio")
    with pytest.raises(EngraverError, match=r'pip install "pnl2\[engrave\]"'):
        engrave_svg(TINY_PNL)


def test_missing_cairosvg_message(monkeypatch, tmp_path):
    _install_fake_verovio(monkeypatch)
    _block_import(monkeypatch, "cairosvg")
    with pytest.raises(EngraverError, match="CairoSVG"):
        engrave(TINY_PNL, tmp_path / "out.png")


def test_engrave_svg_mocked(monkeypatch):
    _install_fake_verovio(monkeypatch)
    pages = engrave_svg(TINY_PNL)
    assert len(pages) == 1
    assert "staff" in pages[0]
    assert "note" in pages[0]


def test_engrave_accepts_document_and_writes_svg(monkeypatch, tmp_path):
    captured = {}

    def factory():
        tk = FakeToolkit()
        captured["toolkit"] = tk
        return tk

    _install_fake_verovio(monkeypatch, factory)
    doc = parse(TINY_PNL)
    dest = tmp_path / "out.svg"
    result = engrave(doc, dest)
    assert result == dest
    assert dest.read_text(encoding="utf-8") == PAGE_SVG
    assert captured["toolkit"].options["inputFrom"] == "xml"
    assert captured["toolkit"].options["adjustPageHeight"] is True
    assert captured["toolkit"].data and "score-partwise" in captured["toolkit"].data


def test_engrave_html_and_stdout_svg(monkeypatch, tmp_path):
    _install_fake_verovio(monkeypatch)
    html_path = tmp_path / "score.html"
    engrave(TINY_PNL, html_path)
    text = html_path.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")
    assert "Tiny" in text
    assert PAGE_SVG in text

    svg = engrave(TINY_PNL)
    assert svg == PAGE_SVG


def test_engrave_png_single_and_multi(monkeypatch, tmp_path):
    _install_fake_verovio(monkeypatch)
    written = []

    fake_cairo = types.ModuleType("cairosvg")

    def svg2png(*, bytestring, write_to, scale, background_color=None):
        written.append((write_to, scale, bytestring, background_color))
        Path(write_to).write_bytes(b"png")

    fake_cairo.svg2png = svg2png
    monkeypatch.setitem(sys.modules, "cairosvg", fake_cairo)

    single = tmp_path / "one.png"
    assert engrave(TINY_PNL, single, scale=3) == single
    assert single.read_bytes() == b"png"
    assert written[0][1] == 3.0
    assert written[0][3] == "white"

    def factory():
        return FakeToolkit([PAGE_SVG, PAGE_SVG])

    _install_fake_verovio(monkeypatch, factory)
    multi = tmp_path / "many.png"
    paths = engrave(TINY_PNL, multi)
    assert paths == [tmp_path / "many-1.png", tmp_path / "many-2.png"]
    assert all(p.read_bytes() == b"png" for p in paths)


def test_engrave_png_requires_path(monkeypatch):
    _install_fake_verovio(monkeypatch)
    with pytest.raises(EngraverError, match="PNG output requires"):
        engrave(TINY_PNL, format="png")


def test_cli_engrave_svg(monkeypatch, tmp_path):
    _install_fake_verovio(monkeypatch)
    src = tmp_path / "in.pnl"
    src.write_text(TINY_PNL, encoding="utf-8")
    dest = tmp_path / "out.svg"
    assert main(["engrave", str(src), "-o", str(dest)]) == 0
    assert dest.read_text(encoding="utf-8") == PAGE_SVG


def test_cli_engrave_stdout(monkeypatch, tmp_path, capsys):
    _install_fake_verovio(monkeypatch)
    src = tmp_path / "in.pnl"
    src.write_text(TINY_PNL, encoding="utf-8")
    assert main(["engrave", str(src)]) == 0
    assert PAGE_SVG in capsys.readouterr().out


def test_cli_missing_verovio(monkeypatch, tmp_path, capsys):
    src = tmp_path / "in.pnl"
    src.write_text(TINY_PNL, encoding="utf-8")

    def boom():
        raise EngraverError('Verovio is required to engrave scores. Install with: pip install "pnl2[engrave]"')

    monkeypatch.setattr("pnl2.engraver._import_verovio", boom)
    assert main(["engrave", str(src), "-o", str(tmp_path / "x.svg")]) == 1
    err = capsys.readouterr().err
    assert "pip install" in err
    assert "engrave" in err


@pytest.mark.parametrize("name", ["simple.pnl", "articulation.pnl", "polyphonic.pnl"])
def test_engrave_examples_svg(name):
    pytest.importorskip("verovio")
    pages = engrave_svg(EXAMPLES / name)
    assert pages
    joined = "\n".join(pages)
    assert "<svg" in joined
    assert "staff" in joined.lower() or "note" in joined.lower()
    if name != "polyphonic.pnl":
        assert "\ueca5" not in joined
        assert 'id="smufl-ECA5"' in joined


def test_engrave_roman_numerals_visible():
    pytest.importorskip("verovio")
    from pnl2.musicxml.to_musicxml import pnl_to_musicxml

    pnl = """pnl/2
score {
    meta { profile=[core,notation,analysis] }
    part piano instrument=piano staves=2 {
        meter at=1:0 beats=4 beat-unit=1/4
        key at=1:0 tonic=C mode=major
        measure 1 {
            staff RH {
                voice RH1 {
                    note n1 pitch=G4 dur=1/2
                    note n2 pitch=C5 dur=1/2
                }
            }
            staff LH {
                voice LH1 {
                    note n3 pitch=G2 dur=1/2
                    note n4 pitch=C3 dur=1/2
                }
            }
        }
        roman rn1 from=1:0 to=1:1/2 degree=5 quality=dominant seventh=true key=C:major
        roman rn2 from=1:1/2 to=1:1 degree=1 quality=major key=C:major
    }
}
"""
    xml = pnl_to_musicxml(pnl)
    assert 'text="V7"' in xml
    assert "<function>V7</function>" in xml
    pages = engrave_svg(pnl)
    joined = "\n".join(pages)
    assert "V7" in joined
    assert re.search(r">I<", joined)


def test_render_reused_across_threads():
    pytest.importorskip("verovio")
    import threading

    from pnl2.engraver import render_svg_pages
    from pnl2.musicxml.to_musicxml import pnl_to_musicxml

    xml = pnl_to_musicxml(TINY_PNL)
    errors: list[str] = []

    def run() -> None:
        try:
            pages = render_svg_pages(xml)
            assert pages and "<svg" in pages[0]
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

    threads = [threading.Thread(target=run) for _ in range(3)]
    for thread in threads:
        thread.start()
        thread.join()
    assert errors == []
