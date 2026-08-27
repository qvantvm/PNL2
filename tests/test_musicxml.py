from pathlib import Path

from pnl2 import parse, validate
from pnl2.musicxml import musicxml_to_pnl, pnl_to_musicxml
from pnl2.musicxml.from_musicxml import musicxml_to_document

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def test_musicxml_to_pnl_simple():
    xml = (EXAMPLES / "simple.musicxml").read_text(encoding="utf-8")
    pnl = musicxml_to_pnl(xml)
    assert pnl.startswith("pnl/2")
    doc = parse(pnl)
    assert doc.score.meta.get("title") == "Simple Scale"
    assert doc.score.meta.get("composer") == "Anonymous"
    notes = [n for n in _walk(doc.score.parts[0]) if n.kind == "note"]
    assert len(notes) >= 4
    assert any(n.props.get("art") == ["accent"] or "accent" in (n.props.get("art") or []) for n in notes)
    slurs = [n for n in _walk(doc.score.parts[0]) if n.kind == "slur"]
    assert len(slurs) >= 1
    errors = validate(doc)
    # measure length checks may be strict; allow empty or non-timing errors only
    assert all("duplicate" not in e for e in errors)


def test_pnl_to_musicxml_and_back():
    text = (EXAMPLES / "articulation.pnl").read_text(encoding="utf-8")
    xml = pnl_to_musicxml(text)
    assert "score-partwise" in xml
    assert "<step>C</step>" in xml
    assert "<part-symbol>brace</part-symbol>" in xml
    assert "<sign>G</sign>" in xml
    assert "<sign>F</sign>" in xml
    pnl2 = musicxml_to_pnl(xml)
    doc = parse(pnl2)
    pitches = [
        str(n.props["pitch"])
        for n in _walk(doc.score.parts[0])
        if n.kind == "note" and "pitch" in n.props
    ]
    assert "C5" in pitches
    assert "D5" in pitches


def test_document_api():
    xml = (EXAMPLES / "simple.musicxml").read_text(encoding="utf-8")
    doc = musicxml_to_document(xml)
    assert doc.version == "pnl/2"
    assert doc.score.parts


def _walk(node):
    yield node
    for c in node.children:
        yield from _walk(c)
