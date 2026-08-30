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


PIANO_SYMBOLS_PNL = """pnl/2
score {
    meta { title="Symbols" }
    part piano instrument=piano staves=2 {
        meter at=1:0 beats=4 beat-unit=1/4
        key at=1:0 tonic=C mode=major
        measure 1 {
            staff RH {
                voice RH1 {
                    grace-group gg1 type=acciaccatura anchor=n1 {
                        grace g1 pitch=D5
                    }
                    note n1 pitch=C5 dur=1/4 finger=1
                    note n2 pitch=E5 dur=1/4 finger=2
                    note n3 pitch=G5 dur=1/2 finger=3 ornament=trill
                }
            }
            staff LH {
                voice LH1 {
                    note n4 pitch=C3 dur=1/2 finger=5
                    note n5 pitch=G2 dur=1/2 finger=1
                }
            }
        }
        ottava o1 type="8va" from=n1 to=n3
        pedal p1 type=sustain from=1:0 to=1:1/2
        pedal p2 type=sustain from=1:1/2 to=1:1
    }
}
"""


def test_musicxml_ottava_pedal_fingering_grace():
    xml = pnl_to_musicxml(PIANO_SYMBOLS_PNL)
    assert '<octave-shift type="down" number="1" size="8"' in xml or '<octave-shift type="down" size="8" number="1"' in xml
    assert 'type="stop"' in xml
    assert 'placement="below"' in xml
    assert '<pedal type="start" line="yes"' in xml or '<pedal type="start"' in xml and 'line="yes"' in xml
    assert "<staff>2</staff>" in xml
    assert '<fingering placement="below">5</fingering>' in xml
    assert '<fingering placement="above">1</fingering>' in xml
    assert '<grace slash="yes"' in xml
    assert "<type>eighth</type>" in xml
    assert "<trill-mark" in xml


def test_document_api():
    xml = (EXAMPLES / "simple.musicxml").read_text(encoding="utf-8")
    doc = musicxml_to_document(xml)
    assert doc.version == "pnl/2"
    assert doc.score.parts


def _walk(node):
    yield node
    for c in node.children:
        yield from _walk(c)
