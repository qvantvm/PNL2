from pathlib import Path

from pnl2 import parse, serialize, validate

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_parse_articulation_example():
    text = (EXAMPLES / "articulation.pnl").read_text(encoding="utf-8")
    doc = parse(text)
    assert doc.version == "pnl/2"
    assert doc.score.meta["title"] == "Articulation Example"
    part = doc.score.parts[0]
    assert part.id == "piano"
    errors = validate(doc)
    assert errors == []


def test_roundtrip_serialize():
    text = (EXAMPLES / "articulation.pnl").read_text(encoding="utf-8")
    doc = parse(text)
    out = serialize(doc)
    doc2 = parse(out)
    assert doc2.score.meta["title"] == "Articulation Example"
    # find dotted note
    note = None
    for part in doc2.score.parts:
        for node in _walk(part):
            if node.kind == "note" and node.id == "n1":
                note = node
    assert note is not None
    assert note.props["augment"] == 1
    assert "staccato" not in str(note.props.get("art", []))


def test_pitch_and_lists():
    text = """pnl/2
score {
part p instrument=piano staves=1 {
measure 1 {
staff RH {
voice RH1 {
note n1 pitch=F#4 dur=1/8 art=[accent,tenuto]
note n2 pitch=Bb3 dur=1/4
}
}
}
}
}
"""
    doc = parse(text)
    notes = [n for n in _walk(doc.score.parts[0]) if n.kind == "note"]
    assert str(notes[0].props["pitch"]) == "F#4"
    assert notes[0].props["art"] == ["accent", "tenuto"]
    assert str(notes[1].props["pitch"]) == "Bb3"


def test_chord_and_tie():
    text = """pnl/2
score {
part p instrument=piano staves=1 {
measure 1 {
staff RH {
voice RH1 {
chord c1 dur=1/4 {
tone n1 pitch=C4
tone n2 pitch=E4
}
note n3 pitch=C4 dur=1/4
}
}
}
tie t1 from=n1 to=n3
}
}
"""
    doc = parse(text)
    assert validate(doc) == []


def _walk(node):
    yield node
    for c in node.children:
        yield from _walk(c)
