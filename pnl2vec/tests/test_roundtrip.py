from pnl2vec.pnl import parse_pnl, serialize_pnl


def _score_signature(doc):
    part = doc.score.parts[0]
    measures = [c for c in part.children if c.kind == "measure"]
    notes = []
    for m in measures:
        for staff in m.children:
            if staff.kind != "staff":
                continue
            for voice in staff.children:
                if voice.kind != "voice":
                    continue
                for ev in voice.children:
                    if ev.kind in {"note", "rest", "chord"}:
                        notes.append((ev.kind, ev.id, str(ev.get("pitch")), str(ev.get("dur"))))
    return (doc.version, len(measures), notes)


def test_parse_serialize_roundtrip(tiny_scale_text):
    d1 = parse_pnl(tiny_scale_text)
    text2 = serialize_pnl(d1)
    d2 = parse_pnl(text2)
    text3 = serialize_pnl(d2)
    d3 = parse_pnl(text3)
    assert _score_signature(d1) == _score_signature(d3)


def test_articulation_roundtrip(articulation_text):
    d1 = parse_pnl(articulation_text)
    d2 = parse_pnl(serialize_pnl(d1))
    assert _score_signature(d1) == _score_signature(d2)
