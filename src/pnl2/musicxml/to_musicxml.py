"""Convert PNL/2 documents to MusicXML partwise."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from fractions import Fraction
from math import lcm
from typing import Any, Iterable
from xml.etree.ElementTree import Element

from ..ast import Document, Node, Pitch, Position, Ratio
from ..parser import parse
from ..rational import Rational, effective_duration

ACC_TO_ALTER = {"": 0, "b": -1, "bb": -2, "#": 1, "##": 2}

DUR_TO_TYPE = [
    (Rational(4), "long"),
    (Rational(2), "breve"),
    (Rational(1), "whole"),
    (Rational(1, 2), "half"),
    (Rational(1, 4), "quarter"),
    (Rational(1, 8), "eighth"),
    (Rational(1, 16), "16th"),
    (Rational(1, 32), "32nd"),
    (Rational(1, 64), "64th"),
    (Rational(1, 128), "128th"),
]

ART_TO_XML = {
    "staccato": "staccato",
    "staccatissimo": "staccatissimo",
    "tenuto": "tenuto",
    "accent": "accent",
    "marcato": "strong-accent",
    "detached-legato": "detached-legato",
}

CLEF_MAP = {
    "treble": ("G", "2", None),
    "bass": ("F", "4", None),
    "alto": ("C", "3", None),
    "tenor": ("C", "4", None),
    "treble-8va": ("G", "2", "1"),
    "treble-8vb": ("G", "2", "-1"),
    "bass-8vb": ("F", "4", "-1"),
    "percussion": ("percussion", "2", None),
}


def pnl_to_musicxml(source: str | Document) -> str:
    """Convert PNL/2 text or Document to MusicXML 3.1 partwise XML string."""
    doc = source if isinstance(source, Document) else parse(source)
    root = build_score_partwise(doc)
    _indent(root)
    xml = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + xml + "\n"


def build_score_partwise(doc: Document) -> Element:
    root = ET.Element("score-partwise", version="3.1")
    meta = doc.score.meta
    if "title" in meta:
        work = ET.SubElement(root, "work")
        ET.SubElement(work, "work-title").text = str(meta["title"])
    if "composer" in meta:
        ident = ET.SubElement(root, "identification")
        creator = ET.SubElement(ident, "creator", type="composer")
        creator.text = str(meta["composer"])
        enc = ET.SubElement(ident, "encoding")
        ET.SubElement(enc, "software").text = "pnl2"

    part_list = ET.SubElement(root, "part-list")
    for i, part in enumerate(doc.score.parts):
        pid = f"P{i+1}"
        sp = ET.SubElement(part_list, "score-part", id=pid)
        ET.SubElement(sp, "part-name").text = part.id or pid

    for i, part in enumerate(doc.score.parts):
        pid = f"P{i+1}"
        root.append(_convert_part(part, pid))
    return root


def _convert_part(part: Node, pid: str) -> Element:
    part_el = ET.Element("part", id=pid)
    staff_names = _collect_staff_names(part)
    staff_index = {name: idx + 1 for idx, name in enumerate(staff_names)}
    divisions = _choose_divisions(part)

    # Index relations
    ties = [n for n in _iter(part) if n.kind == "tie"]
    slurs = [n for n in _iter(part) if n.kind == "slur"]
    beams = [n for n in _iter(part) if n.kind == "beam"]
    pedals = [n for n in _iter(part) if n.kind == "pedal"]
    tempos = [n for n in _iter(part) if n.kind == "tempo"]
    meters = [n for n in _iter(part) if n.kind == "meter"]
    keys = [n for n in _iter(part) if n.kind == "key"]
    clefs = [n for n in _iter(part) if n.kind == "clef"]
    dynamics = [n for n in _iter(part) if n.kind == "dynamic"]

    tie_start = {t.props.get("from") for t in ties}
    tie_stop = {t.props.get("to") for t in ties}
    slur_start = {s.props.get("from"): i + 1 for i, s in enumerate(slurs)}
    slur_stop = {s.props.get("to"): i + 1 for i, s in enumerate(slurs)}
    beam_map = _beam_membership(beams)

    measures = [c for c in part.children if c.kind == "measure"]
    # Also unwrap notation blocks
    if not measures:
        for child in part.children:
            if child.kind == "notation":
                measures.extend(c for c in child.children if c.kind == "measure")

    for measure in measures:
        mnum = str(measure.number or 1)
        m_el = ET.SubElement(part_el, "measure", number=mnum)
        # attributes once per measure if present
        attrs_needed = any(
            _pos_measure(n.props.get("at")) == measure.number
            for n in meters + keys + clefs
        )
        if measure.number == 1 or attrs_needed:
            attr = ET.SubElement(m_el, "attributes")
            ET.SubElement(attr, "divisions").text = str(divisions)
            if len(staff_names) > 1:
                ET.SubElement(attr, "staves").text = str(len(staff_names))
            for key in keys:
                if _pos_measure(key.props.get("at")) == measure.number:
                    _emit_key(attr, key)
            for meter in meters:
                if _pos_measure(meter.props.get("at")) == measure.number:
                    _emit_time(attr, meter)
            for clef in clefs:
                if _pos_measure(clef.props.get("at")) == measure.number:
                    _emit_clef(attr, clef, staff_index)

        # tempos / dynamics at measure start
        for tempo in tempos:
            if _pos_measure(tempo.props.get("at")) == measure.number:
                _emit_tempo_direction(m_el, tempo)
        for dyn in dynamics:
            if _pos_measure(dyn.props.get("at")) == measure.number:
                _emit_dynamic_direction(m_el, dyn, staff_index)

        for pedal in pedals:
            if _pos_measure(pedal.props.get("from")) == measure.number and _pos_offset(
                pedal.props.get("from")
            ) == Fraction(0):
                _emit_pedal(m_el, "start")

        # Emit notes staff by staff, voice by voice with backups
        staff_nodes = [c for c in measure.children if c.kind == "staff"]
        # Also directions nested in measure
        first = True
        prev_end = Fraction(0)
        for staff in staff_nodes:
            sidx = staff_index.get(staff.id or "RH", 1)
            for voice in [c for c in staff.children if c.kind == "voice"]:
                if not first:
                    # backup to start of measure
                    if prev_end > 0:
                        backup = ET.SubElement(m_el, "backup")
                        ET.SubElement(backup, "duration").text = str(
                            _to_div(prev_end, divisions)
                        )
                cursor = Fraction(0)
                for ev in voice.children:
                    if ev.kind == "grace-group":
                        for g in ev.children:
                            if g.kind == "grace":
                                m_el.append(
                                    _emit_note(
                                        g,
                                        divisions,
                                        sidx,
                                        voice.id or "1",
                                        is_grace=True,
                                        chord=False,
                                        tie_start=False,
                                        tie_stop=False,
                                        slur_start_num=None,
                                        slur_stop_num=None,
                                        beams={},
                                    )
                                )
                        continue
                    if ev.kind == "space":
                        dur = _event_effective(ev)
                        if dur > 0:
                            fwd = ET.SubElement(m_el, "forward")
                            ET.SubElement(fwd, "duration").text = str(
                                _to_div(dur, divisions)
                            )
                            cursor += dur
                        continue
                    if ev.kind == "rest":
                        note = _emit_note(
                            ev,
                            divisions,
                            sidx,
                            voice.id or "1",
                            is_rest=True,
                            tie_start=ev.id in tie_start,
                            tie_stop=ev.id in tie_stop,
                            slur_start_num=slur_start.get(ev.id),
                            slur_stop_num=slur_stop.get(ev.id),
                            beams=beam_map.get(ev.id, {}),
                        )
                        m_el.append(note)
                        cursor += _event_effective(ev)
                        continue
                    if ev.kind == "note":
                        note = _emit_note(
                            ev,
                            divisions,
                            sidx,
                            voice.id or "1",
                            tie_start=ev.id in tie_start,
                            tie_stop=ev.id in tie_stop,
                            slur_start_num=slur_start.get(ev.id),
                            slur_stop_num=slur_stop.get(ev.id),
                            beams=beam_map.get(ev.id, {}),
                        )
                        m_el.append(note)
                        cursor += _event_effective(ev)
                        continue
                    if ev.kind == "chord":
                        tones = [c for c in ev.children if c.kind == "tone"]
                        for ti, tone in enumerate(tones):
                            # duration/augment on chord
                            merged = Node(
                                kind="tone",
                                id=tone.id,
                                props={**tone.props, **{
                                    k: v
                                    for k, v in ev.props.items()
                                    if k in ("dur", "augment", "tuplet", "art", "dynamic", "fermata", "arpeggiate")
                                }},
                            )
                            note = _emit_note(
                                merged,
                                divisions,
                                sidx,
                                voice.id or "1",
                                chord=ti > 0,
                                tie_start=tone.id in tie_start,
                                tie_stop=tone.id in tie_stop,
                                slur_start_num=slur_start.get(tone.id) or (slur_start.get(ev.id) if ti == 0 else None),
                                slur_stop_num=slur_stop.get(tone.id) or (slur_stop.get(ev.id) if ti == 0 else None),
                                beams=beam_map.get(tone.id, {}),
                            )
                            m_el.append(note)
                        cursor += _event_effective(ev)
                prev_end = cursor
                first = False

        for pedal in pedals:
            if _pos_measure(pedal.props.get("to")) == measure.number:
                # emit stop at end if to offset is measure end-ish
                _emit_pedal(m_el, "stop")

    return part_el


def _emit_note(
    ev: Node,
    divisions: int,
    staff: int,
    voice: str,
    *,
    is_rest: bool = False,
    is_grace: bool = False,
    chord: bool = False,
    tie_start: bool = False,
    tie_stop: bool = False,
    slur_start_num: int | None = None,
    slur_stop_num: int | None = None,
    beams: dict[int, str] | None = None,
) -> Element:
    note = ET.Element("note")
    if is_grace:
        grace = ET.SubElement(note, "grace")
        if ev.props.get("type") == "acciaccatura":
            grace.set("slash", "yes")
    if chord:
        ET.SubElement(note, "chord")
    if is_rest or ev.kind == "rest":
        ET.SubElement(note, "rest")
    else:
        pitch = ev.props.get("pitch")
        if isinstance(pitch, Pitch):
            p = ET.SubElement(note, "pitch")
            ET.SubElement(p, "step").text = pitch.letter
            alter = ACC_TO_ALTER.get(pitch.accidental, 0)
            if alter:
                ET.SubElement(p, "alter").text = str(alter)
            ET.SubElement(p, "octave").text = str(pitch.octave)

    written = Rational.from_value(ev.props.get("dur", Rational(0)))
    augment = int(ev.props.get("augment", 0) or 0)
    tuplet = ev.props.get("tuplet")
    tpair = (tuplet.left, tuplet.right) if isinstance(tuplet, Ratio) else None
    eff = (
        effective_duration(written, augment, tpair).to_fraction()
        if not is_grace
        else Fraction(0)
    )
    if not is_grace:
        ET.SubElement(note, "duration").text = str(max(1, _to_div(eff, divisions)) if eff > 0 else 0)
        # MusicXML forbids 0 duration on normal notes; grace has no duration

    # voice as trailing digits if present
    vnum = "".join(ch for ch in voice if ch.isdigit()) or "1"
    ET.SubElement(note, "voice").text = vnum

    type_name, dots_from_type = _dur_to_type(written, augment)
    if type_name:
        ET.SubElement(note, "type").text = type_name
    for _ in range(max(augment, dots_from_type)):
        ET.SubElement(note, "dot")

    if isinstance(tuplet, Ratio):
        tm = ET.SubElement(note, "time-modification")
        ET.SubElement(tm, "actual-notes").text = str(tuplet.left)
        ET.SubElement(tm, "normal-notes").text = str(tuplet.right)

    ET.SubElement(note, "staff").text = str(staff)

    if beams:
        for level in sorted(beams):
            b = ET.SubElement(note, "beam", number=str(level))
            b.text = beams[level]

    notations_needed = (
        tie_start
        or tie_stop
        or slur_start_num
        or slur_stop_num
        or ev.props.get("art")
        or ev.props.get("finger")
        or ev.props.get("ornament")
        or ev.props.get("fermata")
        or ev.props.get("arpeggiate")
    )
    if notations_needed:
        notations = ET.SubElement(note, "notations")
        if tie_start:
            ET.SubElement(notations, "tied", type="start")
            # also legacy
        if tie_stop:
            ET.SubElement(notations, "tied", type="stop")
        if slur_start_num:
            ET.SubElement(notations, "slur", type="start", number=str(slur_start_num))
        if slur_stop_num:
            ET.SubElement(notations, "slur", type="stop", number=str(slur_stop_num))
        arts = ev.props.get("art") or []
        if arts:
            articulations = ET.SubElement(notations, "articulations")
            for a in arts:
                tag = ART_TO_XML.get(a)
                if tag:
                    ET.SubElement(articulations, tag)
        finger = ev.props.get("finger")
        if finger:
            technical = ET.SubElement(notations, "technical")
            ET.SubElement(technical, "fingering").text = str(finger)
        orn = ev.props.get("ornament")
        if orn:
            ornaments = ET.SubElement(notations, "ornaments")
            tag = {
                "trill": "trill-mark",
                "mordent": "mordent",
                "inverted-mordent": "inverted-mordent",
                "turn": "turn",
                "inverted-turn": "inverted-turn",
            }.get(str(orn))
            if tag:
                ET.SubElement(ornaments, tag)
        if ev.props.get("fermata"):
            ET.SubElement(notations, "fermata")
        arp = ev.props.get("arpeggiate")
        if arp and arp not in ("none", "non-arpeggiate"):
            ET.SubElement(notations, "arpeggiate")

    if tie_start:
        ET.SubElement(note, "tie", type="start")
    if tie_stop:
        ET.SubElement(note, "tie", type="stop")

    if ev.props.get("visible") is False:
        note.set("print-object", "no")

    return note


def _dur_to_type(written: Rational, augment: int) -> tuple[str | None, int]:
    for base, name in DUR_TO_TYPE:
        if written == base:
            return name, 0
    # try undotting
    for dots in (1, 2, 3):
        # written = base * (2 - 1/2^dots) => base = written / factor
        factor = Rational(2) - Rational(1, 2**dots)
        try:
            base = written / factor
        except Exception:  # noqa: BLE001
            continue
        for b, name in DUR_TO_TYPE:
            if base == b:
                return name, dots
    # fallback nearest
    if written.numerator == 0:
        return None, 0
    best = min(DUR_TO_TYPE, key=lambda x: abs(float(x[0] - written)))
    return best[1], 0


def _event_effective(ev: Node) -> Fraction:
    dur = ev.props.get("dur", Rational(0))
    augment = int(ev.props.get("augment", 0) or 0)
    tuplet = ev.props.get("tuplet")
    tpair = (tuplet.left, tuplet.right) if isinstance(tuplet, Ratio) else None
    return effective_duration(dur, augment, tpair).to_fraction()


def _choose_divisions(part: Node) -> int:
    dens = [1]
    for node in _iter(part):
        for key in ("dur", "beat-unit"):
            val = node.props.get(key)
            if isinstance(val, Rational):
                dens.append(val.denominator)
        aug = int(node.props.get("augment", 0) or 0)
        if aug:
            dens.append(2**aug)
        tuplet = node.props.get("tuplet")
        if isinstance(tuplet, Ratio):
            dens.append(tuplet.left)
    # divisions is ticks per quarter note
    # whole note = 4 quarters → denominator relative to whole; divisions = lcm(dens)/1 * something
    # duration_div = dur_whole * divisions * 4
    den_lcm = 1
    for d in dens:
        den_lcm = lcm(den_lcm, d)
    # Ensure quarter = divisions ticks → need den_lcm dividing properly
    divisions = den_lcm
    # If denominators are of whole notes, quarter needs divisions such that 1/4 * 4 * divisions is int
    # = divisions, so any int works; for 1/den, duration = 4*divisions/den → divisions multiple of den/gcd(4,den)
    divisions = max(1, den_lcm)
    return divisions


def _to_div(whole: Fraction, divisions: int) -> int:
    ticks = whole * divisions * 4
    return int(ticks)


def _collect_staff_names(part: Node) -> list[str]:
    names: list[str] = []
    for n in _iter(part):
        if n.kind == "staff" and n.id and n.id not in names:
            names.append(n.id)
    return names or ["RH"]


def _beam_membership(beams: list[Node]) -> dict[str, dict[int, str]]:
    """Map note id -> {level: begin|continue|end}."""
    out: dict[str, dict[int, str]] = defaultdict(dict)
    for beam in beams:
        notes = beam.props.get("notes") or []
        level = int(beam.props.get("level", 1) or 1)
        if not isinstance(notes, list) or len(notes) < 2:
            continue
        for i, nid in enumerate(notes):
            if i == 0:
                state = "begin"
            elif i == len(notes) - 1:
                state = "end"
            else:
                state = "continue"
            out[str(nid)][level] = state
    return out


def _pos_measure(pos: Any) -> int | None:
    if isinstance(pos, Position):
        return pos.measure
    return None


def _pos_offset(pos: Any) -> Fraction:
    if isinstance(pos, Position):
        return pos.offset.to_fraction()
    return Fraction(0)


def _emit_key(attr: Element, key: Node) -> None:
    tonic = str(key.props.get("tonic", "C"))
    mode = str(key.props.get("mode", "major"))
    fifths = _tonic_to_fifths(tonic, mode)
    kel = ET.SubElement(attr, "key")
    ET.SubElement(kel, "fifths").text = str(fifths)
    ET.SubElement(kel, "mode").text = "minor" if mode in ("minor", "aeolian") else "major"


def _tonic_to_fifths(tonic: str, mode: str) -> int:
    major = {
        "C": 0,
        "G": 1,
        "D": 2,
        "A": 3,
        "E": 4,
        "B": 5,
        "F#": 6,
        "C#": 7,
        "F": -1,
        "Bb": -2,
        "Eb": -3,
        "Ab": -4,
        "Db": -5,
        "Gb": -6,
        "Cb": -7,
    }
    if mode in ("minor", "aeolian"):
        # convert minor tonic to relative major
        rel = {
            "A": "C",
            "E": "G",
            "B": "D",
            "F#": "A",
            "C#": "E",
            "G#": "B",
            "D#": "F#",
            "A#": "C#",
            "D": "F",
            "G": "Bb",
            "C": "Eb",
            "F": "Ab",
            "Bb": "Db",
            "Eb": "Gb",
            "Ab": "Cb",
        }
        tonic = rel.get(tonic, tonic)
    return major.get(tonic, 0)


def _emit_time(attr: Element, meter: Node) -> None:
    beats = int(meter.props.get("beats", 4))
    unit = Rational.from_value(meter.props.get("beat-unit", Rational(1, 4)))
    # beat-type is denominator of unit as note value: 1/4 → 4
    beat_type = unit.denominator if unit.numerator == 1 else int(1 / float(unit))
    tel = ET.SubElement(attr, "time")
    ET.SubElement(tel, "beats").text = str(beats)
    ET.SubElement(tel, "beat-type").text = str(beat_type)


def _emit_clef(attr: Element, clef: Node, staff_index: dict[str, int]) -> None:
    ctype = str(clef.props.get("type", "treble"))
    sign, line, oc = CLEF_MAP.get(ctype, ("G", "2", None))
    staff = clef.props.get("staff")
    c = ET.SubElement(attr, "clef")
    if isinstance(staff, str) and staff in staff_index:
        c.set("number", str(staff_index[staff]))
    ET.SubElement(c, "sign").text = sign
    ET.SubElement(c, "line").text = line
    if oc is not None:
        ET.SubElement(c, "clef-octave-change").text = oc


def _emit_tempo_direction(measure: Element, tempo: Node) -> None:
    direction = ET.SubElement(measure, "direction", placement="above")
    dtype = ET.SubElement(direction, "direction-type")
    metro = ET.SubElement(dtype, "metronome")
    beat = Rational.from_value(tempo.props.get("beat", Rational(1, 4)))
    type_name, _ = _dur_to_type(beat, int(tempo.props.get("augment", 0) or 0))
    ET.SubElement(metro, "beat-unit").text = type_name or "quarter"
    for _ in range(int(tempo.props.get("augment", 0) or 0)):
        ET.SubElement(metro, "beat-unit-dot")
    ET.SubElement(metro, "per-minute").text = str(int(tempo.props.get("bpm", 120)))
    sound = ET.SubElement(direction, "sound", tempo=str(tempo.props.get("bpm", 120)))


def _emit_dynamic_direction(
    measure: Element, dyn: Node, staff_index: dict[str, int]
) -> None:
    direction = ET.SubElement(measure, "direction", placement="below")
    dtype = ET.SubElement(direction, "direction-type")
    dynamics = ET.SubElement(dtype, "dynamics")
    value = str(dyn.props.get("value", "mf"))
    ET.SubElement(dynamics, value)
    staff = dyn.props.get("staff")
    if isinstance(staff, str) and staff in staff_index:
        ET.SubElement(direction, "staff").text = str(staff_index[staff])


def _emit_pedal(measure: Element, ptype: str) -> None:
    direction = ET.SubElement(measure, "direction")
    dtype = ET.SubElement(direction, "direction-type")
    ET.SubElement(dtype, "pedal", type=ptype)


def _iter(node: Node) -> Iterable[Node]:
    yield node
    for child in node.children:
        yield from _iter(child)


def _indent(elem: Element, level: int = 0) -> None:
    pad = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        for child in elem:
            _indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = pad + "  "
        if not elem[-1].tail or not elem[-1].tail.strip():
            elem[-1].tail = pad
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = pad
