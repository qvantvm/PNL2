"""Convert MusicXML (partwise or timewise) to PNL/2."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any
from xml.etree.ElementTree import Element

from ..ast import Document, Node, Pitch, Position, Ratio, Score
from ..rational import Rational, effective_duration
from ..serializer import serialize

# MusicXML often uses namespaces
NS_CANDIDATES = (
    "",
    "{http://www.musicxml.org/xsd/MusicXML}",
)


def _local(tag: str) -> str:
    if tag.startswith("{"):
        return tag.rsplit("}", 1)[-1]
    return tag


def _find(parent: Element, name: str) -> Element | None:
    for child in parent:
        if _local(child.tag) == name:
            return child
    return None


def _findall(parent: Element, name: str) -> list[Element]:
    return [c for c in parent if _local(c.tag) == name]


def _findtext(parent: Element, name: str, default: str | None = None) -> str | None:
    el = _find(parent, name)
    if el is None or el.text is None:
        return default
    return el.text


def _all_depth(parent: Element, name: str) -> list[Element]:
    out: list[Element] = []
    for el in parent.iter():
        if _local(el.tag) == name:
            out.append(el)
    return out


ALTER_TO_ACC = {-2: "bb", -1: "b", 0: "", 1: "#", 2: "##"}

ARTICULATION_MAP = {
    "staccato": "staccato",
    "staccatissimo": "staccatissimo",
    "tenuto": "tenuto",
    "accent": "accent",
    "strong-accent": "marcato",
    "detached-legato": "detached-legato",
    "spiccato": "staccatissimo",
}

DYNAMIC_NAMES = {
    "pppp",
    "ppp",
    "pp",
    "p",
    "mp",
    "mf",
    "f",
    "ff",
    "fff",
    "ffff",
    "sf",
    "sfz",
    "sffz",
    "fp",
    "rfz",
    "fz",
}


@dataclass
class _NoteEvent:
    id: str
    pitch: Pitch | None
    dur: Rational
    augment: int = 0
    tuplet: Ratio | None = None
    at: Rational = field(default_factory=lambda: Rational(0))
    staff: int = 1
    voice: str = "1"
    is_rest: bool = False
    is_chord: bool = False
    is_grace: bool = False
    grace_type: str | None = None
    finger: int | None = None
    arts: list[str] = field(default_factory=list)
    dynamic: str | None = None
    tie_start: bool = False
    tie_stop: bool = False
    slur_starts: list[int] = field(default_factory=list)
    slur_stops: list[int] = field(default_factory=list)
    beams: dict[int, str] = field(default_factory=dict)  # number -> begin/continue/end
    visible: bool = True


def musicxml_to_pnl(
    source: str | bytes | Element,
    *,
    staff_map: dict[int, str] | None = None,
) -> str:
    """Convert MusicXML to a canonical PNL/2 document string."""
    doc = musicxml_to_document(source, staff_map=staff_map)
    return serialize(doc)


def musicxml_to_document(
    source: str | bytes | Element,
    *,
    staff_map: dict[int, str] | None = None,
) -> Document:
    if isinstance(source, Element):
        root = source
    else:
        if isinstance(source, bytes):
            root = ET.fromstring(source)
        else:
            text = source
            if text.lstrip().startswith("<?xml") or "<" in text[:200]:
                root = ET.fromstring(text)
            else:
                # path
                root = ET.parse(text).getroot()

    root_name = _local(root.tag)
    if root_name == "score-timewise":
        root = _timewise_to_partwise(root)
        root_name = "score-partwise"
    if root_name != "score-partwise":
        raise ValueError(f"unsupported MusicXML root {_local(root.tag)}")

    staff_map = staff_map or {1: "RH", 2: "LH"}
    meta = _extract_meta(root)
    part_list = _find(root, "part-list")
    part_names: dict[str, str] = {}
    if part_list is not None:
        for sp in _findall(part_list, "score-part"):
            pid = sp.attrib.get("id", "P1")
            name = _findtext(sp, "part-name", pid) or pid
            part_names[pid] = name

    parts: list[Node] = []
    id_gen = _IdGen()

    for part_el in _findall(root, "part"):
        pid = part_el.attrib.get("id", "P1")
        part_node = _convert_part(
            part_el,
            part_id=_slug(part_names.get(pid, pid)),
            instrument=part_names.get(pid, "piano"),
            staff_map=staff_map,
            id_gen=id_gen,
        )
        parts.append(part_node)

    score = Score(meta=meta, parts=parts)
    return Document(version="pnl/2", score=score)


def _extract_meta(root: Element) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    work = _find(root, "work")
    if work is not None:
        title = _findtext(work, "work-title")
        if title:
            meta["title"] = title
    ident = _find(root, "identification")
    if ident is not None:
        for creator in _findall(ident, "creator"):
            if creator.attrib.get("type") == "composer" and creator.text:
                meta["composer"] = creator.text
                break
        else:
            c = _find(ident, "creator")
            if c is not None and c.text:
                meta["composer"] = c.text
    movement = _findtext(root, "movement-title")
    if movement and "title" not in meta:
        meta["title"] = movement
    meta.setdefault("profile", ["core", "notation"])
    return meta


def _convert_part(
    part_el: Element,
    *,
    part_id: str,
    instrument: str,
    staff_map: dict[int, str],
    id_gen: "_IdGen",
) -> Node:
    staves_used: set[int] = {1}
    divisions = 1
    measure_nodes: list[Node] = []
    top_relations: list[Node] = []
    open_ties: dict[tuple[str, int], str] = {}  # (pitch, staff) -> note id
    open_slurs: dict[int, str] = {}  # number -> from id
    open_pedals: dict[str, tuple[Position, str]] = {}  # type -> (from, id)
    beam_active: dict[tuple[str, int], list[str]] = {}  # (voice, level) -> note ids

    # Directions collected per measure
    for measure_el in _findall(part_el, "measure"):
        mnum = int(measure_el.attrib.get("number", "1") or "1")
        # attributes
        attrs = _find(measure_el, "attributes")
        if attrs is not None:
            div = _findtext(attrs, "divisions")
            if div:
                divisions = int(div)
            staves_el = _findtext(attrs, "staves")
            if staves_el:
                for s in range(1, int(staves_el) + 1):
                    staves_used.add(s)

        # Collect notes by (staff, voice)
        voices: dict[tuple[int, str], list[_NoteEvent]] = defaultdict(list)
        cursor_by_voice: dict[tuple[int, str], Fraction] = defaultdict(lambda: Fraction(0))
        measure_dirs: list[Node] = []

        # First pass: attributes directions at start
        if attrs is not None:
            measure_dirs.extend(
                _attrs_to_nodes(attrs, mnum, staff_map, id_gen, staves_used)
            )

        for child in measure_el:
            tag = _local(child.tag)
            if tag == "note":
                ev = _parse_note(child, divisions, id_gen)
                staves_used.add(ev.staff)
                key = (ev.staff, ev.voice)
                if not ev.is_chord and not ev.is_grace:
                    ev.at = Rational.from_value(cursor_by_voice[key])
                elif ev.is_chord and voices[key]:
                    ev.at = voices[key][-1].at
                elif ev.is_grace:
                    ev.at = Rational.from_value(cursor_by_voice[key])
                voices[key].append(ev)
                if not ev.is_chord and not ev.is_grace:
                    cursor_by_voice[key] = ev.at.to_fraction() + effective_duration(
                        ev.dur,
                        ev.augment,
                        (ev.tuplet.left, ev.tuplet.right) if ev.tuplet else None,
                    ).to_fraction()
            elif tag == "backup":
                dur = int(_findtext(child, "duration", "0") or "0")
                # backup applies to the "current" musical position; approximate by
                # subtracting from all voice cursors max — MusicXML uses a single cursor.
                delta = Fraction(dur, divisions * 4)
                # Subtract from a global sense: reduce each voice that is ahead
                # Simpler: track global cursor
                for k in list(cursor_by_voice.keys()):
                    cursor_by_voice[k] = max(Fraction(0), cursor_by_voice[k] - delta)
            elif tag == "forward":
                dur = int(_findtext(child, "duration", "0") or "0")
                delta = Fraction(dur, divisions * 4)
                # forward is usually voice-specific via following notes; apply to last voice
                # MusicXML forward advances current position for next notes without backup.
                # We store as space when we know voice — handled loosely via note offsets.
                pass
            elif tag == "direction":
                measure_dirs.extend(
                    _direction_to_nodes(
                        child, mnum, divisions, staff_map, id_gen, open_pedals, cursor_by_voice
                    )
                )
            elif tag == "harmony":
                hs = _harmony_to_node(child, mnum, id_gen)
                if hs:
                    measure_dirs.append(hs)

        # Build staff/voice tree for measure
        staff_ids = sorted(staves_used)
        staff_nodes: list[Node] = []
        for s in staff_ids:
            sname = staff_map.get(s, f"S{s}")
            voice_keys = sorted({v for (st, v) in voices if st == s})
            if not voice_keys:
                # empty staff with rest for completeness? skip empty
                continue
            voice_nodes: list[Node] = []
            for v in voice_keys:
                events = voices[(s, v)]
                vname = f"{sname}{v}" if not v.startswith(sname) else v
                # Prefer RH1 / LH1 style
                vname = f"{sname}{v}"
                children = _events_to_nodes(
                    events,
                    id_gen,
                    open_ties,
                    open_slurs,
                    beam_active,
                    top_relations,
                    measure_number=mnum,
                )
                voice_nodes.append(Node(kind="voice", id=vname, children=children))
            if voice_nodes:
                staff_nodes.append(Node(kind="staff", id=sname, children=voice_nodes))

        # If no notes at all, still emit empty measure structure
        if not staff_nodes:
            sname = staff_map.get(1, "RH")
            staff_nodes.append(
                Node(
                    kind="staff",
                    id=sname,
                    children=[
                        Node(
                            kind="voice",
                            id=f"{sname}1",
                            children=[
                                Node(
                                    kind="rest",
                                    id=id_gen.next("r"),
                                    props={"dur": Rational(1)},
                                )
                            ],
                        )
                    ],
                )
            )

        measure_nodes.append(
            Node(kind="measure", number=mnum, children=staff_nodes + measure_dirs)
        )

    # Close any open pedals at end
    for ptype, (frm, pid) in list(open_pedals.items()):
        top_relations.append(
            Node(
                kind="pedal",
                id=pid,
                props={"type": ptype, "from": frm, "to": Position(frm.measure, Rational(1)), "depth": 1},
            )
        )
        del open_pedals[ptype]

    part = Node(
        kind="part",
        id=part_id,
        props={
            "instrument": _slug(instrument).replace("_", "-")[:32] or "piano",
            "staves": max(staves_used) if staves_used else 1,
        },
        children=measure_nodes + top_relations,
    )
    return part


def _parse_note(note_el: Element, divisions: int, id_gen: "_IdGen") -> _NoteEvent:
    is_chord = _find(note_el, "chord") is not None
    is_grace = _find(note_el, "grace") is not None
    grace_el = _find(note_el, "grace")
    grace_type = None
    if grace_el is not None:
        if grace_el.attrib.get("slash") == "yes":
            grace_type = "acciaccatura"
        else:
            grace_type = "appoggiatura"

    rest_el = _find(note_el, "rest")
    pitch = None
    if rest_el is None:
        pitch_el = _find(note_el, "pitch")
        if pitch_el is not None:
            step = _findtext(pitch_el, "step", "C") or "C"
            alter_s = _findtext(pitch_el, "alter")
            alter = int(float(alter_s)) if alter_s else 0
            octave = int(_findtext(pitch_el, "octave", "4") or "4")
            pitch = Pitch(step, ALTER_TO_ACC.get(alter, ""), octave)

    dur_div = int(_findtext(note_el, "duration", "0") or "0")
    dur = Rational.from_value(Fraction(dur_div, divisions * 4)) if not is_grace else Rational(0)

    dots = len(_findall(note_el, "dot"))
    # Prefer type-based written duration + augment when available
    type_el = _findtext(note_el, "type")
    written = _type_to_dur(type_el) if type_el else None
    augment = 0
    if written is not None and dots:
        augment = dots
        dur = written
    elif written is not None and not is_grace:
        # If duration matches typed value (no tuplet), use written
        tm = _find(note_el, "time-modification")
        if tm is None and dots == 0:
            dur = written
        elif tm is None and dots:
            augment = dots
            dur = written

    tuplet = None
    tm = _find(note_el, "time-modification")
    if tm is not None:
        actual = int(_findtext(tm, "actual-notes", "3") or "3")
        normal = int(_findtext(tm, "normal-notes", "2") or "2")
        tuplet = Ratio(actual, normal)
        if written is not None:
            dur = written
            augment = dots

    staff = int(_findtext(note_el, "staff", "1") or "1")
    voice = _findtext(note_el, "voice", "1") or "1"

    arts: list[str] = []
    finger = None
    notations = _find(note_el, "notations")
    tie_start = tie_stop = False
    slur_starts: list[int] = []
    slur_stops: list[int] = []
    beams: dict[int, str] = {}

    for beam in _findall(note_el, "beam"):
        num = int(beam.attrib.get("number", "1"))
        if beam.text:
            beams[num] = beam.text.strip()

    if notations is not None:
        for tied in _findall(notations, "tied"):
            t = tied.attrib.get("type")
            if t == "start":
                tie_start = True
            elif t == "stop":
                tie_stop = True
        # also <tie> under note
        for articulations in _findall(notations, "articulations"):
            for a in articulations:
                name = ARTICULATION_MAP.get(_local(a.tag))
                if name and name not in arts:
                    arts.append(name)
        for technical in _findall(notations, "technical"):
            fing = _find(technical, "fingering")
            if fing is not None and fing.text and fing.text.strip().isdigit():
                finger = int(fing.text.strip())
        for slur in _findall(notations, "slur"):
            num = int(slur.attrib.get("number", "1"))
            st = slur.attrib.get("type")
            if st == "start":
                slur_starts.append(num)
            elif st == "stop":
                slur_stops.append(num)
        for orn in _findall(notations, "ornaments"):
            for o in orn:
                oname = _local(o.tag)
                if oname in {
                    "trill-mark",
                    "mordent",
                    "inverted-mordent",
                    "turn",
                    "inverted-turn",
                }:
                    mapping = {
                        "trill-mark": "trill",
                        "mordent": "mordent",
                        "inverted-mordent": "inverted-mordent",
                        "turn": "turn",
                        "inverted-turn": "inverted-turn",
                    }
                    arts  # keep separate; store later via props if needed
                    # Store ornament via dynamic attribute hack on event after
                    pass

    for tie in _findall(note_el, "tie"):
        t = tie.attrib.get("type")
        if t == "start":
            tie_start = True
        elif t == "stop":
            tie_stop = True

    nid = id_gen.next("r" if rest_el is not None else "n")
    print_obj = _find(note_el, "print-object")
    # print-object is attribute on note
    visible = note_el.attrib.get("print-object", "yes") != "no"

    return _NoteEvent(
        id=nid,
        pitch=pitch,
        dur=dur if dur.numerator or is_grace else Rational(0),
        augment=augment,
        tuplet=tuplet,
        staff=staff,
        voice=voice,
        is_rest=rest_el is not None,
        is_chord=is_chord,
        is_grace=is_grace,
        grace_type=grace_type,
        finger=finger,
        arts=arts,
        tie_start=tie_start,
        tie_stop=tie_stop,
        slur_starts=slur_starts,
        slur_stops=slur_stops,
        beams=beams,
        visible=visible,
    )


def _type_to_dur(type_name: str | None) -> Rational | None:
    mapping = {
        "whole": Rational(1),
        "half": Rational(1, 2),
        "quarter": Rational(1, 4),
        "eighth": Rational(1, 8),
        "16th": Rational(1, 16),
        "32nd": Rational(1, 32),
        "64th": Rational(1, 64),
        "128th": Rational(1, 128),
        "breve": Rational(2),
        "long": Rational(4),
    }
    return mapping.get(type_name or "")


def _events_to_nodes(
    events: list[_NoteEvent],
    id_gen: "_IdGen",
    open_ties: dict,
    open_slurs: dict,
    beam_active: dict,
    top_relations: list[Node],
    *,
    measure_number: int,
) -> list[Node]:
    nodes: list[Node] = []
    i = 0
    while i < len(events):
        ev = events[i]
        if ev.is_grace:
            # collect grace sequence
            graces = []
            while i < len(events) and events[i].is_grace:
                g = events[i]
                props: dict[str, Any] = {}
                if g.pitch:
                    props["pitch"] = g.pitch
                if g.grace_type:
                    props["type"] = g.grace_type
                graces.append(Node(kind="grace", id=g.id, props=props))
                i += 1
            anchor = events[i].id if i < len(events) else None
            gg_props: dict[str, Any] = {"placement": "before"}
            if anchor:
                gg_props["anchor"] = anchor
            nodes.append(
                Node(
                    kind="grace-group",
                    id=id_gen.next("gg"),
                    props=gg_props,
                    children=graces,
                )
            )
            continue

        # Chord grouping: note + following chord tones
        if not ev.is_rest and i + 1 < len(events) and events[i + 1].is_chord:
            tones = [ev]
            i += 1
            while i < len(events) and events[i].is_chord:
                tones.append(events[i])
                i += 1
            chord_id = id_gen.next("c")
            tone_nodes = []
            for t in tones:
                props = _note_props(t)
                tone_nodes.append(Node(kind="tone", id=t.id, props=props))
                _handle_ties_slurs_beams(
                    t, open_ties, open_slurs, beam_active, top_relations, id_gen
                )
            cprops: dict[str, Any] = {"dur": tones[0].dur}
            if tones[0].augment:
                cprops["augment"] = tones[0].augment
            if tones[0].tuplet:
                cprops["tuplet"] = tones[0].tuplet
            nodes.append(Node(kind="chord", id=chord_id, props=cprops, children=tone_nodes))
            continue

        if ev.is_rest:
            props = {"dur": ev.dur}
            if ev.augment:
                props["augment"] = ev.augment
            if not ev.visible:
                props["visible"] = False
            nodes.append(Node(kind="rest", id=ev.id, props=props))
        else:
            props = _note_props(ev)
            nodes.append(Node(kind="note", id=ev.id, props=props))
            _handle_ties_slurs_beams(
                ev, open_ties, open_slurs, beam_active, top_relations, id_gen
            )
        i += 1
    return nodes


def _note_props(ev: _NoteEvent) -> dict[str, Any]:
    props: dict[str, Any] = {}
    if ev.pitch:
        props["pitch"] = ev.pitch
    props["dur"] = ev.dur
    if ev.augment:
        props["augment"] = ev.augment
    if ev.tuplet:
        props["tuplet"] = ev.tuplet
    if ev.finger:
        props["finger"] = ev.finger
    if ev.arts:
        props["art"] = list(ev.arts)
    if ev.dynamic:
        props["dynamic"] = ev.dynamic
    return props


def _handle_ties_slurs_beams(
    ev: _NoteEvent,
    open_ties: dict,
    open_slurs: dict,
    beam_active: dict,
    top_relations: list[Node],
    id_gen: "_IdGen",
) -> None:
    if ev.pitch:
        key = (str(ev.pitch), ev.staff)
        if ev.tie_stop and key in open_ties:
            top_relations.append(
                Node(
                    kind="tie",
                    id=id_gen.next("tie"),
                    props={"from": open_ties.pop(key), "to": ev.id},
                )
            )
        if ev.tie_start:
            open_ties[key] = ev.id

    for num in ev.slur_stops:
        if num in open_slurs:
            top_relations.append(
                Node(
                    kind="slur",
                    id=id_gen.next("s"),
                    props={"from": open_slurs.pop(num), "to": ev.id},
                )
            )
    for num in ev.slur_starts:
        open_slurs[num] = ev.id

    for level, state in ev.beams.items():
        bkey = (ev.voice, level)
        if state == "begin":
            beam_active[bkey] = [ev.id]
        elif state == "continue":
            beam_active.setdefault(bkey, []).append(ev.id)
        elif state == "end":
            ids = beam_active.pop(bkey, []) + [ev.id]
            # dedupe preserve order
            seen = set()
            ordered = []
            for x in ids:
                if x not in seen:
                    seen.add(x)
                    ordered.append(x)
            top_relations.append(
                Node(
                    kind="beam",
                    id=id_gen.next("b"),
                    props={"level": level, "notes": ordered},
                )
            )


def _attrs_to_nodes(
    attrs: Element,
    measure: int,
    staff_map: dict[int, str],
    id_gen: "_IdGen",
    staves_used: set[int],
) -> list[Node]:
    nodes: list[Node] = []
    time_el = _find(attrs, "time")
    if time_el is not None:
        beats = int(_findtext(time_el, "beats", "4") or "4")
        beat_type = int(_findtext(time_el, "beat-type", "4") or "4")
        nodes.append(
            Node(
                kind="meter",
                props={
                    "at": Position(measure, Rational(0)),
                    "beats": beats,
                    "beat-unit": Rational(1, beat_type),
                },
            )
        )
    key_el = _find(attrs, "key")
    if key_el is not None:
        fifths = int(_findtext(key_el, "fifths", "0") or "0")
        mode = _findtext(key_el, "mode", "major") or "major"
        tonic = _fifths_to_tonic(fifths, mode)
        nodes.append(
            Node(
                kind="key",
                props={
                    "at": Position(measure, Rational(0)),
                    "tonic": tonic,
                    "mode": mode,
                },
            )
        )
    for clef in _findall(attrs, "clef"):
        num = int(clef.attrib.get("number", "1"))
        staves_used.add(num)
        sign = _findtext(clef, "sign", "G") or "G"
        line = _findtext(clef, "line")
        clef_type = _clef_type(sign, line, _findtext(clef, "clef-octave-change"))
        nodes.append(
            Node(
                kind="clef",
                props={
                    "staff": staff_map.get(num, f"S{num}"),
                    "at": Position(measure, Rational(0)),
                    "type": clef_type,
                },
            )
        )
    return nodes


def _direction_to_nodes(
    direction: Element,
    measure: int,
    divisions: int,
    staff_map: dict[int, str],
    id_gen: "_IdGen",
    open_pedals: dict,
    cursor_by_voice: dict,
) -> list[Node]:
    nodes: list[Node] = []
    offset = Rational(0)
    # estimate offset from max cursor
    if cursor_by_voice:
        mx = max(cursor_by_voice.values())
        offset = Rational.from_value(mx)
    staff_el = _findtext(direction, "staff")
    staff_name = staff_map.get(int(staff_el), "RH") if staff_el else None

    for dtype in _findall(direction, "direction-type"):
        for dyn in _findall(dtype, "dynamics"):
            for child in dyn:
                name = _local(child.tag)
                if name in DYNAMIC_NAMES:
                    props: dict[str, Any] = {
                        "value": name,
                        "at": Position(measure, offset),
                    }
                    if staff_name:
                        props["staff"] = staff_name
                    nodes.append(Node(kind="dynamic", id=id_gen.next("d"), props=props))
        for wedge in _findall(dtype, "wedge"):
            wtype = wedge.attrib.get("type")
            if wtype == "crescendo":
                nodes.append(
                    Node(
                        kind="hairpin",
                        id=id_gen.next("h"),
                        props={
                            "type": "crescendo",
                            "from": Position(measure, offset),
                            "to": Position(measure, offset + Rational(1, 4)),
                        },
                    )
                )
            elif wtype == "diminuendo":
                nodes.append(
                    Node(
                        kind="hairpin",
                        id=id_gen.next("h"),
                        props={
                            "type": "diminuendo",
                            "from": Position(measure, offset),
                            "to": Position(measure, offset + Rational(1, 4)),
                        },
                    )
                )
        for pedal in _findall(dtype, "pedal"):
            ptype = "sustain"
            pt = pedal.attrib.get("type", "start")
            if pt == "start":
                pid = id_gen.next("p")
                open_pedals[ptype] = (Position(measure, offset), pid)
            elif pt in ("stop", "change") and ptype in open_pedals:
                frm, pid = open_pedals.pop(ptype)
                nodes.append(
                    Node(
                        kind="pedal",
                        id=pid,
                        props={
                            "type": ptype,
                            "from": frm,
                            "to": Position(measure, offset),
                            "depth": 1,
                            "notation": True,
                        },
                    )
                )
        for metro in _findall(dtype, "metronome"):
            beat_unit = _findtext(metro, "beat-unit", "quarter") or "quarter"
            per_min = _findtext(metro, "per-minute")
            dots = len(_findall(metro, "beat-unit-dot"))
            if per_min:
                props = {
                    "at": Position(measure, offset),
                    "beat": _type_to_dur(beat_unit) or Rational(1, 4),
                    "bpm": int(float(per_min)),
                }
                if dots:
                    props["augment"] = dots
                nodes.append(Node(kind="tempo", id=id_gen.next("t"), props=props))
        # words ignored for now
    # Prefer metronome directions; use <sound tempo> only as fallback.
    if not any(n.kind == "tempo" for n in nodes):
        sound = _find(direction, "sound")
        if sound is not None and "tempo" in sound.attrib:
            nodes.append(
                Node(
                    kind="tempo",
                    id=id_gen.next("t"),
                    props={
                        "at": Position(measure, Rational(0)),
                        "beat": Rational(1, 4),
                        "bpm": int(float(sound.attrib["tempo"])),
                    },
                )
            )
    return nodes


def _harmony_to_node(harmony: Element, measure: int, id_gen: "_IdGen") -> Node | None:
    root_el = _find(harmony, "root")
    if root_el is None:
        return None
    step = _findtext(root_el, "root-step", "C") or "C"
    alter_s = _findtext(root_el, "root-alter")
    alter = int(float(alter_s)) if alter_s else 0
    root = f"{step}{ALTER_TO_ACC.get(alter, '')}"
    kind_el = _find(harmony, "kind")
    quality = "major"
    if kind_el is not None and kind_el.text:
        quality = kind_el.text.strip().replace(" ", "-")
    props: dict[str, Any] = {
        "at": Position(measure, Rational(0)),
        "root": root,
        "quality": quality,
    }
    bass = _find(harmony, "bass")
    if bass is not None:
        bstep = _findtext(bass, "bass-step")
        if bstep:
            balter_s = _findtext(bass, "bass-alter")
            balter = int(float(balter_s)) if balter_s else 0
            props["bass"] = f"{bstep}{ALTER_TO_ACC.get(balter, '')}"
    return Node(kind="chord-symbol", id=id_gen.next("cs"), props=props)


def _fifths_to_tonic(fifths: int, mode: str) -> str:
    # major circle
    majors = ["C", "G", "D", "A", "E", "B", "F#", "C#"]
    majors_flat = ["C", "F", "Bb", "Eb", "Ab", "Db", "Gb", "Cb"]
    if fifths >= 0:
        tonic = majors[min(fifths, 7)]
    else:
        tonic = majors_flat[min(-fifths, 7)]
    if mode in ("minor", "aeolian"):
        # relative minor: down 3 semitones spelling — approximate via circle
        minor_map = {
            "C": "A",
            "G": "E",
            "D": "B",
            "A": "F#",
            "E": "C#",
            "B": "G#",
            "F#": "D#",
            "C#": "A#",
            "F": "D",
            "Bb": "G",
            "Eb": "C",
            "Ab": "F",
            "Db": "Bb",
            "Gb": "Eb",
            "Cb": "Ab",
        }
        return minor_map.get(tonic, tonic)
    return tonic


def _clef_type(sign: str, line: str | None, octave_change: str | None) -> str:
    oc = int(octave_change) if octave_change else 0
    if sign == "G" and (line is None or line == "2"):
        if oc == 1:
            return "treble-8va"
        if oc == -1:
            return "treble-8vb"
        return "treble"
    if sign == "F" and (line is None or line == "4"):
        if oc == -1:
            return "bass-8vb"
        return "bass"
    if sign == "C" and line == "3":
        return "alto"
    if sign == "C" and line == "4":
        return "tenor"
    if sign == "percussion":
        return "percussion"
    return "treble"


def _timewise_to_partwise(root: Element) -> Element:
    """Convert score-timewise to a synthetic score-partwise element."""
    new_root = ET.Element("score-partwise", version=root.attrib.get("version", "3.1"))
    for child in root:
        if _local(child.tag) != "measure":
            new_root.append(child)
    parts: dict[str, Element] = {}
    for measure in _findall(root, "measure"):
        mnum = measure.attrib.get("number", "1")
        for part in _findall(measure, "part"):
            pid = part.attrib.get("id", "P1")
            if pid not in parts:
                parts[pid] = ET.SubElement(new_root, "part", id=pid)
            m = ET.SubElement(parts[pid], "measure", number=mnum)
            for c in list(part):
                m.append(c)
    return new_root


def _slug(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalnum() or ch in "_-":
            out.append(ch)
        elif ch in (" ", "/"):
            out.append("_")
    s = "".join(out).strip("_-")
    if not s or s[0].isdigit():
        s = "p" + s
    return s or "part"


class _IdGen:
    def __init__(self) -> None:
        self.counts: dict[str, int] = defaultdict(int)

    def next(self, prefix: str) -> str:
        self.counts[prefix] += 1
        return f"{prefix}{self.counts[prefix]}"
