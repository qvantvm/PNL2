"""Deterministic synthetic PNL/2 corpus generator."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

from pnl2vec.pnl import parse_pnl, serialize_pnl, validate_pnl

logger = logging.getLogger(__name__)

LETTER_STEPS = ["C", "D", "E", "F", "G", "A", "B"]

MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]

# MIDI PC -> preferred spelling
PC_SPELL = {
    0: ("C", ""),
    1: ("C", "#"),
    2: ("D", ""),
    3: ("E", "b"),
    4: ("E", ""),
    5: ("F", ""),
    6: ("F", "#"),
    7: ("G", ""),
    8: ("A", "b"),
    9: ("A", ""),
    10: ("B", "b"),
    11: ("B", ""),
}

ROOTS = ["C", "D", "E", "F", "G", "A", "B", "F#", "Bb", "Eb"]
MODES = ["major", "minor"]
METERS = [(4, "1/4"), (3, "1/4"), (6, "1/8"), (2, "1/4")]
DURS = ["1/4", "1/8", "1/2", "1/16"]
ARTS = ["staccato", "accent", "tenuto", "legato"]
DYNS = ["pp", "p", "mp", "mf", "f", "ff"]

SIZE_COUNTS = {
    "tiny": 100,
    "small": 1000,
    "medium": 10000,
}


@dataclass
class GeneratorConfig:
    size: str = "tiny"
    seed: int = 42
    output_dir: Path = Path("data/raw")


def _pc_to_pitch(root_pc: int, degree_semitone: int, octave: int) -> str:
    pc = (root_pc + degree_semitone) % 12
    letter, acc = PC_SPELL[pc]
    # octave adjustment when wrapping
    base_oct = octave
    if root_pc + degree_semitone >= 12:
        base_oct = octave + 1
    return f"{letter}{acc}{base_oct}"


def _root_pc(root: str) -> int:
    mapping = {
        "C": 0,
        "C#": 1,
        "Db": 1,
        "D": 2,
        "D#": 3,
        "Eb": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "Gb": 6,
        "G": 7,
        "G#": 8,
        "Ab": 8,
        "A": 9,
        "A#": 10,
        "Bb": 10,
        "B": 11,
    }
    return mapping[root]


def _doc_header(title: str, family: str, extra_meta: str = "") -> str:
    return (
        "pnl/2\n"
        "score {\n"
        "    meta {\n"
        f'        title="{title}"\n'
        "        synthetic=true\n"
        f'        family="{family}"\n'
        f"{extra_meta}"
        "        profile=[core,notation]\n"
        "    }\n"
    )


def _part_open(tonic: str, mode: str, beats: int, beat_unit: str, bpm: int) -> str:
    return (
        "    part piano instrument=piano staves=2 {\n"
        f"        meter at=1:0 beats={beats} beat-unit={beat_unit}\n"
        f"        key at=1:0 tonic={tonic} mode={mode}\n"
        f"        tempo t1 at=1:0 beat=1/4 bpm={bpm}\n"
    )


def generate_scale_doc(rng: random.Random, doc_idx: int) -> str:
    root = rng.choice(ROOTS)
    mode = rng.choice(MODES)
    scale = MAJOR_SCALE if mode == "major" else MINOR_SCALE
    octave = rng.choice([3, 4, 5])
    # Sometimes put scale in LH to avoid trivial hand/register correlation
    scale_in_lh = rng.random() < 0.35
    beats, unit = rng.choice(METERS)
    bpm = rng.randint(60, 140)
    dur = rng.choice(["1/4", "1/8"])
    rpc = _root_pc(root)
    notes = [_pc_to_pitch(rpc, s, octave) for s in scale]
    notes.append(_pc_to_pitch(rpc, 12, octave))  # octave

    lines = [
        _doc_header(f"Scale {doc_idx}", "scale"),
        _part_open(root.replace("b", "b").replace("#", "#"), mode, beats, unit, bpm),
        "        measure 1 {\n",
    ]
    if scale_in_lh:
        lines.append("            staff RH {\n                voice RH1 {\n                    rest r1 dur=1\n                }\n            }\n")
        lines.append("            staff LH {\n                voice LH1 {\n")
        for i, p in enumerate(notes[:4]):
            finger = (i % 5) + 1
            art = f" art=[{rng.choice(ARTS)}]" if rng.random() < 0.2 else ""
            lines.append(f"                    note n{i+1} pitch={p} dur={dur} finger={finger}{art}\n")
        lines.append("                }\n            }\n")
    else:
        lines.append("            staff RH {\n                voice RH1 {\n")
        for i, p in enumerate(notes[:4]):
            finger = (i % 5) + 1
            art = f" art=[{rng.choice(ARTS)}]" if rng.random() < 0.2 else ""
            dyn = f" dynamic={rng.choice(DYNS)}" if rng.random() < 0.15 else ""
            lines.append(f"                    note n{i+1} pitch={p} dur={dur} finger={finger}{art}{dyn}\n")
        lines.append("                }\n            }\n")
        lines.append("            staff LH {\n                voice LH1 {\n                    rest r1 dur=1\n                }\n            }\n")
    lines.append("        }\n")
    # measure 2 continuation
    lines.append("        measure 2 {\n")
    hand_staff = "LH" if scale_in_lh else "RH"
    other = "RH" if scale_in_lh else "LH"
    lines.append(f"            staff {hand_staff} {{\n                voice {hand_staff}1 {{\n")
    for i, p in enumerate(notes[4:]):
        idx = i + 5
        lines.append(f"                    note n{idx} pitch={p} dur={dur} finger={(i % 5)+1}\n")
    # pad to fill if needed
    if len(notes[4:]) < 4:
        for j in range(4 - len(notes[4:])):
            lines.append(f"                    rest r{j+2} dur={dur}\n")
    lines.append("                }\n            }\n")
    lines.append(f"            staff {other} {{\n                voice {other}1 {{\n                    rest rx dur=1\n                }}\n            }}\n")
    lines.append("        }\n    }\n}\n")
    return "".join(lines)


def generate_chord_doc(rng: random.Random, doc_idx: int) -> str:
    root = rng.choice(ROOTS)
    mode = rng.choice(MODES)
    quality = rng.choice(["triad", "seventh"])
    inversion = rng.randint(0, 2)
    rpc = _root_pc(root)
    if mode == "major":
        intervals = [0, 4, 7] if quality == "triad" else [0, 4, 7, 10]
    else:
        intervals = [0, 3, 7] if quality == "triad" else [0, 3, 7, 10]
    # rotate for inversion
    pitches = []
    for i, iv in enumerate(intervals):
        octv = 3 + (1 if i >= len(intervals) - inversion and inversion else 0)
        if i < inversion:
            octv = 4
        pitches.append(_pc_to_pitch(rpc, iv + (12 if i < inversion else 0), 3 if i >= inversion else 4))
    # simplify: build root position then shift
    pitches = [_pc_to_pitch(rpc, iv, 3) for iv in intervals]
    for _ in range(inversion):
        p0 = pitches.pop(0)
        # bump octave roughly
        letter = p0[0]
        rest = p0[1:]
        # parse octave at end
        oct_digit = int(rest[-1])
        acc = rest[:-1]
        pitches.append(f"{letter}{acc}{oct_digit + 1}")

    use_pedal = rng.random() < 0.4  # not always
    arpeggio = rng.random() < 0.45
    beats, unit = 4, "1/4"
    bpm = rng.randint(72, 120)
    lines = [
        _doc_header(f"Chord {doc_idx}", "chordal", f'        quality="{quality}"\n        inversion={inversion}\n'),
        _part_open(root, mode, beats, unit, bpm),
        "        measure 1 {\n",
    ]
    # Occasionally put chords in RH
    chord_hand = "RH" if rng.random() < 0.3 else "LH"
    other = "LH" if chord_hand == "RH" else "RH"
    if arpeggio:
        lines.append(f"            staff {chord_hand} {{\n                voice {chord_hand}1 {{\n")
        for i, p in enumerate(pitches):
            lines.append(f"                    note n{i+1} pitch={p} dur=1/8 finger={(5-i) if chord_hand=='LH' else (i%5)+1}\n")
        for j in range(max(0, 4 - len(pitches))):
            lines.append(f"                    rest ra{j} dur=1/8\n")
        lines.append("                }\n            }\n")
    else:
        lines.append(f"            staff {chord_hand} {{\n                voice {chord_hand}1 {{\n")
        lines.append("                    chord c1 dur=1/2 {\n")
        for i, p in enumerate(pitches[:4]):
            lines.append(f"                        tone n{i+1} pitch={p} finger={(i % 5)+1}\n")
        lines.append("                    }\n")
        lines.append("                    rest rb dur=1/2\n")
        lines.append("                }\n            }\n")
    lines.append(f"            staff {other} {{\n                voice {other}1 {{\n                    rest ro dur=1\n                }}\n            }}\n")
    lines.append("        }\n")
    if use_pedal:
        lines.append("        pedal p1 type=sustain from=1:0 to=1:1 depth=1\n")
    lines.append("    }\n}\n")
    return "".join(lines)


def generate_alberti_doc(rng: random.Random, doc_idx: int) -> str:
    root = rng.choice(["C", "G", "F", "D", "A"])
    rpc = _root_pc(root)
    low = _pc_to_pitch(rpc, 0, 3)
    mid = _pc_to_pitch(rpc, 4, 3)
    high = _pc_to_pitch(rpc, 7, 3)
    melody = [_pc_to_pitch(rpc, s, 5) for s in [0, 2, 4, 5]]
    lines = [
        _doc_header(f"Alberti {doc_idx}", "alberti"),
        _part_open(root, "major", 4, "1/4", 108),
        "        measure 1 {\n",
        "            staff RH {\n                voice RH1 {\n",
    ]
    for i, p in enumerate(melody):
        lines.append(f"                    note m{i+1} pitch={p} dur=1/4 finger={i+1}\n")
    lines.append("                }\n            }\n")
    lines.append("            staff LH {\n                voice LH1 {\n")
    pattern = [low, high, mid, high]
    for i, p in enumerate(pattern):
        lines.append(f"                    note a{i+1} pitch={p} dur=1/8 finger={5 if i==0 else (1 if i%2 else 3)}\n")
    for i, p in enumerate(pattern):
        lines.append(f"                    note b{i+1} pitch={p} dur=1/8 finger={5 if i==0 else (1 if i%2 else 3)}\n")
    lines.append("                }\n            }\n        }\n")
    if rng.random() < 0.5:
        lines.append("        slur s1 from=m1 to=m4\n")
    lines.append("    }\n}\n")
    return "".join(lines)


def generate_cadence_doc(rng: random.Random, doc_idx: int) -> str:
    root = rng.choice(ROOTS[:7])
    rpc = _root_pc(root)
    # I - V - I or I - IV - V - I; sometimes V not resolving (negative)
    resolve = rng.random() < 0.7
    chords = [
        [0, 4, 7],
        [7, 11, 14],  # V
        [0, 4, 7],
    ]
    if not resolve:
        chords = [[0, 4, 7], [7, 11, 14], [2, 5, 9]]  # ends on ii-ish
    lines = [
        _doc_header(f"Cadence {doc_idx}", "cadence", f"        resolves={str(resolve).lower()}\n"),
        _part_open(root, "major", 4, "1/4", 96),
    ]
    for mi, chord in enumerate(chords, start=1):
        lines.append(f"        measure {mi} {{\n")
        lines.append("            staff RH {\n                voice RH1 {\n")
        top = _pc_to_pitch(rpc, chord[-1] % 12 + (12 if chord[-1] >= 12 else 0), 5)
        lines.append(f"                    note rh{mi} pitch={top} dur=1 finger=3\n")
        lines.append("                }\n            }\n")
        lines.append("            staff LH {\n                voice LH1 {\n")
        lines.append(f"                    chord c{mi} dur=1 {{\n")
        for ti, iv in enumerate(chord):
            p = _pc_to_pitch(rpc, iv % 12, 3 + iv // 12)
            lines.append(f"                        tone t{mi}_{ti} pitch={p} finger={5-ti}\n")
        lines.append("                    }\n                }\n            }\n        }\n")
    if rng.random() < 0.3:
        lines.append("        pedal p1 type=sustain from=1:0 to=3:1 depth=1\n")
    lines.append("    }\n}\n")
    return "".join(lines)


def generate_counterpoint_doc(rng: random.Random, doc_idx: int) -> str:
    root = rng.choice(["C", "D", "G"])
    rpc = _root_pc(root)
    upper = [_pc_to_pitch(rpc, s, 5) for s in [0, 2, 4, 5]]
    lower = [_pc_to_pitch(rpc, s, 3) for s in [0, 4, 7, 0]]
    lines = [
        _doc_header(f"Counterpoint {doc_idx}", "contrapuntal"),
        _part_open(root, "major", 4, "1/4", 88),
        "        measure 1 {\n",
        "            staff RH {\n                voice soprano {\n",
    ]
    for i, p in enumerate(upper):
        lines.append(f"                    note u{i+1} pitch={p} dur=1/4 finger={i+1}\n")
    lines.append("                }\n            }\n")
    lines.append("            staff LH {\n                voice bass {\n")
    for i, p in enumerate(lower):
        lines.append(f"                    note l{i+1} pitch={p} dur=1/4 finger={5-i}\n")
    lines.append("                }\n            }\n        }\n")
    lines.append("        slur s1 from=u1 to=u4\n")
    lines.append("    }\n}\n")
    return "".join(lines)


def generate_rhythm_doc(rng: random.Random, doc_idx: int) -> str:
    root = "C"
    pitches = ["C4", "E4", "G4", "C5"]
    durs = rng.choice([["1/8", "1/8", "1/4", "1/2"], ["1/16", "1/16", "1/8", "1/4"], ["1/4", "1/4", "1/4", "1/4"]])
    lines = [
        _doc_header(f"Rhythm {doc_idx}", "rhythmic"),
        _part_open(root, "major", 4, "1/4", 120),
        "        measure 1 {\n            staff RH {\n                voice RH1 {\n",
    ]
    for i, (p, d) in enumerate(zip(pitches, durs)):
        aug = " augment=1" if d == "1/4" and rng.random() < 0.25 else ""
        # if augment, shorten list handling — keep simple
        lines.append(f"                    note n{i+1} pitch={p} dur={d}{aug} finger={i+1}\n")
    lines.append("                }\n            }\n")
    lines.append("            staff LH {\n                voice LH1 {\n                    rest r1 dur=1\n                }\n            }\n")
    lines.append("        }\n    }\n}\n")
    return "".join(lines)


def generate_tie_slur_doc(rng: random.Random, doc_idx: int) -> str:
    lines = [
        _doc_header(f"TieSlur {doc_idx}", "phrase"),
        _part_open("G", "major", 4, "1/4", 100),
        "        measure 1 {\n",
        "            staff RH {\n                voice RH1 {\n",
        "                    note n1 pitch=G4 dur=1/2 finger=1\n",
        "                    note n2 pitch=A4 dur=1/2 finger=2\n",
        "                }\n            }\n",
        "            staff LH {\n                voice LH1 {\n",
        "                    note n3 pitch=G3 dur=1 finger=5\n",
        "                }\n            }\n",
        "        }\n",
        "        measure 2 {\n",
        "            staff RH {\n                voice RH1 {\n",
        "                    note n4 pitch=A4 dur=1/2 finger=2\n",
        "                    note n5 pitch=B4 dur=1/2 finger=3 art=[staccato]\n",
        "                }\n            }\n",
        "            staff LH {\n                voice LH1 {\n",
        "                    rest r1 dur=1\n",
        "                }\n            }\n",
        "        }\n",
        "        tie tie1 from=n2 to=n4\n",
        "        slur s1 from=n1 to=n5\n",
        "    }\n}\n",
    ]
    return "".join(lines)


GENERATORS = [
    generate_scale_doc,
    generate_chord_doc,
    generate_alberti_doc,
    generate_cadence_doc,
    generate_counterpoint_doc,
    generate_rhythm_doc,
    generate_tie_slur_doc,
]


def generate_document(rng: random.Random, doc_idx: int) -> str:
    gen = GENERATORS[doc_idx % len(GENERATORS)]
    # Mix in transposition of motifs via different roots already in gens
    return gen(rng, doc_idx)


def generate_corpus(
    size: str = "tiny",
    *,
    seed: int = 42,
    output_dir: Path | str = "data/raw",
    force: bool = False,
) -> list[Path]:
    if size not in SIZE_COUNTS:
        raise ValueError(f"unknown size {size!r}; choose from {list(SIZE_COUNTS)}")
    n = SIZE_COUNTS[size]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    written: list[Path] = []
    failures = 0
    for i in range(n):
        path = output_dir / f"synth_{size}_{i:05d}.pnl"
        if path.exists() and not force:
            written.append(path)
            continue
        text = None
        for attempt in range(5):
            candidate = generate_document(rng, i + attempt * 9973)
            try:
                doc = parse_pnl(candidate, filename=path.name)
                issues = validate_pnl(doc)
                if issues:
                    # still write if only warnings; reject hard errors
                    logger.debug("validation issues for %s: %s", path.name, issues)
                # Prefer canonical serialize
                text = serialize_pnl(doc)
                # re-parse serialized
                parse_pnl(text)
                break
            except Exception as exc:
                logger.debug("regen %s attempt %s: %s", i, attempt, exc)
                text = None
        if text is None:
            failures += 1
            # last-resort minimal valid doc
            text = generate_tie_slur_doc(rng, i)
            doc = parse_pnl(text)
            text = serialize_pnl(doc)
        path.write_text(text, encoding="utf-8")
        written.append(path)
    if failures:
        logger.warning("synthetic generation had %d fallback documents", failures)
    return written
