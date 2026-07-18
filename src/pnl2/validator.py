"""Semantic validation for PNL/2 documents."""

from __future__ import annotations

from typing import Any, Iterable

from .ast import Document, Node, Pitch, Position, Ratio
from .rational import Rational, effective_duration


class ValidationError(Exception):
    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or [message]


def validate(doc: Document) -> list[str]:
    """Validate a document. Returns a list of error messages (empty if ok)."""
    errors: list[str] = []
    if doc.version != "pnl/2":
        errors.append(f"unsupported version {doc.version!r}")

    ids: dict[str, Node] = {}
    notes: dict[str, Node] = {}

    for part in doc.score.parts:
        _walk(part, ids, notes, errors)
    for stmt in doc.score.statements:
        _walk(stmt, ids, notes, errors)

    # Second pass: references and relation rules
    for node in ids.values():
        _check_node_rules(node, ids, notes, errors)

    for part in doc.score.parts:
        _check_measure_lengths(part, errors)

    return errors


def validate_or_raise(doc: Document) -> None:
    errors = validate(doc)
    if errors:
        raise ValidationError(f"{len(errors)} validation error(s)", errors)


# Structural labels (staff/voice/part) may repeat across measures; uniqueness
# applies to referential event and relation identifiers.
_STRUCTURAL_KINDS = frozenset({"score", "meta", "part", "measure", "staff", "voice", "notation", "performance"})


def _walk(
    node: Node,
    ids: dict[str, Node],
    notes: dict[str, Node],
    errors: list[str],
) -> None:
    if node.id is not None and node.kind not in _STRUCTURAL_KINDS:
        if node.id in ids:
            errors.append(f"duplicate id {node.id!r}")
        else:
            ids[node.id] = node
        if node.kind in ("note", "tone", "grace"):
            notes[node.id] = node

    if node.kind == "chord":
        tones = [c for c in node.children if c.kind == "tone"]
        if len(tones) < 2:
            errors.append(f"chord {node.id!r} must contain at least two tones")
        if "finger" in node.props:
            errors.append(f"chord {node.id!r} must not have chord-level finger")

    if node.kind in ("note", "tone", "rest", "grace"):
        _check_event_timing(node, errors)

    if node.kind == "note":
        if "pitch" not in node.props:
            errors.append(f"note {node.id!r} missing pitch")
        if "dur" not in node.props:
            errors.append(f"note {node.id!r} missing dur")

    _check_finger(node, errors)
    _check_rationals(node, errors)

    for child in node.children:
        _walk(child, ids, notes, errors)


def _check_event_timing(node: Node, errors: list[str]) -> None:
    dur = node.props.get("dur")
    if dur is not None:
        try:
            r = Rational.from_value(dur)
            if r.numerator <= 0:
                errors.append(f"{node.kind} {node.id!r} duration must be positive")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{node.kind} {node.id!r} invalid dur: {exc}")

    at = node.props.get("at")
    if isinstance(at, Position):
        if at.offset.numerator < 0:
            errors.append(f"{node.kind} {node.id!r} offset must be non-negative")
    elif isinstance(at, Rational) and at.numerator < 0:
        errors.append(f"{node.kind} {node.id!r} at must be non-negative")

    if "finger" in node.props and "finger-change" in node.props:
        errors.append(f"{node.kind} {node.id!r} cannot have both finger and finger-change")


def _check_finger(node: Node, errors: list[str]) -> None:
    finger = node.props.get("finger")
    if finger is not None:
        if not isinstance(finger, int) or not 1 <= finger <= 5:
            errors.append(f"{node.kind} {node.id!r} finger must be integer 1..5")
    fc = node.props.get("finger-change")
    if isinstance(fc, Ratio):
        if not (1 <= fc.left <= 5 and 1 <= fc.right <= 5):
            errors.append(f"{node.kind} {node.id!r} finger-change values must be 1..5")


def _check_rationals(node: Node, errors: list[str]) -> None:
    for key, value in node.props.items():
        if isinstance(value, Rational) and value.denominator == 0:
            errors.append(f"{node.kind} {node.id!r} {key} has zero denominator")


def _check_node_rules(
    node: Node,
    ids: dict[str, Node],
    notes: dict[str, Node],
    errors: list[str],
) -> None:
    if node.kind == "tie":
        _check_tie(node, ids, notes, errors)
    elif node.kind == "slur":
        _check_endpoints(node, ids, errors, allow_chord=True)
    elif node.kind == "hairpin":
        has_abs = "from" in node.props or "to" in node.props
        has_ev = "from-event" in node.props or "to-event" in node.props
        if has_abs and has_ev:
            errors.append(f"hairpin {node.id!r} must not mix from/to and from-event/to-event")
        for key in ("from-event", "to-event", "from", "to"):
            ref = node.props.get(key)
            if isinstance(ref, str) and key.endswith("event") and ref not in ids:
                errors.append(f"hairpin {node.id!r} unknown reference {ref!r}")
    elif node.kind == "pedal":
        depth = node.props.get("depth")
        if depth is not None:
            try:
                d = float(Rational.from_value(depth)) if not isinstance(depth, (int, float)) else float(depth)
                if d < 0 or d > 1:
                    errors.append(f"pedal {node.id!r} depth must be in [0,1]")
            except Exception:  # noqa: BLE001
                errors.append(f"pedal {node.id!r} invalid depth")
        frm, to = node.props.get("from"), node.props.get("to")
        if isinstance(frm, Position) and isinstance(to, Position):
            if (to.measure, float(to.offset)) <= (frm.measure, float(frm.offset)):
                errors.append(f"pedal {node.id!r} requires to > from")
    elif node.kind == "roman":
        degree = node.props.get("degree")
        if degree is not None and (not isinstance(degree, int) or not 1 <= degree <= 7):
            errors.append(f"roman {node.id!r} degree must be 1..7")
        if "at" in node.props and ("from" in node.props or "to" in node.props):
            errors.append(f"roman {node.id!r} must not use both at and from/to")
        target = node.props.get("target-degree")
        if target is not None and (not isinstance(target, int) or not 1 <= target <= 7):
            errors.append(f"roman {node.id!r} target-degree must be 1..7")
    elif node.kind in ("phrase", "ottava", "trill-span"):
        _check_endpoints(node, ids, errors, allow_chord=True)


def _check_endpoints(
    node: Node,
    ids: dict[str, Node],
    errors: list[str],
    *,
    allow_chord: bool,
) -> None:
    for key in ("from", "to"):
        ref = node.props.get(key)
        if not isinstance(ref, str):
            continue
        if ref not in ids:
            errors.append(f"{node.kind} {node.id!r} unknown reference {ref!r}")
            continue
        target = ids[ref]
        ok = target.kind in ("note", "tone", "grace") or (allow_chord and target.kind == "chord")
        if not ok:
            errors.append(
                f"{node.kind} {node.id!r} endpoint {ref!r} must be note/tone"
                + ("/chord" if allow_chord else "")
            )


def _check_tie(
    node: Node,
    ids: dict[str, Node],
    notes: dict[str, Node],
    errors: list[str],
) -> None:
    frm = node.props.get("from")
    to = node.props.get("to")
    for ref, label in ((frm, "from"), (to, "to")):
        if not isinstance(ref, str) or ref not in notes:
            errors.append(f"tie {node.id!r} {label} must reference a note or tone")
            return
    a, b = notes[frm], notes[to]
    pa, pb = a.props.get("pitch"), b.props.get("pitch")
    if isinstance(pa, Pitch) and isinstance(pb, Pitch) and not pa.enharmonic_equal(pb):
        errors.append(f"tie {node.id!r} pitches are not sounding-equivalent")


def _check_measure_lengths(part: Node, errors: list[str]) -> None:
    """Warn/error when voice content exceeds meter unless open=true."""
    # Collect meters by measure start; simple default 4/4
    default_len = Rational(1)
    for measure in _iter_kind(part, "measure"):
        if measure.props.get("open") is True:
            continue
        measure_len = default_len
        # Find meter statements earlier or inside — best-effort: use last meter in part
        for stmt in _iter_deep(part):
            if stmt.kind == "meter":
                beats = stmt.props.get("beats")
                unit = stmt.props.get("beat-unit")
                if isinstance(beats, int) and unit is not None:
                    measure_len = Rational.from_value(unit) * beats
        for staff in measure.children:
            if staff.kind != "staff":
                continue
            for voice in staff.children:
                if voice.kind != "voice":
                    continue
                cursor = Rational(0)
                for ev in voice.children:
                    if ev.kind in ("note", "rest", "chord", "space"):
                        if "at" in ev.props:
                            at = ev.props["at"]
                            if isinstance(at, Rational):
                                cursor = at
                            elif isinstance(at, Position):
                                cursor = at.offset
                        dur = ev.props.get("dur", Rational(0))
                        augment = int(ev.props.get("augment", 0) or 0)
                        tuplet = ev.props.get("tuplet")
                        tpair = (tuplet.left, tuplet.right) if isinstance(tuplet, Ratio) else None
                        try:
                            eff = effective_duration(dur, augment, tpair)
                        except Exception:  # noqa: BLE001
                            continue
                        end = cursor + eff
                        if end > measure_len:
                            errors.append(
                                f"measure {measure.number} voice {voice.id!r} exceeds measure length"
                            )
                            break
                        if "at" not in ev.props:
                            cursor = end


def _iter_kind(node: Node, kind: str) -> Iterable[Node]:
    if node.kind == kind:
        yield node
    for child in node.children:
        yield from _iter_kind(child, kind)


def _iter_deep(node: Node) -> Iterable[Node]:
    yield node
    for child in node.children:
        yield from _iter_deep(child)
