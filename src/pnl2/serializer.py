"""Canonical serializer for PNL/2 documents."""

from __future__ import annotations

from typing import Any

from .ast import Document, Node, Pitch, Position, Ratio, Score
from .rational import Rational

NOTE_PROP_ORDER = [
    "pitch",
    "dur",
    "augment",
    "tuplet",
    "at",
    "staff",
    "hand",
    "voice",
    "finger",
    "finger-change",
    "art",
    "ornament",
    "ornament-start",
    "ornament-interval",
    "ornament-rate",
    "dynamic",
    "role",
    "velocity",
    "cents",
    "head",
    "fermata",
    "accidental-display",
    "visible",
    "type",
    "arpeggiate",
    "perf-offset",
    "steal-from",
    "steal-ratio",
    "placement",
    "anchor",
    "allow-gap",
]

ARTICULATION_ORDER = [
    "staccato",
    "staccatissimo",
    "tenuto",
    "accent",
    "marcato",
    "portato",
    "soft-accent",
    "detached-legato",
]


def serialize(doc: Document, *, indent: str = "    ") -> str:
    lines = [doc.version]
    lines.append("score {")
    body = _serialize_score(doc.score, indent)
    lines.extend(_indent_lines(body, indent))
    lines.append("}")
    return "\n".join(lines) + "\n"


def _serialize_score(score: Score, indent: str) -> list[str]:
    lines: list[str] = []
    if score.meta:
        lines.append("meta {")
        for key in sorted(score.meta.keys()) if False else list(score.meta.keys()):
            lines.append(f"{indent}{key}={_format_value(score.meta[key])}")
        lines.append("}")
    for part in score.parts:
        lines.extend(_serialize_node(part, indent, 0))
    for stmt in score.statements:
        lines.extend(_serialize_node(stmt, indent, 0))
    return lines


def _serialize_node(node: Node, indent: str, depth: int) -> list[str]:
    ind = indent * depth
    if node.kind == "property":
        return [f"{ind}{node.id}={_format_value(node.props.get('value'))}"]

    head = node.kind
    if node.kind == "measure" and node.number is not None:
        head = f"measure {node.number}"
    elif node.id is not None:
        head = f"{node.kind} {node.id}"

    props = _ordered_props(node)
    prop_str = "".join(f" {k}={_format_value(v)}" for k, v in props)

    block_kinds = {
        "part",
        "measure",
        "staff",
        "voice",
        "chord",
        "grace-group",
        "pedal-curve",
        "notation",
        "performance",
        "meta",
        "score",
    }
    if node.kind in block_kinds or node.children:
        lines = [f"{ind}{head}{prop_str} {{"]
        for child in node.children:
            lines.extend(_serialize_node(child, indent, depth + 1))
        lines.append(f"{ind}}}")
        return lines

    return [f"{ind}{head}{prop_str}"]


def _ordered_props(node: Node) -> list[tuple[str, Any]]:
    items = list(node.props.items())
    if node.kind in ("note", "tone", "grace", "rest"):
        order = {k: i for i, k in enumerate(NOTE_PROP_ORDER)}
        items.sort(key=lambda kv: order.get(kv[0], 1000 + hash(kv[0]) % 1000))
    return items


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Pitch):
        return str(value)
    if isinstance(value, Position):
        return str(value)
    if isinstance(value, Ratio):
        return str(value)
    if isinstance(value, Rational):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        if value and all(isinstance(x, str) for x in value):
            # articulation canonical order when applicable
            if all(x in ARTICULATION_ORDER for x in value):
                value = sorted(value, key=lambda a: ARTICULATION_ORDER.index(a))
        inner = ",".join(_format_value(v) for v in value)
        return f"[{inner}]"
    if isinstance(value, str):
        if _needs_quotes(value):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        return value
    return str(value)


def _needs_quotes(value: str) -> bool:
    if not value:
        return True
    if any(c in value for c in ' \t\n"\\=[]{}'):
        return True
    if value[0].isdigit():
        return True
    return False


def _indent_lines(lines: list[str], indent: str) -> list[str]:
    return [indent + line if line else line for line in lines]
