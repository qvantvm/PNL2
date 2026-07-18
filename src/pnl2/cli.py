"""Command-line interface for PNL/2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .musicxml.from_musicxml import musicxml_to_pnl
from .musicxml.to_musicxml import pnl_to_musicxml
from .parser import ParseError, parse
from .serializer import serialize
from .validator import validate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pnl2",
        description="PNL/2 — Piano Notation Language tools",
    )
    parser.add_argument("--version", action="version", version=f"pnl2 {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_parse = sub.add_parser("parse", help="Parse a PNL/2 file and optionally re-serialize")
    p_parse.add_argument("input", type=Path)
    p_parse.add_argument("-o", "--output", type=Path)
    p_parse.add_argument("--canonical", action="store_true", help="Emit canonical form")

    p_val = sub.add_parser("validate", help="Validate a PNL/2 file")
    p_val.add_argument("input", type=Path)

    p_from = sub.add_parser("from-musicxml", help="Convert MusicXML to PNL/2")
    p_from.add_argument("input", type=Path)
    p_from.add_argument("-o", "--output", type=Path)
    p_from.add_argument(
        "--staff-map",
        type=str,
        default="1=RH,2=LH",
        help="Staff number mapping, e.g. 1=RH,2=LH",
    )

    p_to = sub.add_parser("to-musicxml", help="Convert PNL/2 to MusicXML")
    p_to.add_argument("input", type=Path)
    p_to.add_argument("-o", "--output", type=Path)

    args = parser.parse_args(argv)

    try:
        if args.command == "parse":
            text = args.input.read_text(encoding="utf-8")
            doc = parse(text)
            out = serialize(doc) if args.canonical else text
            if args.output:
                args.output.write_text(out, encoding="utf-8")
            else:
                sys.stdout.write(out if args.canonical else f"OK: parsed {args.input}\n")
            return 0

        if args.command == "validate":
            text = args.input.read_text(encoding="utf-8")
            doc = parse(text)
            errors = validate(doc)
            if errors:
                for err in errors:
                    print(f"error: {err}", file=sys.stderr)
                return 1
            print(f"OK: {args.input}")
            return 0

        if args.command == "from-musicxml":
            staff_map = _parse_staff_map(args.staff_map)
            xml_text = args.input.read_text(encoding="utf-8")
            pnl = musicxml_to_pnl(xml_text, staff_map=staff_map)
            if args.output:
                args.output.write_text(pnl, encoding="utf-8")
            else:
                sys.stdout.write(pnl)
            return 0

        if args.command == "to-musicxml":
            text = args.input.read_text(encoding="utf-8")
            xml = pnl_to_musicxml(text)
            if args.output:
                args.output.write_text(xml, encoding="utf-8")
            else:
                sys.stdout.write(xml)
            return 0
    except ParseError as exc:
        print(f"parse error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 1


def _parse_staff_map(text: str) -> dict[int, str]:
    result: dict[int, str] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        num_s, name = part.split("=", 1)
        result[int(num_s)] = name.strip()
    return result or {1: "RH", 2: "LH"}


if __name__ == "__main__":
    raise SystemExit(main())
