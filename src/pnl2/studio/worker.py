"""Parse, validate, and engrave PNL/2 text for the studio preview."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RenderResult:
    ok: bool
    logs: list[str] = field(default_factory=list)
    svg: str | None = None


def render_source(text: str) -> RenderResult:
    """Parse, validate, and engrave *text*. Always returns log lines."""
    logs: list[str] = []
    from ..engraver import EngraverError, combine_svgs, engrave_svg
    from ..lexer import LexerError
    from ..parser import ParseError, parse
    from ..validator import validate

    try:
        doc = parse(text)
        logs.append("OK parse")
    except (ParseError, LexerError) as exc:
        logs.append(f"parse error: {exc}")
        return RenderResult(ok=False, logs=logs)

    errors = validate(doc)
    if errors:
        for err in errors:
            logs.append(f"validate: {err}")
    else:
        logs.append("OK validate")

    try:
        pages = engrave_svg(doc)
        n = len(pages)
        logs.append(f"engraved {n} page" if n == 1 else f"engraved {n} pages")
        return RenderResult(ok=True, logs=logs, svg=combine_svgs(pages))
    except EngraverError as exc:
        logs.append(f"engrave error: {exc}")
        return RenderResult(ok=False, logs=logs)
    except Exception as exc:  # noqa: BLE001
        logs.append(f"engrave error: {exc}")
        return RenderResult(ok=False, logs=logs)
