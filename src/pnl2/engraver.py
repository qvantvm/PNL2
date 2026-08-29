"""Engrave PNL/2 scores to SVG, PNG, or HTML via MusicXML and Verovio."""

from __future__ import annotations

import html
import re
import threading
from pathlib import Path
from typing import Any

from .ast import Document
from .musicxml.to_musicxml import pnl_to_musicxml
from .parser import parse

INSTALL_HINT = 'pip install "pnl2[engrave]"'
SUPPORTED_FORMATS = ("svg", "png", "html")

_DEFAULT_OPTIONS: dict[str, Any] = {
    "inputFrom": "xml",
    "adjustPageHeight": True,
    "breaks": "auto",
}

_SVG_OPEN = re.compile(r"<svg\b([^>]*)>(.*)</svg>\s*\Z", re.IGNORECASE | re.DOTALL)
_VIEWBOX = re.compile(r'\bviewBox="([^"]+)"', re.IGNORECASE)
_WIDTH = re.compile(r'\bwidth="([^"]+)"', re.IGNORECASE)
_HEIGHT = re.compile(r'\bheight="([^"]+)"', re.IGNORECASE)
_XML_DECL = re.compile(r"^\s*<\?xml[^?]*\?>\s*", re.IGNORECASE)
_TEXT_BLOCK = re.compile(r"<text\b([^>]*)>(.*?)</text>", re.IGNORECASE | re.DOTALL)
_LEIPZIG_TSPAN = re.compile(
    r'<tspan([^>]*font-family="Leipzig"[^>]*)>([^<]*)</tspan>',
    re.IGNORECASE,
)
_SMUFL = re.compile(r"[\uE000-\uF8FF]")
_UPEM = 1000.0
# Leipzig metNoteQuarterUp — used for ♩ = bpm when the WOFF font cannot be loaded.
_FALLBACK_GLYPHS = {
    "ECA5": (
        "M203 126c29 0 53 -9 70 -24l4 -4l3 570h22l-2 -625c0 -85 -111 -169 "
        "-201 -169c-55 0 -97 31 -97 82c0 87 89 170 201 170z",
        302.0,
    ),
}


class EngraverError(Exception):
    """Raised when engraving cannot proceed (missing deps, bad input, Verovio failure)."""


_toolkit_lock = threading.Lock()
_toolkit: Any = None
_toolkit_factory: Any = None


def reset_verovio_toolkit() -> None:
    """Drop the cached Verovio toolkit (tests and resource reloads)."""
    global _toolkit, _toolkit_factory
    with _toolkit_lock:
        _toolkit = None
        _toolkit_factory = None


def warmup_verovio() -> None:
    """Create the toolkit on the current thread so later workers can reuse it."""
    try:
        verovio = _import_verovio()
    except EngraverError:
        return
    with _toolkit_lock:
        _get_toolkit(verovio)


def engrave_svg(
    source: str | Path | Document,
    *,
    options: dict[str, Any] | None = None,
) -> list[str]:
    """Parse *source* and return one Verovio SVG string per page."""
    doc = load_document(source)
    return render_svg_pages(pnl_to_musicxml(doc), options=options)


def engrave(
    source: str | Path | Document,
    output: str | Path | None = None,
    *,
    format: str | None = None,
    scale: float | None = None,
    options: dict[str, Any] | None = None,
) -> str | Path | list[Path]:
    """Engrave a PNL/2 score.

    *source* may be a filesystem path, PNL/2 text, or a parsed ``Document``.
    When *output* is omitted the combined SVG (or HTML) is returned as a string.
    PNG requires an output path. Multi-page PNG writes ``stem-1.png``, ``stem-2.png``, …
    """
    dest = Path(output) if output is not None else None
    fmt = infer_format(dest, format)
    doc = load_document(source)
    pages = render_svg_pages(pnl_to_musicxml(doc), options=options)
    title = str(doc.score.meta.get("title") or "Score")

    if fmt == "svg":
        payload = combine_svgs(pages)
        if dest is None:
            return payload
        dest.write_text(payload, encoding="utf-8")
        return dest

    if fmt == "html":
        payload = wrap_html(pages, title=title)
        if dest is None:
            return payload
        dest.write_text(payload, encoding="utf-8")
        return dest

    if dest is None:
        raise EngraverError("PNG output requires a file path")
    return write_png_pages(pages, dest, scale=scale if scale is not None else 2.0)


def infer_format(output: Path | None, format: str | None) -> str:
    """Resolve output format from an explicit name or the destination suffix."""
    if format:
        fmt = format.lower().lstrip(".")
        if fmt not in SUPPORTED_FORMATS:
            raise EngraverError(
                f"Unsupported format {format!r}. Choose one of: {', '.join(SUPPORTED_FORMATS)}"
            )
        return fmt
    if output is None:
        return "svg"
    suffix = output.suffix.lower().lstrip(".")
    if suffix == "htm":
        return "html"
    if suffix in SUPPORTED_FORMATS:
        return suffix
    raise EngraverError(
        f"Cannot infer format from {output}. Use format='svg', 'png', or 'html'."
    )


def load_document(source: str | Path | Document) -> Document:
    """Accept a path, PNL/2 text, or an already-parsed document."""
    if isinstance(source, Document):
        return source
    if isinstance(source, Path):
        return parse(source.read_text(encoding="utf-8"))
    if isinstance(source, str):
        path = Path(source)
        if "\n" not in source and path.is_file():
            return parse(path.read_text(encoding="utf-8"))
        return parse(source)
    raise TypeError(f"Cannot engrave source of type {type(source).__name__}")


def render_svg_pages(musicxml: str, *, options: dict[str, Any] | None = None) -> list[str]:
    """Load MusicXML into Verovio and render every page to SVG."""
    verovio = _import_verovio()
    merged = {**_DEFAULT_OPTIONS, **(options or {})}
    merged["inputFrom"] = "xml"
    with _toolkit_lock:
        toolkit = _get_toolkit(verovio)
        toolkit.setOptions(merged)
        loaded = toolkit.loadData(musicxml)
        if loaded is False:
            raise EngraverError(_load_failure_message(toolkit))
        count = int(toolkit.getPageCount())
        if count < 1:
            raise EngraverError("Verovio produced no pages")
        return [outline_smufl_text(toolkit.renderToSVG(page)) for page in range(1, count + 1)]


def _verovio_data_dir(verovio: Any) -> Path | None:
    path = Path(getattr(verovio, "__file__", "") or ".").resolve().parent / "data"
    return path if path.is_dir() else None


def _get_toolkit(verovio: Any) -> Any:
    global _toolkit, _toolkit_factory
    factory = verovio.toolkit
    if _toolkit is None or _toolkit_factory is not factory:
        data = _verovio_data_dir(verovio)
        if data is not None and hasattr(verovio, "setDefaultResourcePath"):
            verovio.setDefaultResourcePath(str(data))
        toolkit = factory()
        if data is not None and hasattr(toolkit, "setResourcePath"):
            toolkit.setResourcePath(str(data))
        _toolkit = toolkit
        _toolkit_factory = factory
    return _toolkit


def _load_failure_message(toolkit: Any) -> str:
    detail = ""
    if hasattr(toolkit, "getLog"):
        log = str(toolkit.getLog() or "").strip()
        if log:
            last = log.splitlines()[-1].strip()
            detail = f" ({last})"
    return f"Verovio could not load the generated MusicXML{detail}"


def outline_smufl_text(svg: str) -> str:
    """Replace Leipzig text glyphs with SVG paths.

    Verovio draws metronome notes (and a few other symbols) as SMuFL characters
    in the embedded Leipzig WOFF. CairoSVG and many viewers ignore ``@font-face``,
    so those characters become empty squares. Staff symbols are already paths;
    this does the same for in-text music glyphs.
    """
    if "font-family=\"Leipzig\"" not in svg and "font-family='Leipzig'" not in svg:
        return svg

    needed: set[str] = set()

    def replace_text(match: re.Match[str]) -> str:
        attrs, inner = match.group(1), match.group(2)
        leipzig = list(_LEIPZIG_TSPAN.finditer(inner))
        if not leipzig:
            return match.group(0)
        text_x = _attr_number(attrs, "x", 0.0)
        text_y = _attr_number(attrs, "y", 0.0)
        cursor = 0.0
        chunks: list[str] = []
        for span in leipzig:
            size = _attr_number(span.group(1), "font-size", 720.0)
            scale = size / _UPEM
            for char in span.group(2):
                if not _SMUFL.match(char):
                    continue
                code = f"{ord(char):04X}"
                if not _glyph_path(code):
                    continue
                needed.add(code)
                x = text_x + cursor
                chunks.append(
                    f'<use xlink:href="#smufl-{code}" '
                    f'transform="translate({x:g}, {text_y:g}) scale({scale:g}, {scale:g})"/>'
                )
                cursor += _glyph_advance(code) * scale
        if not chunks:
            return match.group(0)
        stripped = _LEIPZIG_TSPAN.sub("", inner)
        dx = f'{attrs} dx="{cursor:g}"' if cursor else attrs
        return "".join(chunks) + f"<text{dx}>{stripped}</text>"

    svg = _TEXT_BLOCK.sub(replace_text, svg)
    if not needed:
        return svg
    defs = "".join(_glyph_def(code) for code in sorted(needed))
    return _insert_into_defs(svg, defs)


def combine_svgs(pages: list[str], *, gap: float = 80.0) -> str:
    """Return a single SVG, stacking pages vertically when there is more than one."""
    if not pages:
        raise EngraverError("No SVG pages to combine")
    if len(pages) == 1:
        return pages[0]
    parsed = [_split_svg(page) for page in pages]
    width = max(dims[0] for _, dims in parsed)
    total_h = sum(dims[1] for _, dims in parsed) + gap * (len(pages) - 1)
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width:g} {total_h:g}" width="{width:g}" height="{total_h:g}">'
    ]
    y = 0.0
    for inner, (_w, h) in parsed:
        chunks.append(f'<g transform="translate(0 {y:g})">')
        chunks.append(inner)
        chunks.append("</g>")
        y += h + gap
    chunks.append("</svg>")
    return "\n".join(chunks)


def flatten_nested_svgs(svg: str) -> str:
    """Unwrap nested ``<svg>`` so Qt (SVG Tiny 1.2) can paint Verovio output.

    Verovio puts the score in an inner ``<svg class="definition-scale">``.
    QSvgWidget skips that element, which leaves the studio preview blank.
    """
    body = _XML_DECL.sub("", svg).strip()
    match = _SVG_OPEN.match(body)
    if not match:
        return svg
    attrs, inner = match.group(1), match.group(2)
    if not re.search(r"<svg\b", inner, re.IGNORECASE):
        return _with_white_page(f"<svg{attrs}>{inner}</svg>", attrs)

    viewbox = None
    found = _VIEWBOX.search(attrs)
    if found:
        viewbox = found.group(1)

    def _open(nest: re.Match[str]) -> str:
        nonlocal viewbox
        nest_attrs = nest.group(1)
        if viewbox is None:
            inner_box = _VIEWBOX.search(nest_attrs)
            if inner_box:
                viewbox = inner_box.group(1)
        return f"<g{nest_attrs}>"

    inner = re.sub(r"<svg\b([^>]*)>", _open, inner, flags=re.IGNORECASE)
    inner = re.sub(r"</svg>", "</g>", inner, flags=re.IGNORECASE)
    if viewbox and not _VIEWBOX.search(attrs):
        attrs = f'{attrs} viewBox="{viewbox}"'
    return _with_white_page(f"<svg{attrs}>{inner}</svg>", attrs)


def _with_white_page(svg: str, attrs: str) -> str:
    width, height = _svg_dims(attrs)
    bg = f'<rect class="page-bg" x="0" y="0" width="{width:g}" height="{height:g}" fill="white"/>'
    return re.sub(r"(<svg\b[^>]*>)", rf"\1{bg}", svg, count=1, flags=re.IGNORECASE)


def wrap_html(pages: list[str], *, title: str = "Score") -> str:
    """Wrap page SVGs in a simple, self-contained HTML preview."""
    blocks = "\n".join(f'<div class="page">\n{page}\n</div>' for page in pages)
    safe_title = html.escape(title)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8"/>\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1"/>\n'
        f"  <title>{safe_title}</title>\n"
        "  <style>\n"
        "    html, body { margin: 0; padding: 0; background: #f4f1ea; }\n"
        "    main { max-width: 960px; margin: 1.5rem auto; padding: 0 1rem 2rem; }\n"
        "    h1 { font: 600 1.1rem/1.4 system-ui, sans-serif; color: #333; }\n"
        "    .page { background: #fff; margin: 1.25rem 0; padding: 0.75rem;\n"
        "            box-shadow: 0 1px 8px rgba(0, 0, 0, 0.08); }\n"
        "    .page svg { display: block; width: 100%; height: auto; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        "  <main>\n"
        f"    <h1>{safe_title}</h1>\n"
        f"{blocks}\n"
        "  </main>\n"
        "</body>\n"
        "</html>\n"
    )


def write_png_pages(pages: list[str], dest: Path, *, scale: float = 2.0) -> Path | list[Path]:
    """Rasterize SVG pages with CairoSVG. Multiple pages become ``stem-N.png``."""
    cairosvg = _import_cairosvg()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if len(pages) == 1:
        _svg_to_png(cairosvg, pages[0], dest, scale)
        return dest
    written: list[Path] = []
    for index, page in enumerate(pages, start=1):
        path = dest.with_name(f"{dest.stem}-{index}{dest.suffix or '.png'}")
        _svg_to_png(cairosvg, page, path, scale)
        written.append(path)
    return written


def _svg_to_png(cairosvg: Any, svg: str, dest: Path, scale: float) -> None:
    try:
        cairosvg.svg2png(
            bytestring=svg.encode("utf-8"),
            write_to=str(dest),
            scale=scale,
            background_color="white",
        )
    except OSError as exc:
        raise EngraverError(_cairo_os_error(exc)) from exc


def _import_verovio():
    try:
        import verovio
    except ImportError as exc:
        raise EngraverError(
            f"Verovio is required to engrave scores. Install with: {INSTALL_HINT}"
        ) from exc
    return verovio


def _import_cairosvg():
    try:
        import cairosvg
    except ImportError as exc:
        raise EngraverError(
            f"CairoSVG is required to write PNG. Install with: {INSTALL_HINT}"
        ) from exc
    except OSError as exc:
        raise EngraverError(_cairo_os_error(exc)) from exc
    return cairosvg


def _cairo_os_error(exc: OSError) -> str:
    return (
        f"CairoSVG could not load the Cairo library ({exc}). "
        "Install cairo (e.g. `brew install cairo` on macOS) "
        f"or write SVG/HTML instead. {INSTALL_HINT}"
    )


def _split_svg(svg: str) -> tuple[str, tuple[float, float]]:
    body = _XML_DECL.sub("", svg).strip()
    match = _SVG_OPEN.match(body)
    if not match:
        raise EngraverError("Verovio produced SVG that could not be parsed")
    return match.group(2), _svg_dims(match.group(1))


def _svg_dims(attrs: str) -> tuple[float, float]:
    view = _VIEWBOX.search(attrs)
    if view:
        parts = view.group(1).split()
        if len(parts) == 4:
            return _css_number(parts[2]), _css_number(parts[3])
    width = _WIDTH.search(attrs)
    height = _HEIGHT.search(attrs)
    if width and height:
        return _css_number(width.group(1)), _css_number(height.group(1))
    return 2100.0, 2970.0


def _css_number(value: str) -> float:
    return float(re.sub(r"[a-zA-Z%]+$", "", value.strip()))


def _attr_number(attrs: str, name: str, default: float) -> float:
    match = re.search(rf'\b{name}="([^"]+)"', attrs, re.IGNORECASE)
    if not match:
        return default
    try:
        return _css_number(match.group(1))
    except ValueError:
        return default


def _glyph_path(code: str) -> str | None:
    xml = _leipzig_glyph_xml(code)
    if xml:
        match = re.search(r'<path[^>]*\bd="([^"]+)"', xml)
        if match:
            return match.group(1)
    fallback = _FALLBACK_GLYPHS.get(code)
    return fallback[0] if fallback else None


def _glyph_advance(code: str) -> float:
    xml = _leipzig_metrics_xml()
    if xml:
        match = re.search(rf'<g c="{code}"[^>]*h-a-x="([^"]+)"', xml, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
    fallback = _FALLBACK_GLYPHS.get(code)
    return fallback[1] if fallback else 300.0


def _glyph_def(code: str) -> str:
    path = _glyph_path(code)
    if not path:
        return ""
    return f'<g id="smufl-{code}"><path transform="scale(1,-1)" d="{path}"/></g>'


def _insert_into_defs(svg: str, extra: str) -> str:
    if re.search(r"<defs\b", svg, re.IGNORECASE):
        return re.sub(r"(<defs\b[^>]*>)", rf"\1{extra}", svg, count=1, flags=re.IGNORECASE)
    return re.sub(r"(<svg\b[^>]*>)", rf"\1<defs>{extra}</defs>", svg, count=1, flags=re.IGNORECASE)


def _leipzig_dir() -> Path | None:
    try:
        import verovio
    except ImportError:
        return None
    path = Path(verovio.__file__).resolve().parent / "data" / "Leipzig"
    return path if path.is_dir() else None


def _leipzig_glyph_xml(code: str) -> str | None:
    folder = _leipzig_dir()
    if folder is None:
        return None
    path = folder / f"{code}.xml"
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _leipzig_metrics_xml() -> str | None:
    folder = _leipzig_dir()
    if folder is None:
        return None
    path = folder.parent / "Leipzig.xml"
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")
