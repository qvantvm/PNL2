# PNL/2 — Piano Notation Language

Compact, deterministic symbolic piano notation for ML training, MusicXML interchange, and tonal analysis.

## Features

- Formal EBNF grammar and language documentation
- Parser, canonical serializer, and validator
- MusicXML → PNL/2 and PNL/2 → MusicXML converters
- Verovio-based score engraver (SVG, PNG, HTML)
- PyQt6 Sample Studio for authoring `.pnl` + reference-image pairs
- Exact rational durations; explicit dots vs articulations; ID-based ties/slurs

## Install

```bash
pip install -e ".[dev]"
```

To engrave scores (Verovio + CairoSVG):

```bash
pip install -e ".[dev,engrave]"
```

PNG rasterization also needs the system Cairo library (`brew install cairo` on macOS). SVG and HTML do not.

For the sample studio (PyQt6):

```bash
pip install -e ".[dev,engrave,studio]"
```

## CLI

```bash
# MusicXML → PNL/2
pnl2 from-musicxml score.musicxml -o score.pnl

# PNL/2 → MusicXML
pnl2 to-musicxml score.pnl -o score.musicxml

# Engrave to SVG / PNG / HTML
pnl2 engrave score.pnl -o score.svg
pnl2 engrave score.pnl -o score.png --scale 2
pnl2 engrave score.pnl -o score.html

# Validate / canonicalize
pnl2 validate score.pnl
pnl2 parse score.pnl --canonical -o score.canonical.pnl

# Sample studio (edit / live-engrave / compare to a reference image)
pnl2 studio
pnl2 studio examples/simple.pnl
pnl2 studio /path/to/harmony_dataset/samples
```

## Library

```python
from pnl2 import parse, serialize, validate
from pnl2.musicxml import musicxml_to_pnl, pnl_to_musicxml
from pnl2.engraver import engrave, engrave_svg

doc = parse(open("score.pnl").read())
print(validate(doc))
print(serialize(doc))

pnl = musicxml_to_pnl(open("score.musicxml").read())
xml = pnl_to_musicxml(pnl)

engrave("score.pnl", "score.svg")
pages = engrave_svg(doc)
```

## Sample Studio

Four-pane editor for dataset samples: PNL script (top-left), live Verovio preview (top-right), parse/engrave log (bottom-left), and a reference image of the intended result (bottom-right).

```bash
pnl2 studio
pnl2 studio /path/to/harmony_dataset/samples
# or
python -m pnl2.studio examples/simple.pnl
```

The studio opens a dataset folder (File → Open Dataset Folder, or pass a directory on the CLI) and lists every `.pnl` in it. Sidecar `expected` paths are resolved relative to the sidecar, including pointers like `../../ch03/page_002/homr_crops/….png`. Previous/Next (Ctrl+[ / Ctrl+]) steps through the folder. The left pane has a **Metadata** tab for sidecar fields (`title`, `caption`, `constructs`, `split`, and so on). Both the live engraving and the reference image have Zoom + Fit (Ctrl+0 / Ctrl+Shift+0).

Save updates the `.pnl` in place and keeps an existing sidecar `expected` path — it does not copy the crop into the samples folder. Save As still writes `name.pnl`, optional `name.png`, and `name.sample.json`. `PNL2_SAMPLES_DIR` overrides the default library; otherwise the studio prefers a folder that already has samples (the harmony extraction dataset when present, then `./samples`).

## Layout

| Path | Contents |
| --- | --- |
| `grammar/pnl2.ebnf` | Formal grammar |
| `docs/LANGUAGE.md` | Language reference |
| `docs/MUSICXML.md` | Conversion mapping |
| `src/pnl2/` | Parser, serializer, validator, converters, engraver, studio |
| `examples/` | Sample `.pnl` and MusicXML files |
| `samples/` | Working folder for studio dataset pairs |
| `tests/` | Unit and round-trip tests |

## Design principles

1. Every symbol has exactly one meaning.
2. Written duration and articulation are separate (`augment` vs `art=[staccato]`).
3. Notation and performance are separate layers.
4. Slurs, ties, and analysis use explicit IDs.
5. Timing uses exact rationals.

## Example

```text
pnl/2
score {
    meta {
        title="Articulation Example"
    }
    part piano instrument=piano staves=2 {
        meter at=1:0 beats=4 beat-unit=1/4
        key at=1:0 tonic=C mode=major
        measure 1 {
            staff RH {
                voice RH1 {
                    note n1 pitch=C5 dur=1/4 augment=1 finger=1 art=[accent]
                    note n2 pitch=D5 dur=1/8 finger=2 art=[staccato]
                    note n3 pitch=E5 dur=1/4 finger=3
                    note n4 pitch=G5 dur=1/4 finger=5
                }
            }
            staff LH {
                voice LH1 {
                    rest r1 dur=1
                }
            }
        }
        slur s1 from=n1 to=n4 placement=above
    }
}
```

## Tests

```bash
pytest -q
```
