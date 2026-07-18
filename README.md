# PNL/2 — Piano Notation Language

Compact, deterministic symbolic piano notation for ML training, MusicXML interchange, and tonal analysis.

## Features

- Formal EBNF grammar and language documentation
- Parser, canonical serializer, and validator
- MusicXML → PNL/2 and PNL/2 → MusicXML converters
- Exact rational durations; explicit dots vs articulations; ID-based ties/slurs

## Install

```bash
pip install -e ".[dev]"
```

## CLI

```bash
# MusicXML → PNL/2
pnl2 from-musicxml score.musicxml -o score.pnl

# PNL/2 → MusicXML
pnl2 to-musicxml score.pnl -o score.musicxml

# Validate / canonicalize
pnl2 validate score.pnl
pnl2 parse score.pnl --canonical -o score.canonical.pnl
```

## Library

```python
from pnl2 import parse, serialize, validate
from pnl2.musicxml import musicxml_to_pnl, pnl_to_musicxml

doc = parse(open("score.pnl").read())
print(validate(doc))
print(serialize(doc))

pnl = musicxml_to_pnl(open("score.musicxml").read())
xml = pnl_to_musicxml(pnl)
```

## Layout

| Path | Contents |
| --- | --- |
| `grammar/pnl2.ebnf` | Formal grammar |
| `docs/LANGUAGE.md` | Language reference |
| `docs/MUSICXML.md` | Conversion mapping |
| `src/pnl2/` | Parser, serializer, validator, converters |
| `examples/` | Sample `.pnl` and MusicXML files |
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
