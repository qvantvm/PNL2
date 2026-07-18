# PNL/2 specification index

This repository implements **PNL/2** (Piano Notation Language).

## Canonical artifacts

| Artifact | Location |
| --- | --- |
| Formal EBNF grammar | [`grammar/pnl2.ebnf`](../grammar/pnl2.ebnf) |
| Language reference | [`LANGUAGE.md`](LANGUAGE.md) |
| MusicXML mapping | [`MUSICXML.md`](MUSICXML.md) |
| Reference examples | [`examples/`](../examples/) |

## Implementation modules

| Module | Role |
| --- | --- |
| `pnl2.lexer` / `pnl2.parser` | Tokenize and parse `.pnl` text into an AST |
| `pnl2.serializer` | Emit canonical PNL/2 |
| `pnl2.validator` | Enforce identity, timing, chord, tie, fingering, harmony, and pedal rules |
| `pnl2.rational` | Exact rational durations and effective-duration math |
| `pnl2.musicxml.from_musicxml` | MusicXML → PNL/2 |
| `pnl2.musicxml.to_musicxml` | PNL/2 → MusicXML 3.1 partwise |
| `pnl2.cli` | `pnl2` command-line entry point |

## Profiles

Datasets should declare a profile in `meta`:

```text
profile=[core,notation,analysis,performance]
```

Converters currently round-trip **core** and **notation** layers through MusicXML. Analysis and performance layers are first-class in the PNL/2 parser/serializer and are documented for interchange even when MusicXML has no lossless home for them.
