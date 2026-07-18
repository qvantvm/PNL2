# PNL/2 ↔ MusicXML mapping

Converters live in `pnl2.musicxml` and aim for lossless round-trips for common piano scores (notes, chords, rests, meters, keys, clefs, ties, slurs, articulations, dynamics, pedals, grace notes, tuplets, beams, fingerings).

## MusicXML → PNL/2

| MusicXML | PNL/2 |
| --- | --- |
| `<score-partwise>` / `<score-timewise>` | `score` + `part` |
| `<work-title>`, `<creator>` | `meta title=… composer=…` |
| `<part>` | `part ID instrument=… staves=N` |
| `<measure number="N">` | `measure N { … }` |
| staff number 1 / 2 (piano) | `staff RH` / `staff LH` (configurable) |
| `<voice>` | `voice V{n}` |
| `<note>` with pitch | `note` / chord `tone` |
| chorded notes (`<chord/>`) | `chord { tone … }` |
| `<rest/>` | `rest` |
| duration + type + dots | `dur` + optional `augment` |
| `<time-modification>` | `tuplet=actual:normal` |
| `<tie type="start/stop">` | `tie ID from=… to=…` |
| `<slur type="start/stop">` | `slur ID from=… to=…` |
| `<articulations>` / `<technical>` | `art=[…]`, `finger=…` |
| `<dynamics>` / wedge | `dynamic`, `hairpin` |
| pedal directions | `pedal` spans |
| `<grace/>` | `grace` / `grace-group` |
| `<beam>` | `beam` groups |
| `<key>`, `<time>`, `<clef>` | `key`, `meter`, `clef` |
| `<sound tempo>` / metronome | `tempo` |

### Duration mapping

MusicXML divisions are converted to whole-note rationals:

```text
dur = duration / (divisions × 4)
```

then reduced. Dots become `augment=N` when the MusicXML `<dot/>` count is present and the type maps cleanly; otherwise an equivalent reduced `dur` is emitted.

### Pitch mapping

```text
step + alter + octave → pitch letter + accidental + octave
```

| alter | accidental |
| ---: | --- |
| -2 | `bb` |
| -1 | `b` |
| 0 | (none) |
| 1 | `#` |
| 2 | `##` |

### Staff naming

Default piano mapping: staff 1 → `RH`, staff 2 → `LH`. Override with `--staff-map`.

## PNL/2 → MusicXML

The reverse converter emits MusicXML 3.1 partwise scores:

- One `<part>` per PNL part
- Measures contain notes ordered by staff/voice onset
- Chords emit the first tone as a normal note and subsequent tones with `<chord/>`
- Ties/slurs become `<tied>` / `<slur>` with matching numbers
- Articulations, dynamics, pedals, clefs, keys, meters, tempos are restored when present
- Rational durations choose a suitable `divisions` value (LCM of denominators within each part)

### Round-trip guarantees

Preserved when both sides support the construct:

- pitches (enharmonic spelling)
- written durations and dots
- voices and staves
- ties and slurs
- common articulations and dynamics
- fingerings
- sustain pedal start/stop
- tuplets
- grace notes (basic)

Not yet fully round-tripped (may be dropped or approximated):

- analytical layers (`roman`, `harmonic-edge`, `cadence`, …)
- performance millisecond data (`performed-note`, `pedal-curve` with `at-ms`)
- microtonal `cents`
- hierarchical form analysis

These remain valid PNL/2 and are kept on convert-from-MusicXML only when sourced from MusicXML; analytical layers authored in PNL are preserved by the PNL parser/serializer but omitted from MusicXML export unless a future extension maps them to MusicXML harmony/analysis elements.

## CLI

```bash
# Import
pnl2 from-musicxml input.musicxml -o output.pnl

# Export
pnl2 to-musicxml input.pnl -o output.musicxml

# Round-trip check (parse + re-serialize)
pnl2 parse input.pnl --canonical -o canonical.pnl
pnl2 validate input.pnl
```
