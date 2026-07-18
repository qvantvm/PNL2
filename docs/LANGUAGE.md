# PNL/2 — Piano Notation Language

PNL/2 is a compact, deterministic representation for symbolic piano music. It targets machine-learning training, lossless MusicXML conversion where possible, MIDI conversion with explicit notation/performance separation, polyphonic piano writing, advanced harmony, and stable tokenization.

## Design principles

1. Every symbol has exactly one meaning.
2. Written duration and articulation are separate properties.
3. Notation and performance data are separate layers.
4. Relationships such as slurs and ties use explicit identifiers.
5. All timing values use exact rational numbers.

A dot is never used for both dotted rhythm and staccato:

```text
dur=1/4 augment=1          # dotted quarter
art=[staccato]             # staccato articulation
```

## Document structure

Every file begins with `pnl/2` and contains one `score` block:

```text
pnl/2
score {
    meta {
        title="Example"
        composer="Anonymous"
        profile=[core,notation]
    }
    part piano instrument=piano staves=2 {
        ...
    }
}
```

Hierarchy:

```text
score → part → measure → staff → voice → event
```

Relations (`slur`, `tie`, `pedal`, `roman`, …) may appear at part level after measures, or inside `notation` / `performance` layers.

## Time model

### Durations

All symbolic durations are reduced rational fractions of a whole note:

| Written value | PNL duration |
| --- | --- |
| Whole | `1` |
| Half | `1/2` |
| Quarter | `1/4` |
| Eighth | `1/8` |
| Sixteenth | `1/16` |

Valid: `dur=1/4`, `dur=3/8`. Invalid: `dur=2/8`, `dur=0.25`.

### Augmentation dots

```text
dur=1/4 augment=1   # effective = 1/4 × 3/2 = 3/8
dur=1/4 augment=2   # effective = 1/4 × 7/4 = 7/16
```

Formula: `effective = dur × (2 − 1 / 2^augment)`.

`dur=3/8` and `dur=1/4 augment=1` share temporal length but differ in written notation.

### Tuplets

```text
note n1 pitch=C4 dur=1/8 tuplet=3:2   # effective = 1/8 × 2/3 = 1/12
```

### Positions

`measure:offset` with offset in whole-note units from the measure start:

```text
1:0
1:1/4
2:3/8
```

Inside a voice, omitted `at` places events sequentially by effective duration.

## Pitch

```text
pitch=C4
pitch=F#4
pitch=Bb3
pitch=C##5
pitch=Ebb4
```

Accidentals: `bb`, `b`, (none), `#`, `##`. Enharmonic spellings are preserved (`C#4` ≠ `Db4`).

Optional: `accidental-display=natural`, `cents=50`.

## Events

### Notes

```text
note n1 pitch=F#4 dur=1/8 augment=1 at=0 finger=3 art=[accent,tenuto] dynamic=mf
```

Required: `pitch`, `dur`. IDs are required for any related event.

Canonical property order for notes:

`pitch dur augment tuplet at staff hand voice finger finger-change art ornament dynamic role velocity cents`

### Chords

```text
chord c1 dur=1/4 art=[accent] {
    tone n1 pitch=C4 finger=1
    tone n2 pitch=E4 finger=3
    tone n3 pitch=G4 finger=5
}
```

A chord needs at least two tones. Tone-only properties stay on `tone` lines. Use `perf-offset` for performance delay, never tone-level `at`.

### Rests and spacers

```text
rest r1 dur=1/4
rest r2 dur=1/8 visible=false
space dur=1/4
```

`space` advances the notation cursor without implying silence.

### Grace notes

```text
grace-group gg1 placement=before anchor=n3 steal-from=following steal-ratio=1/8 {
    grace g1 pitch=C5 type=acciaccatura
    grace g2 pitch=D5
}
```

## Articulations and dynamics

```text
art=[staccato]
art=[tenuto,accent]
dynamic=mf
dynamic d1 value=mf at=1:0 staff=RH
hairpin h1 type=crescendo from=1:0 to=1:1
hairpin h2 type=diminuendo from-event=n1 to-event=n8
```

A hairpin must use either `from`/`to` or `from-event`/`to-event`, never mixed.

## Relations

```text
slur s1 from=n1 to=n4 placement=above
tie t1 from=n1 to=n2
phrase ph1 from=n1 to=n16 type=antecedent
beam b1 level=1 notes=[n1,n2,n3,n4]
```

Ties require sounding-pitch equality and contiguous timing unless `allow-gap=true`.

## Pedals

```text
pedal p1 type=sustain from=1:0 to=1:1 depth=1
pedal-curve pc1 type=sustain source=performance {
    point at-ms=0 value=0
    point at-ms=90 value=1
    point at-ms=1450 value=0
}
```

Types: `sustain`, `sostenuto`, `soft`. Depth/value ∈ `[0,1]`.

## Harmony and analysis

PNL/2 separates chord symbols, Roman numerals, and pitch structure:

```text
chord-symbol cs1 at=1:0 root=C quality=dominant extension=13 bass=Bb alter=[b9,#11]
roman rn1 from=1:0 to=1:1/2 degree=5 quality=dominant seventh=true inversion=third key=D:major
harmonic-edge he1 from=rn1 to=rn2 relation=resolves-to
cadence cad1 from=4:0 to=4:1 type=perfect-authentic key=C:major
```

## Layers

```text
notation { ... }
performance timing-unit=ms {
    performed-note pn1 source=n1 onset-ms=0 key-release-ms=180 sound-release-ms=720 velocity=82
}
```

Written duration, key release, and sound release are never collapsed silently.

## Profiles

Declare in meta:

```text
profile=[core,notation,analysis,performance]
```

| Profile | Contents |
| --- | --- |
| Core | structure, notes, chords, rests, durations, voices, keys, meters, tempo |
| Notation | fingerings, articulations, slurs, ties, dynamics, beams, ornaments, pedals |
| Analysis | chord symbols, roman, functions, phrases, cadences, harmonic edges |
| Performance | measured onsets, releases, velocities, pedal curves |

## Grammar and validation

- Formal grammar: [`grammar/pnl2.ebnf`](../grammar/pnl2.ebnf)
- Validation rules: section 30 of the language specification (unique IDs, reduced rationals, chord ≥ 2 tones, tie pitch/timing, finger 1–5, pedal depth ∈ [0,1], …)

## MusicXML conversion

```bash
pnl2 from-musicxml score.musicxml -o score.pnl
pnl2 to-musicxml score.pnl -o score.musicxml
pnl2 validate score.pnl
pnl2 parse score.pnl --canonical -o out.pnl
```

See [`docs/MUSICXML.md`](MUSICXML.md) for mapping details.
