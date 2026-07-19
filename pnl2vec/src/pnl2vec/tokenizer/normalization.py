"""Pitch and token normalization helpers."""

from __future__ import annotations

from pnl2.ast import Pitch

# Sounding pitch-class names (C major / sharp-side preference)
_MIDI_PC = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_ACCIDENTAL_NAME = {
    "": "NATURAL",
    "b": "FLAT",
    "bb": "DOUBLE_FLAT",
    "#": "SHARP",
    "##": "DOUBLE_SHARP",
}


def accidental_name(acc: str) -> str:
    return _ACCIDENTAL_NAME.get(acc, acc.upper() or "NATURAL")


def normalize_pitch(
    pitch: Pitch,
    *,
    normalize_enharmonics: bool = False,
    preserve_pitch_spelling: bool = True,
) -> tuple[str, str, int]:
    """Return (pitch_class, accidental_name, octave)."""
    if normalize_enharmonics and not preserve_pitch_spelling:
        midi = pitch.sounding_midi()
        pc = _MIDI_PC[midi % 12]
        octave = midi // 12 - 1
        if len(pc) == 1:
            return pc, "NATURAL", octave
        # sharp spelling
        return pc[0], "SHARP", octave
    return pitch.letter, accidental_name(pitch.accidental), pitch.octave


def duration_token_value(dur: object) -> str:
    """Canonical duration string from Rational or str."""
    return str(dur)


def key_token_value(tonic: str, mode: str) -> str:
    mode_u = mode.upper().replace("-", "_")
    t = tonic.strip()
    if len(t) >= 2 and t.endswith("#"):
        tonic_u = t[0].upper() + "_SHARP"
    elif len(t) >= 2 and t[-1].lower() == "b":
        tonic_u = t[0].upper() + "_FLAT"
    else:
        tonic_u = t[0].upper()
    return f"{tonic_u}_{mode_u}"


def time_signature_value(beats: object, beat_unit: object) -> str:
    return f"{beats}/{beat_unit}"
