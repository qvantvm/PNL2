"""Deterministic PNL/2 → token sequence tokenization."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

from pnl2.ast import Document, Node, Pitch
from pnl2.parser import parse_pitch

from .normalization import (
    duration_token_value,
    key_token_value,
    normalize_pitch,
    time_signature_value,
)
from .token import Token, TokenKind, make_special, token_attrs

TokenizerMode = Literal["atomic", "compound"]


@dataclass
class TokenizerConfig:
    mode: TokenizerMode = "atomic"
    compound_min_frequency: int = 20
    normalize_enharmonics: bool = False
    preserve_pitch_spelling: bool = True
    preserve_source_spans: bool = True
    add_special_boundaries: bool = True


@dataclass
class AnnotatedToken:
    """Token plus grouping metadata for context policies."""

    token: Token
    event_id: str | None = None
    measure: int | None = None
    voice: str | None = None
    hand: str | None = None
    offset: str | None = None
    is_compound: bool = False


@dataclass
class Tokenizer:
    config: TokenizerConfig = field(default_factory=TokenizerConfig)
    _compound_freq: Counter[str] = field(default_factory=Counter)
    _allowed_compounds: set[str] = field(default_factory=set)

    def fit_compound_frequencies(self, documents: Iterable[Document]) -> None:
        """Count potential compound event keys from a training corpus."""
        self._compound_freq = Counter()
        for doc in documents:
            for key, _ in self._iter_compound_candidates(doc):
                self._compound_freq[key] += 1
        thresh = self.config.compound_min_frequency
        self._allowed_compounds = {k for k, c in self._compound_freq.items() if c >= thresh}

    def tokenize(self, document: Document) -> list[Token]:
        return [a.token for a in self.tokenize_annotated(document)]

    def tokenize_text(self, text: str) -> list[Token]:
        from pnl2vec.pnl import parse_pnl

        return self.tokenize(parse_pnl(text))

    def tokenize_annotated(self, document: Document) -> list[AnnotatedToken]:
        out: list[AnnotatedToken] = []
        if self.config.add_special_boundaries:
            out.append(self._ann(make_special("BOS")))
            out.append(self._ann(Token(TokenKind.STRUCT, "DOC_START")))

        for part in document.score.parts:
            out.extend(self._tokenize_part(part))

        for stmt in document.score.statements:
            out.extend(self._tokenize_statement(stmt, measure=None, voice=None, hand=None))

        if self.config.add_special_boundaries:
            out.append(self._ann(Token(TokenKind.STRUCT, "DOC_END")))
            out.append(self._ann(make_special("EOS")))
        return out

    def _ann(
        self,
        token: Token,
        *,
        event_id: str | None = None,
        measure: int | None = None,
        voice: str | None = None,
        hand: str | None = None,
        offset: str | None = None,
        is_compound: bool = False,
    ) -> AnnotatedToken:
        # Embed grouping into Token as well for downstream use
        t = Token(
            kind=token.kind,
            value=token.value,
            attributes=token.attributes,
            source_span=token.source_span,
            event_id=event_id,
            measure=measure,
            voice=voice,
            hand=hand,
            offset=offset,
        )
        return AnnotatedToken(
            token=t,
            event_id=event_id,
            measure=measure,
            voice=voice,
            hand=hand,
            offset=offset,
            is_compound=is_compound,
        )

    def _tokenize_part(self, part: Node) -> list[AnnotatedToken]:
        out: list[AnnotatedToken] = []
        out.append(self._ann(Token(TokenKind.STRUCT, "PART_START")))
        relations: list[Node] = []
        for child in part.children:
            if child.kind == "measure":
                out.extend(self._tokenize_measure(child))
            elif child.kind in {"meter", "key", "tempo", "clef", "barline"}:
                out.extend(self._tokenize_directive(child))
            elif child.kind in {"tie", "slur", "pedal", "phrase", "hairpin", "dynamic"}:
                relations.append(child)
            else:
                out.extend(self._tokenize_statement(child, measure=None, voice=None, hand=None))
        # Emit relations after measures (ids already known conceptually)
        for rel in relations:
            out.extend(self._tokenize_relation(rel))
        out.append(self._ann(Token(TokenKind.STRUCT, "PART_END")))
        return out

    def _tokenize_measure(self, measure: Node) -> list[AnnotatedToken]:
        out: list[AnnotatedToken] = []
        mnum = measure.number
        out.append(self._ann(Token(TokenKind.STRUCT, "MEASURE_START"), measure=mnum))
        out.append(self._ann(make_special("MEASURE_SEP"), measure=mnum))
        for child in measure.children:
            if child.kind == "staff":
                hand = self._hand_from_staff(child)
                out.append(self._ann(Token(TokenKind.HAND, hand), measure=mnum, hand=hand))
                for voice in child.children:
                    if voice.kind != "voice":
                        continue
                    vname = voice.id or "voice"
                    out.append(
                        self._ann(
                            Token(TokenKind.STRUCT, "VOICE_START"),
                            measure=mnum,
                            voice=vname,
                            hand=hand,
                        )
                    )
                    for ev in voice.children:
                        out.extend(
                            self._tokenize_event(ev, measure=mnum, voice=vname, hand=hand)
                        )
                    out.append(
                        self._ann(
                            Token(TokenKind.STRUCT, "VOICE_END"),
                            measure=mnum,
                            voice=vname,
                            hand=hand,
                        )
                    )
            else:
                out.extend(self._tokenize_statement(child, measure=mnum, voice=None, hand=None))
        out.append(self._ann(Token(TokenKind.STRUCT, "MEASURE_END"), measure=mnum))
        return out

    def _hand_from_staff(self, staff: Node) -> str:
        sid = (staff.id or "").upper()
        if sid in {"RH", "RIGHT", "R"}:
            return "RIGHT"
        if sid in {"LH", "LEFT", "L"}:
            return "LEFT"
        return "UNKNOWN"

    def _tokenize_event(
        self,
        node: Node,
        *,
        measure: int | None,
        voice: str | None,
        hand: str | None,
    ) -> list[AnnotatedToken]:
        kind = node.kind
        eid = node.id
        offset = str(node.get("at")) if node.get("at") is not None else None

        if kind == "note":
            return self._tokenize_note(
                node, measure=measure, voice=voice, hand=hand, offset=offset
            )
        if kind == "rest":
            return self._tokenize_rest(
                node, measure=measure, voice=voice, hand=hand, offset=offset
            )
        if kind == "chord":
            return self._tokenize_chord(
                node, measure=measure, voice=voice, hand=hand, offset=offset
            )
        return self._tokenize_statement(node, measure=measure, voice=voice, hand=hand)

    def _tokenize_note(
        self,
        node: Node,
        *,
        measure: int | None,
        voice: str | None,
        hand: str | None,
        offset: str | None,
    ) -> list[AnnotatedToken]:
        eid = node.id
        pitch = node.get("pitch")
        if isinstance(pitch, Pitch):
            p = pitch
        elif isinstance(pitch, str):
            p = parse_pitch(pitch)
        else:
            p = None

        dur = node.get("dur")
        compound_key = None
        if p is not None and dur is not None:
            compound_key = f"NOTE:{p}:DUR_{duration_token_value(dur)}"

        use_compound = (
            self.config.mode == "compound"
            and compound_key is not None
            and compound_key in self._allowed_compounds
        )

        out: list[AnnotatedToken] = []
        if use_compound:
            assert compound_key is not None
            out.append(
                self._ann(
                    Token(TokenKind.NOTE, compound_key.removeprefix("NOTE:")),
                    event_id=eid,
                    measure=measure,
                    voice=voice,
                    hand=hand,
                    offset=offset,
                    is_compound=True,
                )
            )
            # Attach less common expressive properties separately
            out.extend(
                self._expressive_tokens(
                    node, event_id=eid, measure=measure, voice=voice, hand=hand, offset=offset
                )
            )
            return out

        out.append(
            self._ann(
                Token(TokenKind.EVENT, "NOTE"),
                event_id=eid,
                measure=measure,
                voice=voice,
                hand=hand,
                offset=offset,
            )
        )
        if p is not None:
            pc, acc, octv = normalize_pitch(
                p,
                normalize_enharmonics=self.config.normalize_enharmonics,
                preserve_pitch_spelling=self.config.preserve_pitch_spelling,
            )
            out.append(
                self._ann(
                    Token(TokenKind.PITCH_CLASS, pc),
                    event_id=eid,
                    measure=measure,
                    voice=voice,
                    hand=hand,
                    offset=offset,
                )
            )
            out.append(
                self._ann(
                    Token(TokenKind.ACCIDENTAL, acc),
                    event_id=eid,
                    measure=measure,
                    voice=voice,
                    hand=hand,
                    offset=offset,
                )
            )
            out.append(
                self._ann(
                    Token(TokenKind.OCTAVE, octv),
                    event_id=eid,
                    measure=measure,
                    voice=voice,
                    hand=hand,
                    offset=offset,
                )
            )
        if dur is not None:
            out.append(
                self._ann(
                    Token(TokenKind.DURATION, duration_token_value(dur)),
                    event_id=eid,
                    measure=measure,
                    voice=voice,
                    hand=hand,
                    offset=offset,
                )
            )
        augment = node.get("augment")
        if augment:
            out.append(
                self._ann(
                    Token(TokenKind.DOT_COUNT, int(augment)),
                    event_id=eid,
                    measure=measure,
                    voice=voice,
                    hand=hand,
                    offset=offset,
                )
            )
        out.extend(
            self._expressive_tokens(
                node, event_id=eid, measure=measure, voice=voice, hand=hand, offset=offset
            )
        )
        return out

    def _expressive_tokens(
        self,
        node: Node,
        *,
        event_id: str | None,
        measure: int | None,
        voice: str | None,
        hand: str | None,
        offset: str | None,
    ) -> list[AnnotatedToken]:
        out: list[AnnotatedToken] = []
        arts = node.get("art") or []
        if isinstance(arts, str):
            arts = [arts]
        for a in arts:
            out.append(
                self._ann(
                    Token(TokenKind.ARTICULATION, str(a).upper()),
                    event_id=event_id,
                    measure=measure,
                    voice=voice,
                    hand=hand,
                    offset=offset,
                )
            )
        dyn = node.get("dynamic")
        if dyn is not None:
            out.append(
                self._ann(
                    Token(TokenKind.DYNAMIC, str(dyn).upper()),
                    event_id=event_id,
                    measure=measure,
                    voice=voice,
                    hand=hand,
                    offset=offset,
                )
            )
        finger = node.get("finger")
        if finger is not None:
            out.append(
                self._ann(
                    Token(TokenKind.FINGER, int(finger) if str(finger).isdigit() else str(finger)),
                    event_id=event_id,
                    measure=measure,
                    voice=voice,
                    hand=hand,
                    offset=offset,
                )
            )
        return out

    def _tokenize_rest(
        self,
        node: Node,
        *,
        measure: int | None,
        voice: str | None,
        hand: str | None,
        offset: str | None,
    ) -> list[AnnotatedToken]:
        eid = node.id
        out = [
            self._ann(
                Token(TokenKind.REST),
                event_id=eid,
                measure=measure,
                voice=voice,
                hand=hand,
                offset=offset,
            )
        ]
        dur = node.get("dur")
        if dur is not None:
            out.append(
                self._ann(
                    Token(TokenKind.DURATION, duration_token_value(dur)),
                    event_id=eid,
                    measure=measure,
                    voice=voice,
                    hand=hand,
                    offset=offset,
                )
            )
        return out

    def _tokenize_chord(
        self,
        node: Node,
        *,
        measure: int | None,
        voice: str | None,
        hand: str | None,
        offset: str | None,
    ) -> list[AnnotatedToken]:
        eid = node.id
        out: list[AnnotatedToken] = [
            self._ann(
                Token(TokenKind.STRUCT, "CHORD_START"),
                event_id=eid,
                measure=measure,
                voice=voice,
                hand=hand,
                offset=offset,
            ),
            self._ann(
                Token(TokenKind.EVENT, "CHORD"),
                event_id=eid,
                measure=measure,
                voice=voice,
                hand=hand,
                offset=offset,
            ),
        ]
        dur = node.get("dur")
        if dur is not None:
            out.append(
                self._ann(
                    Token(TokenKind.DURATION, duration_token_value(dur)),
                    event_id=eid,
                    measure=measure,
                    voice=voice,
                    hand=hand,
                    offset=offset,
                )
            )
        for tone in node.children:
            if tone.kind in {"tone", "note"}:
                # Share chord event_id for same-event grouping of tones' pitches
                pitch = tone.get("pitch")
                if isinstance(pitch, str):
                    pitch = parse_pitch(pitch)
                if isinstance(pitch, Pitch):
                    pc, acc, octv = normalize_pitch(
                        pitch,
                        normalize_enharmonics=self.config.normalize_enharmonics,
                        preserve_pitch_spelling=self.config.preserve_pitch_spelling,
                    )
                    tid = tone.id or eid
                    out.append(
                        self._ann(
                            Token(TokenKind.PITCH_CLASS, pc),
                            event_id=eid,
                            measure=measure,
                            voice=voice,
                            hand=hand,
                            offset=offset,
                        )
                    )
                    out.append(
                        self._ann(
                            Token(TokenKind.ACCIDENTAL, acc),
                            event_id=eid,
                            measure=measure,
                            voice=voice,
                            hand=hand,
                            offset=offset,
                        )
                    )
                    out.append(
                        self._ann(
                            Token(TokenKind.OCTAVE, octv),
                            event_id=eid,
                            measure=measure,
                            voice=voice,
                            hand=hand,
                            offset=offset,
                        )
                    )
                    finger = tone.get("finger")
                    if finger is not None:
                        out.append(
                            self._ann(
                                Token(
                                    TokenKind.FINGER,
                                    int(finger) if str(finger).isdigit() else str(finger),
                                ),
                                event_id=eid,
                                measure=measure,
                                voice=voice,
                                hand=hand,
                                offset=offset,
                            )
                        )
        out.append(
            self._ann(
                Token(TokenKind.STRUCT, "CHORD_END"),
                event_id=eid,
                measure=measure,
                voice=voice,
                hand=hand,
                offset=offset,
            )
        )
        return out

    def _tokenize_directive(self, node: Node) -> list[AnnotatedToken]:
        out: list[AnnotatedToken] = []
        if node.kind == "meter":
            beats = node.get("beats")
            unit = node.get("beat-unit") or node.get("beat_unit")
            if beats is not None and unit is not None:
                out.append(
                    self._ann(
                        Token(TokenKind.TIME_SIGNATURE, time_signature_value(beats, unit))
                    )
                )
        elif node.kind == "key":
            tonic = node.get("tonic")
            mode = node.get("mode") or "major"
            if tonic is not None:
                out.append(self._ann(Token(TokenKind.KEY, key_token_value(str(tonic), str(mode)))))
        elif node.kind == "tempo":
            bpm = node.get("bpm")
            if bpm is not None:
                out.append(self._ann(Token(TokenKind.TEMPO_BPM, int(float(bpm)))))
        elif node.kind == "barline":
            style = node.get("style") or node.get("type") or "SINGLE"
            out.append(self._ann(Token(TokenKind.BARLINE, str(style).upper())))
        return out

    def _tokenize_relation(self, node: Node) -> list[AnnotatedToken]:
        out: list[AnnotatedToken] = []
        if node.kind == "tie":
            out.append(self._ann(Token(TokenKind.TIE, "START"), event_id=str(node.get("from"))))
            out.append(self._ann(Token(TokenKind.TIE, "END"), event_id=str(node.get("to"))))
        elif node.kind == "slur":
            out.append(self._ann(Token(TokenKind.SLUR, "START"), event_id=str(node.get("from"))))
            out.append(self._ann(Token(TokenKind.SLUR, "END"), event_id=str(node.get("to"))))
        elif node.kind == "pedal":
            # Pedal span: DOWN at from, UP at to
            out.append(self._ann(Token(TokenKind.PEDAL, "DOWN")))
            out.append(self._ann(Token(TokenKind.PEDAL, "UP")))
        elif node.kind == "dynamic":
            val = node.get("value") or node.id
            if val is not None:
                out.append(self._ann(Token(TokenKind.DYNAMIC, str(val).upper())))
        return out

    def _tokenize_statement(
        self,
        node: Node,
        *,
        measure: int | None,
        voice: str | None,
        hand: str | None,
    ) -> list[AnnotatedToken]:
        if node.kind in {"meter", "key", "tempo", "clef", "barline"}:
            return self._tokenize_directive(node)
        if node.kind in {"tie", "slur", "pedal", "phrase", "hairpin", "dynamic"}:
            return self._tokenize_relation(node)
        if node.kind in {"note", "rest", "chord"}:
            return self._tokenize_event(node, measure=measure, voice=voice, hand=hand)
        return []

    def _iter_compound_candidates(self, document: Document) -> Iterable[tuple[str, Node]]:
        for part in document.score.parts:
            for child in part.children:
                if child.kind != "measure":
                    continue
                for staff in child.children:
                    if staff.kind != "staff":
                        continue
                    for voice in staff.children:
                        if voice.kind != "voice":
                            continue
                        for ev in voice.children:
                            if ev.kind != "note":
                                continue
                            pitch = ev.get("pitch")
                            if isinstance(pitch, str):
                                pitch = parse_pitch(pitch)
                            dur = ev.get("dur")
                            if isinstance(pitch, Pitch) and dur is not None:
                                key = f"NOTE:{pitch}:DUR_{duration_token_value(dur)}"
                                yield key, ev

    def to_canonical_strings(self, tokens: list[Token]) -> list[str]:
        result: list[str] = []
        for t in tokens:
            if t.kind == TokenKind.SPECIAL:
                result.append(f"<{t.value}>")
            else:
                result.append(t.canonical())
        return result
