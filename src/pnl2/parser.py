"""Recursive-descent parser for PNL/2."""

from __future__ import annotations

import re
from typing import Any

from .ast import Document, Node, Pitch, Position, Ratio, Score
from .lexer import BLOCK_KEYWORDS, KEYWORDS, Lexer, Token, TokenType
from .rational import Rational

PITCH_RE = re.compile(r"^([A-G])(bb|b|##|#)?([+-]?\d+)$")


class ParseError(Exception):
    def __init__(self, message: str, token: Token | None = None) -> None:
        if token is not None:
            super().__init__(f"{message} at line {token.line}, column {token.column}")
            self.line = token.line
            self.column = token.column
        else:
            super().__init__(message)
            self.line = None
            self.column = None


class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.i = 0

    @property
    def cur(self) -> Token:
        return self.tokens[self.i]

    def peek(self, n: int = 0) -> Token:
        j = self.i + n
        if j >= len(self.tokens):
            return self.tokens[-1]
        return self.tokens[j]

    def advance(self) -> Token:
        tok = self.cur
        if tok.type is not TokenType.EOF:
            self.i += 1
        return tok

    def expect(self, ttype: TokenType, value: str | None = None) -> Token:
        tok = self.cur
        if tok.type is not ttype or (value is not None and tok.value != value):
            expected = value or ttype.name
            raise ParseError(f"expected {expected}, got {tok.type.name} {tok.value!r}", tok)
        return self.advance()

    def skip_newlines(self) -> None:
        while self.cur.type is TokenType.NEWLINE:
            self.advance()

    def parse_document(self) -> Document:
        self.skip_newlines()
        ver = self.expect(TokenType.VERSION)
        self.skip_newlines()
        score_node = self.parse_block_statement("score")
        self.skip_newlines()
        if self.cur.type is not TokenType.EOF:
            raise ParseError("unexpected tokens after score block", self.cur)
        score = self._score_from_node(score_node)
        return Document(version=ver.value, score=score)

    def _score_from_node(self, node: Node) -> Score:
        meta: dict[str, Any] = {}
        parts: list[Node] = []
        statements: list[Node] = []
        for child in node.children:
            if child.kind == "meta":
                meta.update(child.props)
                for item in child.children:
                    if item.kind == "property" and item.id is not None:
                        meta[item.id] = item.props.get("value")
            elif child.kind == "part":
                parts.append(child)
            else:
                statements.append(child)
        return Score(meta=meta, parts=parts, statements=statements)

    def parse_block_statement(self, expected_kind: str | None = None) -> Node:
        tok = self.cur
        if tok.type is not TokenType.KEYWORD:
            raise ParseError("expected keyword", tok)
        kind = tok.value
        if expected_kind and kind != expected_kind:
            raise ParseError(f"expected {expected_kind}", tok)
        self.advance()

        node_id: str | None = None
        number: int | None = None
        props: dict[str, Any] = {}

        if kind == "measure":
            num_tok = self.expect(TokenType.INTEGER)
            number = int(num_tok.value)
        elif kind in (
            "part",
            "staff",
            "voice",
            "chord",
            "grace-group",
            "pedal-curve",
            "notation",
            "performance",
        ):
            if kind in ("notation", "performance"):
                # optional props then block; optional id not used
                pass
            elif self.cur.type in (TokenType.IDENT, TokenType.KEYWORD):
                # part/staff/voice/chord require identifier (keywords reserved — reject)
                if self.cur.type is TokenType.KEYWORD:
                    raise ParseError(
                        f"reserved keyword {self.cur.value!r} cannot be used as identifier",
                        self.cur,
                    )
                node_id = self.advance().value

        # Inline properties before block
        while self.cur.type in (
            TokenType.IDENT,
            TokenType.KEYWORD,
        ) and self.peek(1).type is TokenType.EQUALS:
            key_tok = self.advance()
            if key_tok.type is TokenType.KEYWORD:
                # property names may reuse words like type= — allow
                pass
            self.expect(TokenType.EQUALS)
            props[key_tok.value] = self.parse_value()

        self.skip_newlines()
        self.expect(TokenType.LBRACE)
        self.skip_newlines()

        children: list[Node] = []
        while self.cur.type is not TokenType.RBRACE:
            if self.cur.type is TokenType.NEWLINE:
                self.advance()
                continue
            if self.cur.type is TokenType.EOF:
                raise ParseError("unterminated block", self.cur)
            children.append(self.parse_statement())
            self.skip_newlines()

        self.expect(TokenType.RBRACE)
        # Optional trailing newline consumed by caller
        return Node(kind=kind, id=node_id, props=props, children=children, number=number)

    def parse_statement(self) -> Node:
        tok = self.cur
        if tok.type is TokenType.KEYWORD and tok.value in BLOCK_KEYWORDS:
            return self.parse_block_statement()

        if tok.type is TokenType.KEYWORD:
            return self.parse_line_event()

        # Bare property assignment inside meta or similar
        if tok.type is TokenType.IDENT and self.peek(1).type is TokenType.EQUALS:
            key = self.advance().value
            self.expect(TokenType.EQUALS)
            value = self.parse_value()
            node = Node(kind="property", id=key, props={"value": value})
            if self.cur.type is TokenType.NEWLINE:
                self.advance()
            return node

        raise ParseError(f"unexpected token {tok.type.name} {tok.value!r}", tok)

    def parse_line_event(self) -> Node:
        kind_tok = self.expect(TokenType.KEYWORD)
        kind = kind_tok.value
        node_id: str | None = None

        # Optional identifier (space has none; clef/meter/key may omit)
        no_id = {"space", "clef", "key", "meter", "barline"}
        optional_id = {
            "tempo",
            "tempo-curve",
            "dynamic",
            "hairpin",
            "pedal",
            "beam",
            "ending",
            "section",
            "ottava",
            "trill-span",
            "tremolo",
        }

        if kind == "space":
            pass
        elif kind in no_id:
            # may still have properties only
            pass
        elif self.cur.type is TokenType.IDENT:
            node_id = self.advance().value
        elif self.cur.type is TokenType.KEYWORD:
            raise ParseError(
                f"reserved keyword {self.cur.value!r} cannot be used as identifier",
                self.cur,
            )
        elif kind not in optional_id and kind not in no_id:
            # analysis/relation events require id
            if kind not in optional_id:
                raise ParseError(f"{kind} requires an identifier", self.cur)

        props = self.parse_property_list()
        if self.cur.type is TokenType.NEWLINE:
            self.advance()
        elif self.cur.type not in (TokenType.RBRACE, TokenType.EOF):
            # allow end without newline before }
            pass
        return Node(kind=kind, id=node_id, props=props)

    def parse_property_list(self) -> dict[str, Any]:
        props: dict[str, Any] = {}
        while True:
            if self.cur.type is TokenType.NEWLINE or self.cur.type in (
                TokenType.RBRACE,
                TokenType.EOF,
                TokenType.LBRACE,
            ):
                break
            if self.cur.type not in (TokenType.IDENT, TokenType.KEYWORD):
                break
            if self.peek(1).type is not TokenType.EQUALS:
                break
            key = self.advance().value
            self.expect(TokenType.EQUALS)
            props[key] = self.parse_value(property_name=key)
        return props

    def parse_value(self, property_name: str | None = None) -> Any:
        tok = self.cur
        if tok.type is TokenType.STRING:
            self.advance()
            return tok.value
        if tok.type is TokenType.INTEGER:
            self.advance()
            return int(tok.value)
        if tok.type is TokenType.RATIONAL:
            self.advance()
            return Rational.from_value(tok.value)
        if tok.type is TokenType.PITCH:
            self.advance()
            return parse_pitch(tok.value)
        if tok.type is TokenType.POSITION:
            self.advance()
            return self._position_or_ratio(tok.value, property_name)
        if tok.type is TokenType.RATIO:
            self.advance()
            a, b = tok.value.split(":", 1)
            return Ratio(int(a), int(b))
        if tok.type is TokenType.KEY_REF:
            self.advance()
            return tok.value
        if tok.type is TokenType.LBRACKET:
            return self.parse_list()
        if tok.type is TokenType.IDENT:
            self.advance()
            if tok.value == "true":
                return True
            if tok.value == "false":
                return False
            return tok.value
        if tok.type is TokenType.KEYWORD:
            self.advance()
            return tok.value
        if tok.type is TokenType.VERSION:
            self.advance()
            return tok.value
        raise ParseError(f"invalid value {tok.value!r}", tok)

    def _position_or_ratio(self, text: str, property_name: str | None) -> Any:
        left_s, right_s = text.split(":", 1)
        left = int(left_s)
        right = Rational.from_value(right_s)
        ratio_props = {
            "tuplet",
            "finger-change",
            "interval",  # legacy 4:3 style discouraged but parseable
        }
        if property_name in ratio_props or (
            property_name
            and property_name.endswith("-change")
            and right.denominator == 1
        ):
            if right.denominator != 1:
                raise ParseError(f"invalid ratio {text}", self.cur)
            return Ratio(left, right.numerator)
        return Position(left, right)

    def parse_list(self) -> list[Any]:
        self.expect(TokenType.LBRACKET)
        items: list[Any] = []
        if self.cur.type is TokenType.RBRACKET:
            self.advance()
            return items
        while True:
            items.append(self.parse_value())
            if self.cur.type is TokenType.COMMA:
                self.advance()
                continue
            break
        self.expect(TokenType.RBRACKET)
        return items


def parse_pitch(text: str) -> Pitch:
    m = PITCH_RE.match(text)
    if not m:
        raise ParseError(f"invalid pitch {text!r}")
    letter, accidental, octave = m.group(1), m.group(2) or "", int(m.group(3))
    return Pitch(letter=letter, accidental=accidental, octave=octave)


def parse(text: str) -> Document:
    tokens = Lexer(text).tokenize()
    return Parser(tokens).parse_document()


def is_reserved(name: str) -> bool:
    return name in KEYWORDS
