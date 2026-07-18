"""Lexer for PNL/2."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    VERSION = auto()
    KEYWORD = auto()
    IDENT = auto()
    STRING = auto()
    INTEGER = auto()
    RATIONAL = auto()
    PITCH = auto()
    POSITION = auto()
    RATIO = auto()
    KEY_REF = auto()
    EQUALS = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    NEWLINE = auto()
    EOF = auto()


KEYWORDS = frozenset(
    {
        "score",
        "meta",
        "part",
        "measure",
        "staff",
        "voice",
        "notation",
        "performance",
        "note",
        "tone",
        "chord",
        "rest",
        "space",
        "grace",
        "grace-group",
        "slur",
        "tie",
        "phrase",
        "pedal",
        "pedal-curve",
        "point",
        "beam",
        "hairpin",
        "tempo",
        "tempo-curve",
        "meter",
        "key",
        "clef",
        "roman",
        "chord-symbol",
        "cadence",
        "section",
        "ending",
        "barline",
        "dissonance",
        "prolongation",
        "harmonic-edge",
        "performed-note",
        "tonality",
        "ottava",
        "trill-span",
        "tremolo",
        "dynamic",
    }
)

BLOCK_KEYWORDS = frozenset(
    {
        "score",
        "meta",
        "part",
        "measure",
        "staff",
        "voice",
        "notation",
        "performance",
        "chord",
        "grace-group",
        "pedal-curve",
    }
)

PITCH_LETTERS = frozenset("ABCDEFG")


@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int


class LexerError(Exception):
    def __init__(self, message: str, line: int, column: int) -> None:
        super().__init__(f"{message} at line {line}, column {column}")
        self.line = line
        self.column = column


class Lexer:
    def __init__(self, text: str) -> None:
        # Normalize newlines; keep content otherwise
        self.text = text.replace("\r\n", "\n").replace("\r", "\n")
        self.pos = 0
        self.line = 1
        self.column = 1
        self.length = len(self.text)

    def tokenize(self) -> list[Token]:
        tokens: list[Token] = []
        while True:
            tok = self._next()
            tokens.append(tok)
            if tok.type is TokenType.EOF:
                break
        return tokens

    def _peek(self, n: int = 0) -> str:
        i = self.pos + n
        if i >= self.length:
            return ""
        return self.text[i]

    def _advance(self) -> str:
        ch = self.text[self.pos]
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return ch

    def _skip_spaces(self) -> None:
        while self._peek() in (" ", "\t"):
            self._advance()

    def _next(self) -> Token:
        self._skip_spaces()
        if self.pos >= self.length:
            return Token(TokenType.EOF, "", self.line, self.column)

        line, column = self.line, self.column
        ch = self._peek()

        if ch == "\n":
            self._advance()
            return Token(TokenType.NEWLINE, "\n", line, column)

        if ch == "/" and self._peek(1) == "/":
            while self._peek() and self._peek() != "\n":
                self._advance()
            # Comment consumes to end of line; emit newline if present
            if self._peek() == "\n":
                self._advance()
                return Token(TokenType.NEWLINE, "\n", line, column)
            return self._next()

        if ch == "{":
            self._advance()
            return Token(TokenType.LBRACE, "{", line, column)
        if ch == "}":
            self._advance()
            return Token(TokenType.RBRACE, "}", line, column)
        if ch == "[":
            self._advance()
            return Token(TokenType.LBRACKET, "[", line, column)
        if ch == "]":
            self._advance()
            return Token(TokenType.RBRACKET, "]", line, column)
        if ch == ",":
            self._advance()
            return Token(TokenType.COMMA, ",", line, column)
        if ch == "=":
            self._advance()
            return Token(TokenType.EQUALS, "=", line, column)

        if ch == '"':
            return self._string(line, column)

        # Version header pnl/2
        if self.text.startswith("pnl/2", self.pos) and (
            self.pos + 5 >= self.length or not _is_ident_cont(self._peek(5))
        ):
            for _ in range(5):
                self._advance()
            return Token(TokenType.VERSION, "pnl/2", line, column)

        if ch.isdigit() or (ch == "-" and self._peek(1).isdigit()):
            return self._numberish(line, column)

        if ch.isalpha():
            return self._ident_or_special(line, column)

        raise LexerError(f"unexpected character {ch!r}", line, column)

    def _string(self, line: int, column: int) -> Token:
        self._advance()  # opening quote
        chars: list[str] = []
        while True:
            ch = self._peek()
            if not ch:
                raise LexerError("unterminated string", line, column)
            if ch == '"':
                self._advance()
                break
            if ch == "\\":
                self._advance()
                esc = self._peek()
                if esc not in ('"', "\\"):
                    raise LexerError(f"invalid escape \\{esc}", self.line, self.column)
                chars.append(self._advance())
                continue
            if ch == "\n":
                raise LexerError("unterminated string", line, column)
            chars.append(self._advance())
        return Token(TokenType.STRING, "".join(chars), line, column)

    def _numberish(self, line: int, column: int) -> Token:
        start = self.pos
        if self._peek() == "-":
            self._advance()
        while self._peek().isdigit():
            self._advance()

        # measure:offset or ratio or key-like after number
        if self._peek() == ":":
            # Could be position (1:0, 1:1/4) or later ratio handled elsewhere
            self._advance()
            if not (self._peek().isdigit() or self._peek() == "-"):
                raise LexerError("expected number after ':'", self.line, self.column)
            if self._peek() == "-":
                self._advance()
            while self._peek().isdigit():
                self._advance()
            if self._peek() == "/":
                self._advance()
                if not self._peek().isdigit():
                    raise LexerError("expected denominator", self.line, self.column)
                while self._peek().isdigit():
                    self._advance()
            text = self.text[start : self.pos]
            # Heuristic: if left side looks like measure and right rational → POSITION
            left, right = text.split(":", 1)
            if left.lstrip("-").isdigit():
                return Token(TokenType.POSITION, text, line, column)
            return Token(TokenType.RATIO, text, line, column)

        if self._peek() == "/":
            self._advance()
            if not self._peek().isdigit():
                raise LexerError("expected denominator", self.line, self.column)
            while self._peek().isdigit():
                self._advance()
            return Token(TokenType.RATIONAL, self.text[start : self.pos], line, column)

        text = self.text[start : self.pos]
        return Token(TokenType.INTEGER, text, line, column)

    def _ident_or_special(self, line: int, column: int) -> Token:
        start = self.pos
        # Pitch: letter + optional accidental + octave
        if self._peek() in PITCH_LETTERS and self._looks_like_pitch():
            return self._pitch(line, column)

        while _is_ident_cont(self._peek()):
            self._advance()

        # key-ref C:major after identifier-like pitch letter + accidental?
        text = self.text[start : self.pos]
        if self._peek() == ":" and text[0] in PITCH_LETTERS and _is_pitch_root(text):
            self._advance()
            while _is_ident_cont(self._peek()):
                self._advance()
            return Token(TokenType.KEY_REF, self.text[start : self.pos], line, column)

        # ratio after integer-like? handled in numberish
        # finger-change style already numeric

        if text in KEYWORDS:
            return Token(TokenType.KEYWORD, text, line, column)
        return Token(TokenType.IDENT, text, line, column)

    def _looks_like_pitch(self) -> bool:
        """True if current position matches pitch spelling followed by non-ident or end."""
        i = self.pos
        # letter
        if i >= self.length or self.text[i] not in PITCH_LETTERS:
            return False
        i += 1
        # accidental
        if self.text.startswith("bb", i) or self.text.startswith("##", i):
            i += 2
        elif i < self.length and self.text[i] in ("b", "#"):
            i += 1
        # octave (required signed integer)
        if i >= self.length:
            return False
        if self.text[i] in "+-":
            i += 1
        if i >= self.length or not self.text[i].isdigit():
            return False
        while i < self.length and self.text[i].isdigit():
            i += 1
        # must not continue as longer identifier
        if i < self.length and _is_ident_cont(self.text[i]) and self.text[i] not in ":":
            # allow if next is not letter/digit/_/- that would make it an ident
            if self.text[i].isalnum() or self.text[i] in "_-":
                return False
        return True

    def _pitch(self, line: int, column: int) -> Token:
        start = self.pos
        self._advance()  # letter
        if self.text.startswith("bb", self.pos) or self.text.startswith("##", self.pos):
            self._advance()
            self._advance()
        elif self._peek() in ("b", "#"):
            self._advance()
        if self._peek() in "+-":
            self._advance()
        while self._peek().isdigit():
            self._advance()
        return Token(TokenType.PITCH, self.text[start : self.pos], line, column)


def _is_ident_cont(ch: str) -> bool:
    return bool(ch) and (ch.isalnum() or ch in "_-")


def _is_pitch_root(text: str) -> bool:
    """True for C, F#, Bb, etc. without octave."""
    if not text or text[0] not in PITCH_LETTERS:
        return False
    rest = text[1:]
    return rest in ("", "b", "bb", "#", "##")
