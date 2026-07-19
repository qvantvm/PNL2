"""Lexer facade — re-exports pnl2 lexer for tests and tooling."""

from __future__ import annotations

from pnl2.lexer import KEYWORDS, Lexer, Token, TokenType

__all__ = ["KEYWORDS", "Lexer", "Token", "TokenType"]
