from pnl2vec.pnl.lexer import Lexer, TokenType


def test_lexer_version_and_keywords():
    text = "pnl/2\nscore {\n}\n"
    tokens = Lexer(text).tokenize()
    types = [t.type for t in tokens]
    assert TokenType.VERSION in types
    assert TokenType.KEYWORD in types
    assert TokenType.LBRACE in types
    assert TokenType.RBRACE in types


def test_lexer_pitch_token():
    text = "pnl/2\nscore {\nnote n1 pitch=C#4 dur=1/4\n}\n"
    tokens = Lexer(text).tokenize()
    pitches = [t for t in tokens if t.type == TokenType.PITCH]
    assert any(t.value == "C#4" for t in pitches)
