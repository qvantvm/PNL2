from pnl2vec.pnl import parse_pnl
from pnl2vec.tokenizer import Tokenizer, TokenizerConfig


def test_atomic_deterministic(tiny_scale_text):
    doc = parse_pnl(tiny_scale_text)
    tok = Tokenizer(TokenizerConfig(mode="atomic"))
    a = tok.to_canonical_strings(tok.tokenize(doc))
    b = tok.to_canonical_strings(tok.tokenize(doc))
    assert a == b
    assert "PITCH_CLASS:C" in a
    assert "DURATION:1/4" in a
    assert "HAND:RIGHT" in a
    assert "HAND:LEFT" in a
    assert "STRUCT:MEASURE_START" in a
    assert a[0] == "<BOS>"
    assert a[-1] == "<EOS>"


def test_atomic_note_components(articulation_text):
    doc = parse_pnl(articulation_text)
    strings = Tokenizer().to_canonical_strings(Tokenizer().tokenize(doc))
    assert "ARTICULATION:STACCATO" in strings
    assert "ARTICULATION:ACCENT" in strings
    assert "DYNAMIC:MF" in strings
    assert "DOT_COUNT:1" in strings
    assert "SLUR:START" in strings


def test_compound_fallback_when_rare(tiny_scale_text):
    doc = parse_pnl(tiny_scale_text)
    tok = Tokenizer(TokenizerConfig(mode="compound", compound_min_frequency=1000))
    tok.fit_compound_frequencies([doc])
    strings = tok.to_canonical_strings(tok.tokenize(doc))
    # Rare compounds fall back to atomic
    assert "EVENT:NOTE" in strings
    assert not any(s.startswith("NOTE:") for s in strings)


def test_compound_when_frequent(tiny_scale_text):
    doc = parse_pnl(tiny_scale_text)
    tok = Tokenizer(TokenizerConfig(mode="compound", compound_min_frequency=1))
    tok.fit_compound_frequencies([doc])
    strings = tok.to_canonical_strings(tok.tokenize(doc))
    assert any(s.startswith("NOTE:") for s in strings)


def test_no_ambiguous_namespaces(tiny_scale_text):
    strings = Tokenizer().to_canonical_strings(Tokenizer().tokenize(parse_pnl(tiny_scale_text)))
    # Pitch and key must differ in namespace
    assert any(s.startswith("PITCH_CLASS:") for s in strings)
    assert any(s.startswith("KEY:") for s in strings)
    assert all(not s.startswith("PITCH_CLASS:") or "MAJOR" not in s for s in strings)
