import pytest

from pnl2vec.pnl import PNLParseError, parse_pnl, validate_pnl


def test_parse_tiny_scale(tiny_scale_text):
    doc = parse_pnl(tiny_scale_text, filename="tiny_scale.pnl")
    assert doc.version == "pnl/2"
    assert len(doc.score.parts) == 1
    measures = [c for c in doc.score.parts[0].children if c.kind == "measure"]
    assert len(measures) == 2


def test_validate_ok(tiny_scale_text):
    doc = parse_pnl(tiny_scale_text)
    issues = validate_pnl(doc, filename="tiny_scale.pnl")
    assert issues == []


def test_malformed_diagnostics():
    with pytest.raises(PNLParseError) as excinfo:
        parse_pnl("pnl/2\nscore {\n  not-a-valid {{{{\n", filename="bad.pnl")
    issue = excinfo.value.issue
    assert issue.filename == "bad.pnl"
    assert issue.message
