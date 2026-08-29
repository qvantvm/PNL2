from pnl2.studio.sample import BLANK_PNL
from pnl2.studio.worker import render_source


def test_render_source_parse_error():
    result = render_source("not pnl")
    assert result.ok is False
    assert result.svg is None
    assert any(line.startswith("parse error:") for line in result.logs)


def test_render_source_ok_with_mocked_engrave(monkeypatch):
    pages = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><g class="staff"/></svg>'
    ]
    monkeypatch.setattr("pnl2.engraver.engrave_svg", lambda doc, options=None: pages)
    result = render_source(BLANK_PNL)
    assert result.ok is True
    assert result.svg is not None and "<svg" in result.svg
    assert "OK parse" in result.logs
    assert "OK validate" in result.logs
    assert any(line.startswith("engraved ") for line in result.logs)


def test_render_source_engraver_missing(monkeypatch):
    from pnl2.engraver import EngraverError

    def boom(doc, options=None):
        raise EngraverError('Verovio is required. Install with: pip install "pnl2[engrave]"')

    monkeypatch.setattr("pnl2.engraver.engrave_svg", boom)
    result = render_source(BLANK_PNL)
    assert result.ok is False
    assert any("engrave error:" in line for line in result.logs)
