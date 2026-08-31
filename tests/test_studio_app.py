import os

import pytest

from pnl2.cli import main
from pnl2.studio.sample import BLANK_PNL


def test_cli_studio_missing_qt(monkeypatch, capsys):
    def boom(_path=None):
        print('PyQt6 is required for the studio. Install with: pip install "pnl2[studio]"')
        return 1

    monkeypatch.setattr("pnl2.studio.app.main", boom)
    assert main(["studio"]) == 1
    assert "pnl2[studio]" in capsys.readouterr().out


def test_sample_studio_smoke():
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from pnl2.studio.app import SampleStudio

    app = QApplication.instance() or QApplication([])
    studio = SampleStudio()
    studio.editor.setPlainText(BLANK_PNL)
    assert studio.editor.toPlainText().startswith("pnl/2")
    assert studio._combo.count() >= 0
    assert studio._editor_tabs.count() == 2
    assert studio.meta_title.text() == ""
    assert not studio.dirty
    studio.meta_title.setText("A title")
    assert studio.dirty
    studio.meta_title.setText("")
    assert not studio.dirty
    studio.fit_engraving()
    studio.close()
    del app
