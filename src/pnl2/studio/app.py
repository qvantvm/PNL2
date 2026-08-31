"""Four-pane PyQt6 studio for authoring PNL/2 dataset samples."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from .sample import (
    Sample,
    default_samples_dir,
    list_samples,
    load_sample,
    new_sample,
    save_sample,
)
from .worker import RenderResult, render_source

ORG = "pnl2"
APP = "studio"
DEBOUNCE_MS = 500
INSTALL_STUDIO = 'pip install "pnl2[studio]"'


def main(path: Path | None = None) -> int:
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        print(f"PyQt6 is required for the studio. Install with: {INSTALL_STUDIO}", file=sys.stderr)
        return 1

    app = QApplication.instance() or QApplication(sys.argv)
    app.setOrganizationName(ORG)
    app.setApplicationName(APP)
    window = SampleStudio()
    window.show()
    if path is not None:
        window.open_path(Path(path))
    else:
        window.restore_session()
    return app.exec()


class SampleStudio:
    """Created after PyQt6 is imported so tests can skip the widget class."""

    def __init__(self) -> None:
        from PyQt6.QtCore import QSettings, Qt, QThread, QTimer
        from PyQt6.QtGui import QAction, QCloseEvent, QFont, QKeySequence, QPixmap
        from PyQt6.QtSvgWidgets import QSvgWidget
        from PyQt6.QtWidgets import (
            QComboBox,
            QFileDialog,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QMessageBox,
            QPlainTextEdit,
            QPushButton,
            QScrollArea,
            QSizePolicy,
            QSlider,
            QSplitter,
            QStatusBar,
            QToolBar,
            QVBoxLayout,
            QWidget,
        )

        self._Qt = Qt
        self._QSettings = QSettings
        self._QThread = QThread
        self._QTimer = QTimer
        self._QAction = QAction
        self._QCloseEvent = QCloseEvent
        self._QFont = QFont
        self._QKeySequence = QKeySequence
        self._QPixmap = QPixmap
        self._QSvgWidget = QSvgWidget
        self._QFileDialog = QFileDialog
        self._QMessageBox = QMessageBox
        self._QPlainTextEdit = QPlainTextEdit

        self.window = QMainWindow()
        self.window.setWindowTitle("PNL/2 Sample Studio")
        self.window.resize(1280, 840)
        self.window.closeEvent = self._close_event  # type: ignore[method-assign]

        self.sample = new_sample()
        self._saved_text = self.sample.text
        self._last_svg: str | None = None
        self._job_id = 0
        self._threads: list = []
        self._library_dir: Path | None = None
        self._sample_paths: list[Path] = []
        try:
            from ..engraver import warmup_verovio

            warmup_verovio()
        except Exception:  # noqa: BLE001
            pass

        self.editor = QPlainTextEdit()
        font = QFont("Menlo")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(12)
        self.editor.setFont(font)
        self.editor.setPlainText(self.sample.text)
        self.editor.textChanged.connect(self._on_text_changed)

        self.preview_box = QGroupBox("Live engraving")
        self.svg_widget = QSvgWidget()
        self.svg_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.svg_widget.setStyleSheet("background: white;")
        self.svg_widget.setAutoFillBackground(True)
        svg_scroll = QScrollArea()
        svg_scroll.setWidget(self.svg_widget)
        svg_scroll.setWidgetResizable(True)
        svg_scroll.setStyleSheet("QScrollArea { background: white; border: none; }")
        svg_scroll.viewport().setStyleSheet("background: white;")
        preview_layout = QVBoxLayout(self.preview_box)
        preview_layout.addWidget(svg_scroll)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(font)
        self.log.setMaximumBlockCount(2000)

        self.ref_box = QGroupBox("Reference image")
        self.ref_label = QLabel("No reference image loaded")
        self.ref_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ref_label.setMinimumSize(200, 160)
        self.ref_pix: QPixmap | None = None
        self._zoom = 1.0
        ref_scroll = QScrollArea()
        ref_scroll.setWidget(self.ref_label)
        ref_scroll.setWidgetResizable(True)
        zoom_row = QHBoxLayout()
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(25, 400)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self._on_zoom)
        fit_btn = QPushButton("Fit")
        fit_btn.clicked.connect(self._fit_reference)
        zoom_row.addWidget(QLabel("Zoom"))
        zoom_row.addWidget(self.zoom_slider)
        zoom_row.addWidget(fit_btn)
        ref_layout = QVBoxLayout(self.ref_box)
        ref_layout.addWidget(ref_scroll, 1)
        ref_layout.addLayout(zoom_row)

        editor_box = QGroupBox("PNL/2 script")
        editor_layout = QVBoxLayout(editor_box)
        editor_layout.addWidget(self.editor)

        log_box = QGroupBox("Parser / engraver log")
        log_layout = QVBoxLayout(log_box)
        log_layout.addWidget(self.log)

        vsplit = QSplitter(Qt.Orientation.Vertical)
        htop = QSplitter(Qt.Orientation.Horizontal)
        hbot = QSplitter(Qt.Orientation.Horizontal)
        htop.addWidget(editor_box)
        htop.addWidget(self.preview_box)
        hbot.addWidget(log_box)
        hbot.addWidget(self.ref_box)
        htop.setStretchFactor(0, 1)
        htop.setStretchFactor(1, 1)
        hbot.setStretchFactor(0, 1)
        hbot.setStretchFactor(1, 1)
        vsplit.addWidget(htop)
        vsplit.addWidget(hbot)
        vsplit.setStretchFactor(0, 3)
        vsplit.setStretchFactor(1, 2)
        self._vsplit = vsplit
        self._htop = htop
        self._hbot = hbot

        self.window.setCentralWidget(vsplit)
        self.window.setStatusBar(QStatusBar())
        self._build_menus()
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.window.addToolBar(toolbar)
        for act in (self._act_new, self._act_open, self._act_save):
            toolbar.addAction(act)
        toolbar.addSeparator()
        toolbar.addAction(self._act_prev)
        self._combo = QComboBox()
        self._combo.setMinimumContentsLength(18)
        self._combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._combo.currentIndexChanged.connect(self._on_combo)
        toolbar.addWidget(self._combo)
        toolbar.addAction(self._act_next)
        toolbar.addSeparator()
        for act in (self._act_ref, self._act_render, self._act_export):
            toolbar.addAction(act)

        self._timer = QTimer(self.window)
        self._timer.setSingleShot(True)
        self._timer.setInterval(DEBOUNCE_MS)
        self._timer.timeout.connect(self.render_now)
        self._update_title()

    def _build_menus(self) -> None:
        QAction = self._QAction
        QKeySequence = self._QKeySequence
        bar = self.window.menuBar()
        file_menu = bar.addMenu("&File")
        self._act_new = QAction("&New", self.window)
        self._act_new.setShortcut(QKeySequence.StandardKey.New)
        self._act_new.triggered.connect(self.new_document)
        self._act_open = QAction("&Open Sample…", self.window)
        self._act_open.setShortcut(QKeySequence.StandardKey.Open)
        self._act_open.triggered.connect(self.open_sample_dialog)
        self._act_open_dir = QAction("Open Dataset &Folder…", self.window)
        self._act_open_dir.setShortcut(QKeySequence("Ctrl+Shift+O"))
        self._act_open_dir.triggered.connect(self.open_library_dialog)
        self._act_prev = QAction("&Previous Sample", self.window)
        self._act_prev.setShortcut(QKeySequence("Ctrl+["))
        self._act_prev.triggered.connect(lambda: self.step_sample(-1))
        self._act_next = QAction("&Next Sample", self.window)
        self._act_next.setShortcut(QKeySequence("Ctrl+]"))
        self._act_next.triggered.connect(lambda: self.step_sample(1))
        self._act_save = QAction("&Save", self.window)
        self._act_save.setShortcut(QKeySequence.StandardKey.Save)
        self._act_save.triggered.connect(self.save)
        self._act_save_as = QAction("Save &As…", self.window)
        self._act_save_as.setShortcut(QKeySequence.StandardKey.SaveAs)
        self._act_save_as.triggered.connect(self.save_as)
        self._act_ref = QAction("Load &Reference Image…", self.window)
        self._act_ref.triggered.connect(self.load_reference_dialog)
        self._act_export = QAction("&Export Preview PNG…", self.window)
        self._act_export.triggered.connect(self.export_preview)
        self._act_quit = QAction("&Quit", self.window)
        self._act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        self._act_quit.triggered.connect(self.window.close)
        for act in (
            self._act_new,
            self._act_open,
            self._act_open_dir,
            self._act_save,
            self._act_save_as,
        ):
            file_menu.addAction(act)
        file_menu.addSeparator()
        file_menu.addAction(self._act_ref)
        file_menu.addAction(self._act_export)
        file_menu.addSeparator()
        file_menu.addAction(self._act_quit)

        view_menu = bar.addMenu("&View")
        self._act_render = QAction("&Render", self.window)
        self._act_render.setShortcut(QKeySequence("Ctrl+R"))
        self._act_render.triggered.connect(self.render_now)
        view_menu.addAction(self._act_render)
        view_menu.addSeparator()
        view_menu.addAction(self._act_prev)
        view_menu.addAction(self._act_next)

    def show(self) -> None:
        self.window.show()

    def close(self) -> None:
        self.window.close()

    @property
    def dirty(self) -> bool:
        return self.editor.toPlainText() != self._saved_text

    def new_document(self) -> None:
        if not self._confirm_discard():
            return
        self._apply_sample(new_sample(), saved=True)
        self._append_log("new untitled sample")

    def open_sample_dialog(self) -> None:
        if not self._confirm_discard():
            return
        path, _ = self._QFileDialog.getOpenFileName(
            self.window,
            "Open Sample",
            str(self._start_dir()),
            "PNL/2 samples (*.pnl *.sample.json);;All files (*)",
        )
        if path:
            self.open_path(Path(path))

    def open_library_dialog(self) -> None:
        if not self._confirm_discard():
            return
        path = self._QFileDialog.getExistingDirectory(
            self.window,
            "Open Dataset Folder",
            str(self._start_dir()),
        )
        if path:
            self.open_path(Path(path))

    def open_path(self, path: Path) -> None:
        path = Path(path).expanduser()
        if path.is_dir():
            self._open_library(path)
            return
        try:
            sample = load_sample(path)
        except (OSError, ValueError) as exc:
            self._QMessageBox.warning(self.window, "Open failed", str(exc))
            self._append_log(f"open error: {exc}")
            return
        self._apply_sample(sample, saved=True)
        if sample.pnl_path is not None:
            self._refresh_library(sample.pnl_path.parent, sample.pnl_path)
        self._append_log(f"opened {sample.display_name}")
        if sample.expected_path:
            shown = sample.expected_ref or sample.expected_path.name
            self._append_log(f"reference {shown}")
        elif sample.expected_ref:
            self._append_log(f"reference missing: {sample.expected_ref}")
        self.render_now()

    def step_sample(self, delta: int) -> None:
        if not self._sample_paths:
            return
        current = self._combo.currentIndex()
        nxt = current + delta
        if nxt < 0 or nxt >= len(self._sample_paths):
            return
        if not self._confirm_discard():
            return
        self.open_path(self._sample_paths[nxt])

    def save(self) -> bool:
        if self.sample.pnl_path is None:
            return self.save_as()
        return self._write_sample(self.sample.pnl_path)

    def save_as(self) -> bool:
        path, _ = self._QFileDialog.getSaveFileName(
            self.window,
            "Save Sample",
            str((self.sample.pnl_path or self._start_dir() / "untitled.pnl")),
            "PNL/2 (*.pnl)",
        )
        if not path:
            return False
        return self._write_sample(Path(path))

    def load_reference_dialog(self) -> None:
        path, _ = self._QFileDialog.getOpenFileName(
            self.window,
            "Load Reference Image",
            str(self._start_dir()),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*)",
        )
        if not path:
            return
        self.sample.expected_ref = None
        self._set_reference(Path(path))
        self._append_log(f"reference {Path(path).name}")
        self._update_title()

    def export_preview(self) -> None:
        if not self._last_svg:
            self._QMessageBox.information(self.window, "Export", "Render a score before exporting.")
            return
        start = self._start_dir() / "preview.png"
        if self.sample.pnl_path:
            start = self.sample.pnl_path.with_name(self.sample.pnl_path.stem + "-preview.png")
        path, _ = self._QFileDialog.getSaveFileName(
            self.window,
            "Export Preview PNG",
            str(start),
            "PNG (*.png)",
        )
        if not path:
            return
        try:
            from ..engraver import engrave

            engrave(self.editor.toPlainText(), Path(path), format="png")
            self._append_log(f"exported {Path(path).name}")
        except Exception as exc:  # noqa: BLE001
            self._QMessageBox.warning(self.window, "Export failed", str(exc))
            self._append_log(f"export error: {exc}")

    def render_now(self) -> None:
        self._timer.stop()
        text = self.editor.toPlainText()
        self._job_id += 1
        job_id = self._job_id
        self.window.statusBar().showMessage("Rendering…")

        thread = self._QThread()
        holder = _FnWorker(lambda: render_source(text))
        holder.moveToThread(thread)

        def finish(result: object) -> None:
            thread.quit()
            if job_id == self._job_id and isinstance(result, RenderResult):
                self._on_render(result)

        holder.finished.connect(finish)
        thread.started.connect(holder.run)
        thread.finished.connect(lambda: self._drop_thread(thread))
        self._threads.append((thread, holder))
        thread.start()

    def restore_session(self) -> None:
        settings = self._QSettings(ORG, APP)
        geometry = settings.value("geometry")
        if geometry is not None:
            self.window.restoreGeometry(geometry)
        for key, splitter in (
            ("vsplit", self._vsplit),
            ("htop", self._htop),
            ("hbot", self._hbot),
        ):
            state = settings.value(key)
            if state is not None:
                splitter.restoreState(state)
        last = settings.value("last_sample")
        if last:
            path = Path(str(last))
            if path.is_file():
                self.open_path(path)
                return
        last_lib = settings.value("last_library")
        for candidate in (last_lib, default_samples_dir()):
            if not candidate:
                continue
            directory = Path(str(candidate))
            if directory.is_dir() and list_samples(directory):
                self.open_path(directory)
                return
        self._apply_sample(new_sample(), saved=True)
        self.render_now()

    def _write_sample(self, dest: Path) -> bool:
        self.sample.text = self.editor.toPlainText()
        try:
            self.sample = save_sample(self.sample, dest)
        except OSError as exc:
            self._QMessageBox.warning(self.window, "Save failed", str(exc))
            self._append_log(f"save error: {exc}")
            return False
        self._saved_text = self.sample.text
        self._update_title()
        self._append_log(f"saved {self.sample.display_name}")
        if self.sample.pnl_path is not None:
            self._refresh_library(self.sample.pnl_path.parent, self.sample.pnl_path)
        self._persist_settings()
        return True

    def _apply_sample(self, sample: Sample, *, saved: bool) -> None:
        self.sample = sample
        self.editor.blockSignals(True)
        self.editor.setPlainText(sample.text)
        self.editor.blockSignals(False)
        self._saved_text = sample.text if saved else None
        self._set_reference(sample.expected_path)
        self._update_title()

    def _set_reference(self, path: Path | None) -> None:
        self.sample.expected_path = path
        if path is None or not Path(path).is_file():
            self.ref_pix = None
            self.ref_label.setPixmap(self._QPixmap())
            missing = self.sample.expected_ref
            self.ref_label.setText(
                f"Missing reference:\n{missing}" if missing else "No reference image loaded"
            )
            self.ref_box.setTitle("Reference image")
            return
        pix = self._QPixmap(str(path))
        if pix.isNull():
            self.ref_pix = None
            self.ref_label.setText(f"Could not load {path.name}")
            return
        self.ref_pix = pix
        label = self.sample.expected_ref or path.name
        self.ref_box.setTitle(f"Reference image — {Path(label).name}")
        self.ref_label.setText("")
        self._apply_zoom()

    def _on_zoom(self, value: int) -> None:
        self._zoom = value / 100.0
        self._apply_zoom()

    def _fit_reference(self) -> None:
        if self.ref_pix is None or self.ref_pix.isNull():
            return
        area = self.ref_label.parentWidget()
        if area is None:
            return
        avail = area.size()
        if self.ref_pix.width() <= 0:
            return
        factor = min(avail.width() / self.ref_pix.width(), avail.height() / self.ref_pix.height())
        factor = max(0.25, min(4.0, factor))
        self.zoom_slider.setValue(int(factor * 100))

    def _apply_zoom(self) -> None:
        if self.ref_pix is None or self.ref_pix.isNull():
            return
        scaled = self.ref_pix.scaled(
            int(self.ref_pix.width() * self._zoom),
            int(self.ref_pix.height() * self._zoom),
            self._Qt.AspectRatioMode.KeepAspectRatio,
            self._Qt.TransformationMode.SmoothTransformation,
        )
        self.ref_label.setPixmap(scaled)
        self.ref_label.resize(scaled.size())

    def _on_text_changed(self) -> None:
        self.sample.text = self.editor.toPlainText()
        self._update_title()
        if self._last_svg:
            self.preview_box.setTitle("Live engraving (stale)")
        self._timer.start()

    def _on_render(self, result: RenderResult) -> None:
        for line in result.logs:
            self._append_log(line)
        if result.ok and result.svg:
            self._last_svg = result.svg
            self._show_svg(result.svg)
            self.preview_box.setTitle("Live engraving")
            self.window.statusBar().showMessage("Rendered", 3000)
        else:
            if self._last_svg:
                self.preview_box.setTitle("Live engraving (stale)")
            self.window.statusBar().showMessage("Render failed", 4000)

    def _show_svg(self, svg: str) -> None:
        from PyQt6.QtCore import QByteArray

        from ..engraver import flatten_nested_svgs

        flat = flatten_nested_svgs(svg)
        self.svg_widget.load(QByteArray(flat.encode("utf-8")))
        renderer = self.svg_widget.renderer()
        if renderer is not None:
            size = renderer.defaultSize()
            if size.width() > 0:
                self.svg_widget.setMinimumSize(size)

    def _append_log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{stamp}] {message}")

    def _update_title(self) -> None:
        mark = " *" if self.dirty else ""
        self.window.setWindowTitle(f"PNL/2 Sample Studio — {self.sample.display_name}{mark}")

    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        choice = self._QMessageBox.question(
            self.window,
            "Unsaved changes",
            "Save the current sample before continuing?",
            self._QMessageBox.StandardButton.Save
            | self._QMessageBox.StandardButton.Discard
            | self._QMessageBox.StandardButton.Cancel,
        )
        if choice == self._QMessageBox.StandardButton.Cancel:
            return False
        if choice == self._QMessageBox.StandardButton.Save:
            return self.save()
        return True

    def _close_event(self, event) -> None:
        if not self._confirm_discard():
            event.ignore()
            return
        self._persist_settings()
        event.accept()

    def _persist_settings(self) -> None:
        settings = self._QSettings(ORG, APP)
        settings.setValue("geometry", self.window.saveGeometry())
        settings.setValue("vsplit", self._vsplit.saveState())
        settings.setValue("htop", self._htop.saveState())
        settings.setValue("hbot", self._hbot.saveState())
        if self.sample.pnl_path is not None:
            settings.setValue("last_sample", str(self.sample.pnl_path))
        if self._library_dir is not None:
            settings.setValue("last_library", str(self._library_dir))

    def _start_dir(self) -> Path:
        if self._library_dir is not None and self._library_dir.is_dir():
            return self._library_dir
        if self.sample.pnl_path is not None:
            return self.sample.pnl_path.parent
        return default_samples_dir()

    def _open_library(self, directory: Path, select: Path | None = None) -> None:
        directory = Path(directory).expanduser().resolve()
        paths = list_samples(directory)
        self._refresh_library(directory, select)
        target = None
        if select is not None:
            wanted = Path(select).resolve()
            for path in paths:
                if path == wanted:
                    target = path
                    break
        if target is None and paths:
            target = paths[0]
        if target is None:
            self._append_log(f"no .pnl samples in {directory}")
            return
        self.open_path(target)

    def _refresh_library(self, directory: Path, select: Path | None = None) -> None:
        self._library_dir = directory
        self._sample_paths = list_samples(directory)
        self._combo.blockSignals(True)
        self._combo.clear()
        current = -1
        wanted = select.resolve() if select is not None else None
        for index, path in enumerate(self._sample_paths):
            self._combo.addItem(path.name, str(path))
            if wanted is not None and path == wanted:
                current = index
        if current >= 0:
            self._combo.setCurrentIndex(current)
        self._combo.blockSignals(False)
        self._act_prev.setEnabled(current > 0)
        self._act_next.setEnabled(0 <= current < len(self._sample_paths) - 1)
        n = len(self._sample_paths)
        self.window.statusBar().showMessage(f"{n} sample{'s' if n != 1 else ''} in {directory.name}", 4000)

    def _on_combo(self, index: int) -> None:
        if index < 0 or index >= len(self._sample_paths):
            return
        path = self._sample_paths[index]
        if self.sample.pnl_path is not None and path == self.sample.pnl_path.resolve():
            return
        if not self._confirm_discard():
            if self.sample.pnl_path is not None and self._library_dir is not None:
                self._refresh_library(self._library_dir, self.sample.pnl_path)
            return
        self.open_path(path)

    def _drop_thread(self, thread) -> None:
        self._threads = [(t, w) for t, w in self._threads if t is not thread]
        thread.deleteLater()


class _FnWorker:
    """QObject wrapper created after PyQt6 import."""

    def __init__(self, fn) -> None:
        from PyQt6.QtCore import QObject, pyqtSignal

        class _Worker(QObject):
            finished = pyqtSignal(object)

            def __init__(inner, callback) -> None:
                super().__init__()
                inner._callback = callback

            def run(inner) -> None:
                inner.finished.emit(inner._callback())

        self._impl = _Worker(fn)
        self.finished = self._impl.finished
        self.run = self._impl.run
        self.moveToThread = self._impl.moveToThread
