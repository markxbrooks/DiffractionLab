"""Common base of DiffractionLab's processing windows.

Two processing windows share it:

- :class:`~diffraction_lab.processing_wizard.ProcessingWizard`: DIALS for
  HDF5 / NeXus masters, MOSFLM for other images
  (:attr:`ProcessingEngine.DIALS` / :attr:`ProcessingEngine.MOSFLM`).
- :class:`~diffraction_lab.processinglab_window.ProcessingLabWindow`:
  ProcessingLab, Pointless and Aimless (:attr:`ProcessingEngine.PROCESSINGLAB`).

:class:`ProcessingWindowBase` holds what they have in common: the engine
(shown as the window title), the main window that owns them, clearing the
spot overlay on close, and running a step in a background thread.
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import QDialog, QMessageBox, QWidget

logger = logging.getLogger(__name__)


class CallWorker(QObject):
    """Run one callable in a thread; ``line`` relays its log output."""

    finished = Signal(object)
    failed = Signal(str)
    line = Signal(str)

    def __init__(self, fn: Callable[[Callable[[str], None]], Any]) -> None:
        super().__init__()
        self._fn = fn

    def run(self) -> None:
        try:
            self.finished.emit(self._fn(self.line.emit))
        except Exception as ex:  # noqa: BLE001 - surfaced by the window
            logger.exception("Processing job failed")
            self.failed.emit(str(ex))


class ProcessingEngine(Enum):
    """The program a processing window drives."""

    DIALS = "dials"
    MOSFLM = "mosflm"
    PROCESSINGLAB = "processinglab"

    @property
    def label(self) -> str:
        """``"ProcessingEngine.DIALS"``, …: the window title."""
        return f"{type(self).__name__}.{self.name}"


class ProcessingWindowBase(QDialog):
    """Base of the processing windows: engine title, viewer, background jobs.

    Subclasses pass their engine (and may change it with :meth:`set_engine`)
    and override :meth:`report_status` / :meth:`append_log` to show job
    progress.
    """

    def __init__(self, parent: QWidget | None, engine: ProcessingEngine, *,
                 minimum_size: tuple[int, int] = (780, 560)) -> None:
        super().__init__(parent)
        self.setModal(False)
        self.setMinimumSize(*minimum_size)
        self._engine = engine
        self._job_thread: QThread | None = None
        self._job_worker: CallWorker | None = None
        self.setWindowTitle(engine.label)

    # ----- engine

    @property
    def engine(self) -> ProcessingEngine:
        """The engine this window drives."""
        return self._engine

    def set_engine(self, engine: ProcessingEngine) -> None:
        """Change the engine (e.g. DIALS for an HDF5 master, MOSFLM otherwise) and the title."""
        self._engine = engine
        self.setWindowTitle(engine.label)

    # ----- the main window

    def viewer(self) -> QWidget | None:
        """The DiffractionLab main window that owns this one (it draws the spot overlay)."""
        widget = self.parentWidget()
        while widget is not None:
            if hasattr(widget, "set_indexing_spot_overlay"):
                return widget
            widget = widget.parentWidget()
        return None

    def current_image_path(self) -> Path | None:
        """The image (or HDF5 master) open in the main window."""
        current = getattr(self.viewer() or self.parentWidget(), "current_file", None)
        return Path(current) if current else None

    def clear_spot_overlay(self) -> None:
        """Remove spot markers from the main window's image."""
        viewer = self.viewer()
        if viewer is not None:
            viewer.set_indexing_spot_overlay(None)

    def closeEvent(self, event) -> None:  # noqa: N802
        """Clear the spot overlay when the window closes."""
        self.clear_spot_overlay()
        super().closeEvent(event)

    # ----- background jobs

    def job_running(self) -> bool:
        """Whether a :meth:`start_job` job is still running."""
        return self._job_thread is not None and self._job_thread.isRunning()

    def start_job(self, message: str, fn: Callable[[Callable[[str], None]], Any],
                  on_finished: Callable[[Any], None], *,
                  on_line: Callable[[str], None] | None = None) -> bool:
        """Run ``fn(emit_line)`` in a background thread.

        :param message: Status text while it runs.
        :param fn: The job; it may call ``emit_line(text)`` with progress, and
            its return value goes to *on_finished*.
        :param on_finished: Called (in the GUI thread) with the result.
        :param on_line: Also called (in the GUI thread) with every line.
        :returns: False if another job is still running (nothing started).
        """
        if self.job_running():
            QMessageBox.information(self, self.engine.label, "A step is already running.")
            return False
        self.report_status(message)
        self._job_thread = QThread(self)
        self._job_worker = CallWorker(fn)
        self._job_worker.moveToThread(self._job_thread)
        self._job_thread.started.connect(self._job_worker.run)
        self._job_worker.line.connect(self.append_log)
        if on_line is not None:
            self._job_worker.line.connect(on_line)
        self._job_worker.finished.connect(on_finished)
        self._job_worker.failed.connect(self.on_job_failed)
        self._job_worker.finished.connect(self._job_thread.quit)
        self._job_worker.failed.connect(self._job_thread.quit)
        self._job_thread.finished.connect(self._job_worker.deleteLater)
        self._job_thread.finished.connect(self.on_job_ended)
        self._job_thread.start()
        return True

    def on_job_failed(self, message: str) -> None:
        """A job raised: report it."""
        self.report_status(f"Failed: {message}")
        self.append_log(f"FAILED: {message}")
        QMessageBox.warning(self, self.engine.label, message)

    def on_job_ended(self) -> None:
        """A job's thread finished (after success or failure); refresh controls."""

    def report_status(self, text: str) -> None:
        """Show a one-line status (override; the default logs it)."""
        logger.info("%s: %s", self.engine.label, text)

    def append_log(self, text: str) -> None:
        """Add a line of job output (override; the default logs it)."""
        logger.debug("%s: %s", self.engine.label, text)
