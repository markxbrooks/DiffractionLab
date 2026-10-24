"""Tests for the processing windows' common base and engine titles."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from diffraction_lab.processing_window_base import ProcessingEngine, ProcessingWindowBase  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


class _Viewer(QWidget):
    """Stands in for the main window."""

    current_file = "/data/x_master.h5"

    def __init__(self) -> None:
        super().__init__()
        self.overlays: list = []

    def set_indexing_spot_overlay(self, spots, *, solution=None) -> None:
        self.overlays.append(spots)


def test_engine_labels() -> None:
    assert [e.label for e in ProcessingEngine] == [
        "ProcessingEngine.DIALS", "ProcessingEngine.MOSFLM", "ProcessingEngine.PROCESSINGLAB"]


def test_base_title_viewer_and_overlay(qapp: QApplication) -> None:
    viewer = _Viewer()
    window = ProcessingWindowBase(viewer, ProcessingEngine.DIALS)
    assert window.windowTitle() == "ProcessingEngine.DIALS" and not window.isModal()
    window.set_engine(ProcessingEngine.MOSFLM)
    assert window.engine is ProcessingEngine.MOSFLM and window.windowTitle() == "ProcessingEngine.MOSFLM"
    assert window.viewer() is viewer
    assert window.current_image_path() == Path("/data/x_master.h5")
    window.close()
    assert viewer.overlays == [None]  # closing clears the overlay


def test_background_job(qapp: QApplication) -> None:
    window = ProcessingWindowBase(None, ProcessingEngine.PROCESSINGLAB)
    lines: list[str] = []
    results: list = []
    ended: list[bool] = []
    window.append_log = lines.append  # type: ignore[method-assign]
    window.on_job_ended = lambda: ended.append(True)  # type: ignore[method-assign]

    def job(emit):
        emit("working")
        return 42

    assert window.start_job("Running…", job, results.append)
    while window.job_running() or not ended:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()
    assert results == [42] and lines == ["working"] and ended == [True]


def test_window_classes_share_the_base() -> None:
    from diffraction_lab.processing_wizard import ProcessingWizard
    from diffraction_lab.processinglab_window import ProcessingLabWindow

    assert issubclass(ProcessingWizard, ProcessingWindowBase)
    assert issubclass(ProcessingLabWindow, ProcessingWindowBase)
