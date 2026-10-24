"""Tests for the ProcessingLab tab's helpers (no event loop needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from diffraction_lab.aimless_runner import (  # noqa: E402
    AimlessRunResult,
    ScaleMergeResult,
    parse_aimless_summary,
)
from diffraction_lab.pointless_runner import PointlessRunResult  # noqa: E402
from diffraction_lab.processinglab_panel import (  # noqa: E402
    default_work_parent,
    format_record,
    format_scaling,
    parse_cell_text,
)
from diffraction_lab.processing_sweep import Frames  # noqa: E402
from diffraction_lab.processinglab_run import RunRecord  # noqa: E402
from diffraction_lab.test.test_aimless_runner import AIMLESS_SUMMARY  # noqa: E402


def test_parse_cell_text() -> None:
    assert parse_cell_text("") is None
    assert parse_cell_text("57.9, 57.9, 150.2, 90, 90, 90") == (57.9, 57.9, 150.2, 90.0, 90.0, 90.0)
    with pytest.raises(ValueError, match="six"):
        parse_cell_text("57.9 57.9")
    with pytest.raises(ValueError, match="six"):
        parse_cell_text("a b c d e f")


def test_default_work_parent(tmp_path: Path) -> None:
    image = tmp_path / "data" / "thaumatin_9_1_master.h5"
    image.parent.mkdir()
    image.touch()
    assert default_work_parent(image) == tmp_path / "data" / "thaumatin_9_1-processinglab"


def test_format_record_and_scaling(tmp_path: Path) -> None:
    rec = RunRecord(directory=tmp_path, status="completed", error=None, schema_version=2, image=None,
                    frames=Frames(1, 100), space_group_number=75, space_group_symbol="P 4",
                    cell=(57.938, 57.938, 150.244, 90.0, 90.0, 90.0), wavelength_a=0.9795,
                    distance_mm=199.982, beam_mm=(166.119, 172.925), tilt_twist_mrad=(9.39, -8.43),
                    mosaicity_deg=0.101, refinement={"orientation": "refined", "unit_cell": "fixed"},
                    integrated_reflections=60881, quality_total={"r_meas_pct": 17.9})
    html = format_record(rec)
    assert "75 (P 4)" in html and "57.938 57.938 150.244" in html
    assert "<td>orientation</td>" in html and "<td>unit_cell</td>" in html  # refined / held fixed
    assert "60881 reflections" in html

    pointless = PointlessRunResult(hklin=tmp_path, hklout=tmp_path, log=tmp_path, space_group="P 4 2 2",
                                   reindex_operator="[h,k,l]", confidence=0.0, command="",
                                   solution_kind="point group", laue_confidence=0.999)
    aimless = AimlessRunResult(hklin=tmp_path, hklout=tmp_path, log=tmp_path,
                               summary=parse_aimless_summary(AIMLESS_SUMMARY), command="")
    text = format_scaling(ScaleMergeResult(pointless=pointless, aimless=aimless))
    assert "point group P 4 2 2" in text and "Only the point group" in text
    assert "0.148" in text and "1.45 Å" in text


def test_window_formatters_and_construct(tmp_path: Path) -> None:
    from PySide6.QtWidgets import QApplication

    from diffraction_lab.pointless_runner import TwinningTest
    from diffraction_lab.processinglab_window import ProcessingLabWindow, format_aimless, format_pointless

    _app = QApplication.instance() or QApplication([])
    pointless = PointlessRunResult(hklin=tmp_path, hklout=tmp_path, log=tmp_path, space_group="P 41 21 2",
                                   reindex_operator="[k,h,-l]", confidence=0.93, command="",
                                   solution_kind="space group", laue_confidence=0.99,
                                   twinning=TwinningTest(0.42, 0.25, 0.2, "The L-test suggests that the data "
                                                                          "may be twinned"))
    html = format_pointless(pointless)
    assert "P 41 21 2" in html and "reindexes the data ([k,h,-l])" in html
    assert "color:#b00" in html and "may be twinned" in html
    aimless = format_aimless(parse_aimless_summary(AIMLESS_SUMMARY))
    assert "R<sub>meas</sub>" in aimless and "0.148" in aimless
    assert "1.45 Å" in aimless and "P 4 2 2" in aimless

    # A master whose data files hold 1000 and 400 frames: the viewer counts
    # 1000 (one data file), but the sweep is 1400 frames.
    import h5py
    import numpy as np

    image = tmp_path / "x_master.h5"
    with h5py.File(image, "w") as f:
        f.create_dataset("/entry/data/data_000001", data=np.zeros((1000, 2, 2), dtype=np.uint8))
        f.create_dataset("/entry/data/data_000002", data=np.zeros((400, 2, 2), dtype=np.uint8))
    window = ProcessingLabWindow(None)
    window.set_dataset(image)
    assert (window.sweep_first.value(), window.sweep_last.value()) == (1, 1400)
    assert window.spot_last.value() == 50
    assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == [
        "1. Setup", "2. Find spots", "3. Index", "4. Integrate", "5. Symmetry", "6. Scale"]
    assert not window.index_btn.isEnabled() and not window.pointless_btn.isEnabled()
    # The chosen range is always passed explicitly, even when it is the whole sweep.
    window.sweep_last.setValue(1000)
    assert window._integrate_frames() == Frames(1, 1000)
    window.sweep_last.setValue(1400)
    assert window._integrate_frames() == Frames(1, 1400)
    window.sweep_last.setValue(100)
    window.set_dataset(image)  # same sweep: the edited range is kept
    assert window.sweep_last.value() == 100
