"""Tests for the per-image Reports: the integration table and R_meas by image."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from diffraction_lab.reports import (
    read_image_table,
    rmeas_from_observations,
    unmerged_path,
)

_HEADER = " IMAGE IER  SCALE     NBKG NOVL NEWALD NSTRONG  NREJ  SIGMAB  SIGMAR  MEANIS"


def _row(image: int, strong: int) -> str:
    return f"{image:6d}   0  1.000 1039257 1807  12894 {strong:7d}  5324  0.0000  0.1500     0.9"


def test_image_table_keeps_rows_after_interruptions(tmp_path: Path) -> None:
    """Rows printed after a block's SIGMAR / refine lines still count; the last copy wins."""
    log = tmp_path / "process.log"
    log.write_text("\n".join([
        "junk before", _HEADER, _row(1, 100), _row(2, 110),
        "  SIGMAR: 0.454 deg from 1305 reflections → MOSAIC 0.250→0.300",
        "  POSITIONAL REFINE (frames 1-3): n=10/10 rms=1.0→1.0 (xy 1 px, mean 0,0 px, φ 0.2°)",
        _row(3, 120), "  PROCESS: frames 4-6 … done", "",
        "  26    9912  oI    62.30    72.62   381.22    98.6  97.7 100.7    5.131   I222,I212121",
        _HEADER, _row(3, 130), _row(4, 140), "",
    ]))
    table = read_image_table(log)
    assert table is not None
    assert table.images.tolist() == [1, 2, 3, 4]
    assert table.column("NSTRONG").tolist() == [100, 110, 130, 140]
    assert table.column("MISSING") is None


def test_image_table_missing_or_empty(tmp_path: Path) -> None:
    """No file or no table gives None."""
    assert read_image_table(tmp_path / "nope.log") is None
    (tmp_path / "x.log").write_text("no table here\n")
    assert read_image_table(tmp_path / "x.log") is None


def test_rmeas_by_batch_and_blocks() -> None:
    """√(n/(n−1))·|I − ⟨I⟩| per observation, Σ I per batch; singletons are left out."""
    hkl = np.array([[1, 0, 0], [1, 0, 0], [2, 0, 0], [2, 0, 0], [2, 0, 0], [3, 0, 0]])
    isym = np.array([1, 2, 1, 1, 2, 1])
    batch = np.array([1, 2, 1, 2, 30, 30])
    intensity = np.array([10.0, 20.0, 30.0, 30.0, 60.0, 99.0])
    r = rmeas_from_observations(hkl, isym, batch, intensity)
    assert r.batches.tolist() == [1, 2, 30]
    # (1,0,0): n=2, mean 15, √2·5 each. (2,0,0): n=3, mean 40, √1.5·{10,10,20}.
    s2, s15 = np.sqrt(2.0), np.sqrt(1.5)
    assert r.numerator == pytest.approx([s2 * 5 + s15 * 10, s2 * 5 + s15 * 10, s15 * 20])
    assert r.denominator == pytest.approx([40.0, 50.0, 60.0])  # (3,0,0) is a singleton
    assert r.n_obs.tolist() == [2, 2, 1]
    assert r.overall == pytest.approx(r.numerator.sum() / 150.0)
    first, last, values = r.blocks(25)
    assert first.tolist() == [1, 26] and last.tolist() == [25, 50]
    assert values[0] == pytest.approx((r.numerator[0] + r.numerator[1]) / 90.0)


def test_anomalous_keeps_friedel_mates_apart() -> None:
    """With anomalous on, I+ (odd ISYM) and I− (even) merge separately."""
    hkl = np.array([[1, 0, 0]] * 4)
    isym = np.array([1, 1, 2, 2])
    batch = np.array([1, 1, 1, 1])
    intensity = np.array([10.0, 10.0, 30.0, 30.0])
    assert rmeas_from_observations(hkl, isym, batch, intensity, anomalous=True).overall == pytest.approx(0.0)
    assert rmeas_from_observations(hkl, isym, batch, intensity).overall > 0.0


def test_unmerged_path() -> None:
    """Aimless names the unmerged file after the merged one."""
    assert unmerged_path(Path("/w/aimless.mtz")) == Path("/w/aimless_unmerged.mtz")


def test_reports_widget_loads(tmp_path: Path) -> None:
    """The widget draws a table and reports a missing unmerged file."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from diffraction_lab.reports import ReportsWidget

    QApplication.instance() or QApplication([])
    log = tmp_path / "process.log"
    log.write_text("\n".join([_HEADER, _row(1, 5), _row(2, 7), ""]))
    widget = ReportsWidget()
    widget.load(log, tmp_path / "aimless_unmerged.mtz")
    assert "re-run Aimless" in widget.note.text()
    widget.load(None, None)
    assert widget.note.text() == ""
