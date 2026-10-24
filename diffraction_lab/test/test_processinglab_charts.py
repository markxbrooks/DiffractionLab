"""Tests for the live ProcessingLab integration charts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from diffraction_lab.processinglab_charts import (  # noqa: E402
    BatchRange,
    IntegrationChartsWidget,
    IntegrationMonitor,
    LogBlockParser,
    ProfileMap,
    ResolutionBins,
    read_log_charts,
)

# One batch of a real ProcessingLab thaumatin log (header, PROFILE_MAP, RESOLUTION_BINS).
BATCH_LOG = Path(__file__).parent / "data" / "processinglab_batch.log"


def _events(lines: list[str]) -> list:
    parser = LogBlockParser()
    return [event for line in lines for event in parser.feed(line)]


def test_parser_real_batch() -> None:
    events = _events(BATCH_LOG.read_text().splitlines())
    assert [type(e) for e in events] == [BatchRange, ProfileMap, ResolutionBins]
    assert events[0] == BatchRange(1, 25)
    profile = events[1]
    assert profile.volume.shape == (9, 15, 15) and profile.scale == 100
    assert profile.phi_offsets == (-4, -3, -2, -1, 0, 1, 2, 3, 4)
    # The peak (100) is the centre of the central slice.
    assert profile.volume.max() == 100 and profile.volume[4, 7, 7] == 100
    bins = events[2]
    assert len(bins.shells) == 9 and bins.columns[0] == "d_limit_A"
    assert bins.column("d_limit_A")[0] == pytest.approx(9.99)
    assert bins.total is not None and bins.total["completeness_pct"] == pytest.approx(8.1)


def test_parser_streams_in_pieces() -> None:
    """Blocks complete only on their END line; ordinary lines yield nothing."""
    parser = LogBlockParser()
    lines = ["noise", "PROFILE_MAP side=2 nphi=2 scale=100", "# phi=-1", "   0   10", "  20  100",
             "# phi=0", "   5    5", "   5    5"]
    assert all(parser.feed(line) == [] for line in lines)
    (profile,) = parser.feed("PROFILE_MAP_END")
    assert profile.volume.tolist() == [[[0, 10], [20, 100]], [[5, 5], [5, 5]]]
    assert parser.feed("RESOLUTION_BINS nshells=1 columns=d_limit_A,rmeas_pct") == []
    assert parser.feed("# comment") == []
    assert parser.feed("  2.00  -99.9") == []
    assert parser.feed("  total  12.5") == []
    (bins,) = parser.feed("RESOLUTION_BINS_END")
    assert np.isnan(bins.column("rmeas_pct")[0])  # XDS's −99.9 "undefined"
    assert bins.total == {"d_limit_A": pytest.approx(np.nan, nan_ok=True), "rmeas_pct": 12.5}


def test_read_log_charts(tmp_path: Path) -> None:
    profile, bins, batch = read_log_charts(BATCH_LOG)
    assert profile is not None and bins is not None and batch == BatchRange(1, 25)
    assert read_log_charts(tmp_path / "missing.log") == (None, None, None)


def test_monitor_signals_and_widget() -> None:
    from PySide6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication([])
    monitor = IntegrationMonitor()
    seen: list[str] = []
    monitor.batchStarted.connect(lambda first, last: seen.append(f"batch {first}-{last}"))
    monitor.profileReady.connect(lambda p: seen.append(f"profile {p.volume.shape[0]}"))
    monitor.binsReady.connect(lambda b: seen.append(f"bins {len(b.shells)}"))
    for line in BATCH_LOG.read_text().splitlines():
        monitor.feed(line)
    assert seen == ["batch 1-25", "profile 9", "bins 9"]

    widget = IntegrationChartsWidget()
    widget.reset()
    assert "waiting" in widget.batch_label.text().lower() or "Integrating" in widget.batch_label.text()
    for line in BATCH_LOG.read_text().splitlines():
        widget.feed(line)
    assert "frames 1–25" in widget.batch_label.text() and "completeness 8.1%" in widget.batch_label.text()
    assert len(widget.profile_canvas.figure.axes) == 9
    assert len(widget.bins_canvas.figure.axes) == 4  # two panels, each with a twin axis
    widget.reset()
    assert widget.load_log(BATCH_LOG)
    assert len(widget.profile_canvas.figure.axes) == 9
