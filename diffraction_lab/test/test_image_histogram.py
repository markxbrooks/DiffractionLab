"""Tests for the pixel-intensity histogram."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from diffraction_lab.image_histogram import pixel_histogram, plot_pixel_histogram

CBF_76C1B = Path("/Users/brooks/projects/test_xray_data_cbf/76c1b_0002212_1_0001.cbf")


def test_eiger_like_frame() -> None:
    """uint16 counts: zeros kept, 2^16-1 module gaps excluded, unit then log bins."""
    frame = np.zeros((100, 100), dtype=np.uint16)
    frame[0, :10] = 1
    frame[1, :5] = 2
    frame[2, 0] = 300  # a Bragg-spot pixel
    frame[50, :] = np.iinfo(np.uint16).max  # module gap
    hist = pixel_histogram(frame, overload=65535)
    assert hist.n_total == 10_000 and hist.n_masked == 100 and hist.n_gap == 0
    assert hist.n_valid == 9_900 and hist.n_zero == 9_900 - 16
    assert hist.integer and hist.maximum == 300 and hist.median == 0
    # One bin per count up to 20: 0, 1, 2 hold 9884, 10, 5 pixels.
    assert hist.edges[0] == -0.5 and list(hist.counts[:3]) == [9_884, 10, 5]
    assert np.allclose(np.diff(hist.edges[:22]), 1.0)
    assert hist.counts.sum() == hist.n_valid  # nothing outside the bins
    assert hist.n_overloaded == 0
    assert "1.0% zero" not in hist.statistics and "excluded 100 masked" in hist.notes


def test_pilatus_like_frame() -> None:
    """int32 counts: -1 gaps / -2 bad excluded; at the overload counted; above it excluded."""
    frame = np.ones((10, 10), dtype=np.int32)
    frame[0, :] = -1
    frame[1, 0] = -2
    frame[2, 0] = 244_849  # saturated
    frame[3, 0] = 2**20 - 1  # stuck pixel: above the cutoff
    hist = pixel_histogram(frame, overload=244_849)
    assert hist.n_gap == 11 and hist.n_masked == 0 and hist.n_above_cutoff == 1
    assert hist.n_valid == 88 and hist.n_overloaded == 1 and hist.maximum == 244_849
    assert hist.counts.sum() == 88 and hist.edges[-1] == 244_850
    assert "1 overloaded" in hist.notes and "11 gap/bad, 1 above cutoff" in hist.notes
    # Without an overload nothing is cut.
    assert pixel_histogram(frame).maximum == 2**20 - 1


def test_float_frame_and_errors() -> None:
    frame = np.linspace(0.0, 1.0, 1000).reshape(10, 100)
    frame[0, 0] = np.nan
    hist = pixel_histogram(frame, float_bins=10)
    assert not hist.integer and len(hist.counts) == 10 and hist.n_masked == 1
    with pytest.raises(ValueError, match="no valid pixels"):
        pixel_histogram(np.full((3, 3), -1, dtype=np.int32))


def test_plot() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    frame = np.zeros((50, 50), dtype=np.uint16)
    frame[0, :] = np.arange(50) * 10
    ax = Figure().add_subplot(111)
    plot_pixel_histogram(ax, pixel_histogram(frame, overload=400.0), "test")
    assert ax.get_yscale() == "log" and ax.get_xscale() == "symlog"
    assert ax.get_title().splitlines()[0] == "test"
    assert any("overload" in text.get_text() for text in ax.get_legend().get_texts())


@pytest.mark.skipif(not CBF_76C1B.is_file(), reason="needs the 76c1b CBF")
def test_real_pilatus_frame() -> None:
    fabio = pytest.importorskip("fabio")
    from diffraction_lab.cbf_header import read_cbf_geometry

    image = fabio.open(str(CBF_76C1B))
    geometry = read_cbf_geometry(image.header)
    assert geometry.overload == 244849
    hist = pixel_histogram(image.data, overload=geometry.overload)
    # Pilatus gaps (-1) and bad pixels (-2) are excluded; zeros are kept.
    assert hist.n_gap == 526_101 + 1_816
    assert hist.n_zero == 1_518_647 and hist.median == 1
    # A stuck pixel near the beam centre reads 2^20-1 in every frame.
    assert hist.n_above_cutoff == 1 and hist.maximum < 244849
    assert hist.counts.sum() == hist.n_valid
