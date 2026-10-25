"""Tests for beamstop shadow detection and its overlay."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from diffraction_lab.beamstop import (
    bin_size,
    detect_beamstop,
    frame_groups,
    median_frame,
    sibling_frames,
)

BEAM = (150.0, 120.0)


def _frames(*, shadow: bool = True, n: int = 3, scale: float = 1.0) -> list[np.ndarray]:
    """Synthetic frames: falling background, a disc-and-arm shadow, a gap row, spots.

    *scale* multiplies the counts; 0.005 mimics fine-sliced Eiger frames.
    """
    rng = np.random.default_rng(7)
    yy, xx = np.indices((240, 300))
    r = np.hypot(xx - BEAM[0], yy - BEAM[1])
    background = 40.0 * np.exp(-r / 150.0) + 5.0
    if shadow:
        stop = (r <= 18.0) | ((np.abs(yy - BEAM[1]) <= 6) & (xx >= BEAM[0]))
        background = np.where(stop, 0.03 * background, background)
    background = background * scale
    frames = []
    for _ in range(n):
        frame = rng.poisson(background).astype(np.int32)
        for _ in range(40):  # spots that move between frames
            sx, sy = rng.integers(5, 295), rng.integers(5, 235)
            frame[sy - 1: sy + 2, sx - 1: sx + 2] += 500
        frame[160:167, :] = -1  # module gap
        frame[118:123, 250:254] = -1  # gap crossing the arm
        frames.append(frame)
    return frames


def test_detects_disc_and_arm_across_gap() -> None:
    """The disc and the whole arm are shadow; lit background is not."""
    found = detect_beamstop(_frames(), *BEAM)
    assert found is not None
    mask = found.mask
    yy, xx = np.indices(mask.shape)
    r = np.hypot(xx - BEAM[0], yy - BEAM[1])
    disc = r <= 14.0
    assert mask[disc].mean() > 0.95
    arm = (np.abs(yy - BEAM[1]) <= 3) & (xx >= 175)
    assert mask[arm].mean() > 0.9  # including beyond the gap at x 250–253
    lit = (r >= 30.0) & (np.abs(yy - BEAM[1]) >= 15)
    assert mask[lit].mean() < 0.01
    assert found.bbox[2] >= 290  # the arm reaches the image edge


def test_low_count_frames_grouped_and_binned() -> None:
    """Fine-sliced frames (~0.1 count/pixel) still give the shadow, not noise."""
    sweep = _frames(n=60, scale=0.005)
    groups = frame_groups(lambda i: sweep[i], len(sweep), [0, 20, 40], max_frames=20)
    assert len(groups) == 3
    found = detect_beamstop(groups, *BEAM)
    assert found is not None
    yy, xx = np.indices(found.mask.shape)
    r = np.hypot(xx - BEAM[0], yy - BEAM[1])
    assert found.mask[r <= 12.0].mean() > 0.9
    lit = (r >= 40.0) & (np.abs(yy - BEAM[1]) >= 25)
    assert found.mask[lit].mean() < 0.02


def test_bin_size_depends_on_counts() -> None:
    """No binning with a few counts per pixel; coarser bins as counts fall."""
    assert bin_size(2.0) == 1
    assert 1 < bin_size(0.1) < bin_size(0.01) <= 16


def test_frame_groups_stop_at_target() -> None:
    """A group stops adding frames once it reaches the target counts."""
    frames = [np.full((30, 30), 1, dtype=np.uint16) for _ in range(10)]
    frames[0][3, 3] = np.iinfo(np.uint16).max  # Eiger gap value
    groups = frame_groups(lambda i: frames[i], 10, [0, 5], target=3.0, max_frames=8)
    assert [float(np.nanmax(g)) for g in groups] == [3.0, 3.0]
    assert np.isnan(groups[0][3, 3])


def test_no_shadow_returns_none() -> None:
    """A detector with no beamstop gives no shadow."""
    assert detect_beamstop(_frames(shadow=False), *BEAM) is None


def test_median_frame_removes_moving_spots() -> None:
    """Spots on one frame of three vanish from the median; gaps become NaN."""
    a = np.full((4, 4), 10, dtype=np.int32)
    b, c = a.copy(), a.copy()
    b[1, 1] = 900
    c[0, 0] = -1
    med = median_frame([a, b, c])
    assert med[1, 1] == 10
    assert np.isnan(med[0, 0])


def test_sibling_frames_spread_over_sweep(tmp_path: Path) -> None:
    """Up to five images of the sweep, the given one first."""
    for i in range(1, 21):
        (tmp_path / f"x_1_{i:04d}.cbf").touch()
    (tmp_path / "other_0001.cbf").touch()
    picks = sibling_frames(tmp_path / "x_1_0005.cbf", 5)
    assert picks[0].name == "x_1_0005.cbf"
    assert len(picks) == 5 and len(set(picks)) == 5
    assert all(p.name.startswith("x_1_") for p in picks)


def test_canvas_and_session_toggle() -> None:
    """The overlay is hidden by default; the toggle is saved with the session."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from diffraction_lab.image_canvas import DiffractionImageCanvas
    from diffraction_lab.session.model import ViewerState
    from diffraction_lab.session.serialize import viewer_from_dict, viewer_to_dict

    QApplication.instance() or QApplication([])
    canvas = DiffractionImageCanvas()
    canvas.set_beamstop(np.eye(8, dtype=bool))
    assert canvas.has_beamstop() and canvas._show_beamstop is False
    canvas.set_beamstop_visible(True)
    assert canvas._show_beamstop is True
    canvas.set_beamstop(None)
    assert not canvas.has_beamstop()
    assert viewer_from_dict(viewer_to_dict(ViewerState(show_beamstop=True))).show_beamstop is True
    assert viewer_from_dict({}).show_beamstop is False
