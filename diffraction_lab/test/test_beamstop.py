"""Tests for the DEFPIX beamstop port and its overlay."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from diffraction_lab.beamstop import (
    DEFECTIVE,
    DEFPIX_FRAMES,
    DETECTOR_MASKED,
    HALO,
    SHADED,
    classify_frames,
    classify_pixels,
    defpix_frames,
    detect_beamstop,
    masked_pixels,
    pixel_classes,
    pixel_sums,
    sweep_files,
)

BEAM = (150.0, 120.0)


def _frames(*, shadow: bool = True, n: int = DEFPIX_FRAMES, scale: float = 1.0) -> list[np.ndarray]:
    """Synthetic frames: falling background, a disc-and-arm shadow, gaps, moving spots.

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
            frame[sy - 1: sy + 2, sx - 1: sx + 2] += rng.poisson(500 * scale, (3, 3))
        frame[160:167, :] = -1  # module gap
        frame[118:123, 250:254] = -1  # gap crossing the arm
        frames.append(frame)
    return frames


def test_shades_disc_and_arm() -> None:
    """Blocks over the disc and the arm are shaded; lit background and gaps are not."""
    found = detect_beamstop(_frames(), *BEAM)
    assert found is not None and found.block_px == 8
    mask = found.mask
    yy, xx = np.indices(mask.shape)
    r = np.hypot(xx - BEAM[0], yy - BEAM[1])
    assert mask[r <= 10.0].mean() > 0.95
    arm = (np.abs(yy - BEAM[1]) <= 3) & (xx >= 175)
    assert mask[arm].mean() > 0.9  # all but the gap pixels at x 250–253
    assert not mask[118:123, 250:254].any()  # masked by the detector, not shaded
    assert not mask[160:167, :].any()
    lit = (r >= 30.0) & (np.abs(yy - BEAM[1]) >= 15)
    assert mask[lit].mean() < 0.01
    assert found.bbox[2] >= 290  # the arm reaches the image edge


def test_low_counts_use_bigger_blocks() -> None:
    """Fine-sliced frames (~0.1 count/pixel) are judged in blocks big enough to count."""
    found = detect_beamstop(_frames(scale=0.005), *BEAM)
    assert found is not None and found.block_px > 8
    assert found.mask[int(BEAM[1]) + 5, int(BEAM[0])]


def test_halo_beside_the_stop() -> None:
    """A bright lobe touching the shadow is rejected; a bright patch far away is never halo."""
    rng = np.random.default_rng(3)
    yy, xx = np.indices((240, 300))
    r = np.hypot(xx - BEAM[0], yy - BEAM[1])
    background = 40.0 * np.exp(-r / 150.0) + 5.0
    stop = (r <= 18.0) | ((np.abs(yy - BEAM[1]) <= 6) & (xx >= BEAM[0]))
    lobe = np.hypot(xx - BEAM[0], yy - (BEAM[1] - 24.0)) <= 7.0
    far = np.hypot(xx - 40.0, yy - 40.0) <= 7.0
    background = np.where(stop, 0.03 * background, background)
    background = np.where(lobe | far, 6.0 * background, background)
    frames = [rng.poisson(background).astype(np.int32) for _ in range(DEFPIX_FRAMES)]
    found = detect_beamstop(frames, *BEAM)
    assert found is not None and found.halo is not None
    # Too bright for its ring: shaded blocks (XDS value range) or halo.
    assert (found.mask | found.halo)[lobe].mean() > 0.8
    assert found.halo[lobe].any()
    assert not found.halo[far].any()
    assert not (found.halo & found.mask).any()
    assert found.n_halo == int(found.halo.sum())


def test_no_shadow_returns_none() -> None:
    """A detector with no beamstop gives nothing to tint."""
    assert detect_beamstop(_frames(shadow=False), *BEAM) is None


def test_needs_three_frames() -> None:
    """DEFPIX judges shading only with at least three sampled frames."""
    assert detect_beamstop(_frames(n=2), *BEAM) is None


def test_defpix_frames_match_processinglab() -> None:
    """The frames ProcessingLab sampled on thaumatin (its DEFPIX control image headers)."""
    assert defpix_frames(2, 31, 10) == [2, 5, 8, 12, 15, 18, 21, 25, 28, 31]
    assert defpix_frames(2, 61, 10) == [2, 9, 15, 22, 28, 35, 41, 48, 54, 61]
    assert defpix_frames(1, 4, 10) == [1, 2, 3, 4]
    assert defpix_frames(5, 4, 10) == []


def test_pixel_classes_follow_processinglab_sentinels() -> None:
    """Negative is masked; all-ones is masked at 32 bits but overloaded at 16 bits."""
    sixteen = np.array([0, 5, 65535], dtype=np.uint16)
    valid, over = pixel_classes(sixteen, saturation=65535)
    assert valid.tolist() == [True, True, False] and over.tolist() == [False, False, True]
    thirty_two = np.array([7, np.iinfo(np.uint32).max], dtype=np.uint32)
    valid, over = pixel_classes(thirty_two, saturation=1_000_000)
    assert valid.tolist() == [True, False] and not over.any()
    valid, _ = pixel_classes(np.array([-1, -2, 3], dtype=np.int32))
    assert valid.tolist() == [False, False, True]


def test_background_drops_each_pixels_largest_value() -> None:
    """A spot in one sampled frame does not raise the background."""
    frames = [np.full((2, 2), v, dtype=np.int32) for v in (1, 2, 100)]
    frames[0][0, 0] = -1
    sums = pixel_sums(frames)
    background = sums.background()
    assert background[1, 1] == pytest.approx(1.5)
    assert background[0, 0] == pytest.approx(2.0)  # one value left after dropping 100


def test_statuses_are_processinglab_codes() -> None:
    """Shaded and halo carry ProcessingLab's PixelStatus values."""
    status, block_px, _ = classify_pixels(pixel_sums(_frames()), *BEAM)
    assert block_px == 8
    assert (status == SHADED).any()
    assert set(np.unique(status)) <= {0, 1, 4, 5, 6, 7}
    assert SHADED == 4 and HALO == 7


def test_module_gaps_are_masked() -> None:
    """Gap pixels (−1, or 16-bit 65535 at saturation 65535) are masked; lit pixels are not."""
    found = classify_frames(_frames(), *BEAM)
    masked = masked_pixels(found.status)
    assert masked[160:167, :].all()  # −1 gap row
    assert (found.status[160:167, :] == DETECTOR_MASKED).all()
    assert not masked[50:100, 20:80].any()

    sixteen = [np.clip(f, 0, 65534).astype(np.uint16) for f in _frames()]
    for f in sixteen:
        f[160:167, :] = 65535  # Eiger gap value equal to the saturation level
    found = classify_frames(sixteen, *BEAM, saturation=65535, find_defects=False)
    assert (found.status[160:167, :] == DEFECTIVE).all()  # overloaded in every frame
    assert masked_pixels(found.status)[160:167, :].all()


def test_masked_pixels_need_no_shading_frames() -> None:
    """With one frame only the detector's own masks are known."""
    found = classify_frames(_frames(n=1), *BEAM)
    assert found.n_frames == 1 and found.block_px == 0
    assert masked_pixels(found.status)[160:167, :].all()
    assert not (found.status == SHADED).any()


def test_sweep_files_in_frame_order(tmp_path: Path) -> None:
    """The images of the sweep, sorted, without other sweeps."""
    for i in (3, 1, 2):
        (tmp_path / f"x_1_{i:04d}.cbf").touch()
    (tmp_path / "other_0001.cbf").touch()
    assert [p.name for p in sweep_files(tmp_path / "x_1_0002.cbf")] == [
        "x_1_0001.cbf", "x_1_0002.cbf", "x_1_0003.cbf",
    ]


def test_canvas_and_session_toggle() -> None:
    """The overlay is hidden by default; the toggle is saved with the session."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from diffraction_lab.image_canvas import DiffractionImageCanvas
    from diffraction_lab.session.model import ViewerState
    from diffraction_lab.session.serialize import viewer_from_dict, viewer_to_dict

    QApplication.instance() or QApplication([])
    canvas = DiffractionImageCanvas()
    canvas.set_beamstop(np.eye(8, dtype=bool), halo=np.fliplr(np.eye(8, dtype=bool)))
    assert canvas.has_beamstop() and canvas._show_beamstop is False
    canvas.set_beamstop_visible(True)
    assert canvas._show_beamstop is True
    canvas.set_beamstop(None)
    assert not canvas.has_beamstop()
    canvas.set_masked_pixels(np.eye(8, dtype=bool))
    assert canvas.has_masked_pixels() and canvas._show_masked_pixels is False
    canvas.set_masked_pixels_visible(True)
    assert canvas._show_masked_pixels is True
    canvas.set_masked_pixels(None)
    assert not canvas.has_masked_pixels()
    assert viewer_from_dict(viewer_to_dict(ViewerState(show_beamstop=True))).show_beamstop is True
    assert viewer_from_dict({}).show_beamstop is False
    restored = viewer_from_dict(viewer_to_dict(ViewerState(show_masked_pixels=True)))
    assert restored.show_masked_pixels is True
    assert viewer_from_dict({}).show_masked_pixels is False
