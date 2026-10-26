"""Beamstop shadow and beam halo, as ProcessingLab's DEFPIX finds them.

The viewer tints exactly the pixels ProcessingLab leaves out of integration
for the beamstop, so a wrong beam centre, axis convention or threshold shows
up as a tint that misses the shadow on the image. This module is a port of
ProcessingLab's ``run_defpix`` (``src/diffraction/Defpix.cpp``); keep the two
in step.

DEFPIX samples :data:`DEFPIX_FRAMES` frames spread over the sweep
(:func:`defpix_frames`). Each pixel's background is its mean over them without
its largest value, so a spot crossing it in one sampled frame does not count.
Square blocks, sized so a typical block holds :data:`SHADOW_MIN_COUNTS`
counts, are compared with the median block of their resolution ring, as the
XDS ``VALUE_RANGE_FOR_TRUSTED_DETECTOR_PIXELS`` test: a block outside
:data:`VALUE_RANGE_LOW`–:data:`VALUE_RANGE_HIGH` (× 10⁻⁴ of its ring) is
shaded, wherever it lies. A ring wholly in shadow is judged against the
brightest ring median up to 64 px further out.

The beam halo, where the unblocked beam flares beside the stop, is found by
:func:`find_beam_halo`: binned, smoothed counts above :data:`HALO_FACTOR` ×
their ring's normal level by :data:`HALO_SIGNIFICANCE` σ, in regions that
touch the beam or the shadow.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)

# ProcessingLab ``DefpixSettings`` defaults.
#: Frames sampled across the sweep (PROCESS; FINDSPOTS samples 5).
DEFPIX_FRAMES = 10
#: Shading, hot and defective pixels need at least this many sampled frames.
MIN_DETECT_FRAMES = 3
#: XDS ``VALUE_RANGE_FOR_TRUSTED_DETECTOR_PIXELS``: block ÷ ring × 10⁴.
VALUE_RANGE_LOW = 7000.0
VALUE_RANGE_HIGH = 30000.0
#: A block is judged only with this many counts over the sampled frames.
SHADOW_MIN_COUNTS = 400.0
#: Block edge limits (pixels).
MIN_BLOCK_PX = 8
MAX_BLOCK_PX = 32
#: A ring wholly in shadow is compared with rings up to this far out (pixels).
SHADED_RING_REACH_PX = 64
#: Hot: every sampled frame at least this many counts and this × its ring.
HOT_MIN_COUNTS = 10.0
HOT_FACTOR = 20.0
#: Halo thresholds (``halo_*`` in ``DefpixSettings``).
HALO_FACTOR = 2.0
HALO_SIGNIFICANCE = 5.0
HALO_WINDOW_PX = 100
HALO_SMOOTH_PX = 3.0
HALO_KERNEL_COUNTS = 100.0
HALO_MAX_BIN_PX = 16
HALO_BEAM_PX = 15.0
HALO_REACH_PX = 18.0

# ProcessingLab ``PixelStatus`` values.
TRUSTED = 0
DETECTOR_MASKED = 1
SHADED = 4
DEFECTIVE = 5
HOT = 6
HALO = 7


@dataclass(frozen=True)
class Beamstop:
    """DEFPIX's shaded and halo pixels (image-pixel coordinates)."""

    #: ``True`` on shaded pixels (DEFPIX ``Shaded``), the shape of the image.
    mask: np.ndarray
    #: Shaded pixel count.
    n_pixels: int
    #: Bounding box ``(x0, y0, x1, y1)`` of shaded and halo pixels, inclusive.
    bbox: tuple[int, int, int, int]
    #: ``True`` on halo pixels (DEFPIX ``Halo``), or ``None``.
    halo: np.ndarray | None = None
    #: Halo pixel count.
    n_halo: int = 0
    #: Edge of the blocks the shadow test judged (pixels).
    block_px: int = 0
    #: Farthest halo pixel from the beam among regions at the beam (pixels).
    halo_radius_px: float = 0.0


@dataclass(frozen=True)
class PixelSums:
    """Per-pixel statistics over the sampled frames (ProcessingLab ``PixelSums``)."""

    sum: np.ndarray
    max: np.ndarray
    min: np.ndarray
    n_valid: np.ndarray
    n_overloaded: np.ndarray
    n_frames: int

    def background(self) -> np.ndarray:
        """Return each pixel's mean per frame without its largest value; −1 if none.

        In float32, as ProcessingLab: block means are quantised, so ties at the
        thresholds are exact and the rounding must match.

        :returns: Float32 image.
        """
        n = self.n_valid
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(n > 1, (self.sum - self.max) / np.maximum(n - 1, 1).astype(np.float32), self.sum)
        return np.where(n == 0, np.float32(-1.0), out).astype(np.float32)


def pixel_classes(frame: np.ndarray, saturation: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return which pixels hold counts and which are overloaded.

    As ProcessingLab's readers: negative values (Pilatus gaps, bad pixels) are
    masked. An unsigned all-ones value (DECTRIS "no valid value") is masked
    when it is 32-bit or above *saturation*; otherwise a gap and a saturated
    pixel are the same value, and it is overloaded. Values at or above
    *saturation* are overloaded.

    :param frame: Raw detector frame.
    :param saturation: The detector's overload level, or ``None`` if unknown.
    :returns: ``(valid, overloaded)`` boolean images.
    """
    data = np.asarray(frame)
    masked = ~np.isfinite(data) | (data < 0)
    if np.issubdtype(data.dtype, np.unsignedinteger):
        all_ones = np.iinfo(data.dtype).max
        if data.dtype.itemsize >= 4 or saturation is None or saturation < all_ones:
            masked |= data == all_ones
    overloaded = ~masked & (data >= saturation) if saturation else np.zeros(data.shape, dtype=bool)
    return ~masked & ~overloaded, overloaded


def pixel_sums(frames: Sequence[np.ndarray], saturation: float | None = None) -> PixelSums:
    """Accumulate :class:`PixelSums` over *frames*, one frame at a time.

    :param frames: Raw frames of one shape.
    :param saturation: The detector's overload level, or ``None`` if unknown.
    :returns: The statistics.
    :raises ValueError: If *frames* is empty or the frames differ in shape.
    """
    acc: dict[str, np.ndarray] | None = None
    n_frames = 0
    for frame in frames:
        data = np.asarray(frame)
        valid, overloaded = pixel_classes(data, saturation)
        values = np.where(valid, data, 0).astype(np.float32)
        if acc is None:
            acc = {
                "sum": np.zeros(data.shape, dtype=np.float32),
                "max": np.full(data.shape, -np.inf, dtype=np.float32),
                "min": np.full(data.shape, np.inf, dtype=np.float32),
                "n_valid": np.zeros(data.shape, dtype=np.int32),
                "n_overloaded": np.zeros(data.shape, dtype=np.int32),
            }
        elif data.shape != acc["sum"].shape:
            raise ValueError(f"Frame {n_frames} is {data.shape}, not {acc['sum'].shape}")
        acc["sum"] += values
        acc["max"] = np.where(valid, np.maximum(acc["max"], values), acc["max"])
        acc["min"] = np.where(valid, np.minimum(acc["min"], values), acc["min"])
        acc["n_valid"] += valid
        acc["n_overloaded"] += overloaded
        n_frames += 1
    if acc is None:
        raise ValueError("No frames to sample")
    return PixelSums(n_frames=n_frames, **acc)


def defpix_frames(first: int, last: int, n: int = DEFPIX_FRAMES) -> list[int]:
    """Return *n* frames spread evenly over ``first``…``last`` (all if fewer).

    :param first: First frame.
    :param last: Last frame.
    :param n: Frames wanted.
    :returns: Frame numbers, ascending, both ends included.
    """
    total = last - first + 1
    if total <= 0 or n <= 0:
        return []
    if n >= total:
        return list(range(first, last + 1))
    picks: list[int] = []
    for k in range(n):
        # C++ std::lround: halves away from zero.
        f = first + int(math.floor(k * (total - 1) / (n - 1) + 0.5))
        if not picks or picks[-1] != f:
            picks.append(f)
    return picks


def _cpp_median(values: np.ndarray) -> float:
    """``std::nth_element`` at ``size / 2``: the upper median; 0 if empty."""
    if values.size == 0:
        return 0.0
    k = values.size // 2
    return float(np.partition(values, k)[k])


def _cpp_percentile(values: np.ndarray, p: float) -> float:
    """ProcessingLab's ``percentile``: linear, ``v[lo] + f × (v[hi] − v[lo])``.

    ``numpy.percentile`` interpolates from the upper value above the midpoint,
    which can differ in the last bit.
    """
    v = np.sort(values)
    pos = p / 100.0 * (v.size - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, v.size - 1)
    return float(v[lo] + (pos - lo) * (v[hi] - v[lo]))


def _sequential_mean(values: np.ndarray) -> float:
    """Mean with a left-to-right double sum, as ProcessingLab's loops."""
    return float(np.cumsum(values, dtype=np.float64)[-1] / values.size)


def _reflect(i: np.ndarray, n: int) -> np.ndarray:
    """SciPy's ``reflect`` boundary index (d c b a | a b c d | d c b a)."""
    if n == 1:
        return np.zeros_like(i)
    period = 2 * n
    i = np.mod(i, period)
    return np.where(i < n, i, period - i - 1)


def _block_view(image: np.ndarray, size: int, fill: float) -> np.ndarray:
    """Pad *image* to whole blocks with *fill* and view it as ``(bh, size, bw, size)``."""
    h, w = image.shape
    padded = np.full((-(-h // size) * size, -(-w // size) * size), fill, dtype=image.dtype)
    padded[:h, :w] = image
    return padded.reshape(padded.shape[0] // size, size, padded.shape[1] // size, size)


def _block_sums(blocks: np.ndarray) -> np.ndarray:
    """Sum each block of a ``(bh, size, bw, size)`` view in row-major order, in double.

    The same order as ProcessingLab's loops (NumPy's pairwise ``sum`` can
    differ in the last bit, which flips exact ties).
    """
    _, size, _, _ = blocks.shape
    total = np.zeros((blocks.shape[0], blocks.shape[2]))
    for r in range(size):
        for c in range(size):
            total += blocks[:, r, :, c]
    return total


def find_beam_halo(
    counts: np.ndarray,
    usable: np.ndarray,
    shaded: np.ndarray,
    beam_col: float,
    beam_row: float,
    row_scale: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Find the beam halo (ProcessingLab ``find_beam_halo``).

    Bin the counts until the smoothing kernel holds :data:`HALO_KERNEL_COUNTS`,
    smooth (NaN-aware Gaussian), and take each ring's normal level as the
    median of its unshaded bins, or the lowest such median within
    :data:`HALO_WINDOW_PX` further out. Halo bins are above :data:`HALO_FACTOR`
    × that level with an excess over :data:`HALO_SIGNIFICANCE` σ, in
    4-connected regions within :data:`HALO_BEAM_PX` of the beam or
    :data:`HALO_REACH_PX` of a shaded bin.

    :param counts: Spot-free background counts summed over the sampled
        frames; < 0 if unknown.
    :param usable: Pixels holding a usable value.
    :param shaded: Shaded pixels.
    :param beam_col: Beam position along the fast axis (pixels).
    :param beam_row: Beam position along the slow axis (pixels).
    :param row_scale: Slow pixel size over fast pixel size.
    :returns: ``(halo, radius_px)``: halo pixels, and the farthest of those in
        regions at the beam (0 if none).
    """
    height, width = counts.shape
    ok = usable & (counts >= 0)
    empty = np.zeros(counts.shape, dtype=bool)
    if not ok.any():
        return empty, 0.0
    level = _sequential_mean(counts[ok])
    sigma = HALO_SMOOTH_PX
    needed = math.sqrt(HALO_KERNEL_COUNTS / (4.0 * math.pi * max(level, 1e-9)))
    size = 1 if needed <= sigma else min(HALO_MAX_BIN_PX, math.ceil(needed))
    kernel_px = 4.0 * math.pi * max(sigma, float(size)) ** 2

    blocks = _block_view(np.where(ok, counts, 0.0).astype(np.float64), size, 0.0)
    n_ok = _block_view(ok, size, False).sum(axis=(1, 3))
    n_shaded = _block_view(shaded, size, False).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        img = np.where(2 * n_ok >= size * size, _block_sums(blocks) / n_ok, np.nan)
    shadow = 2 * n_shaded > size * size
    valid = np.isfinite(img)
    s = max(1.0, sigma / size)
    num = ndimage.gaussian_filter(np.where(valid, img, 0.0), s)
    den = ndimage.gaussian_filter(valid.astype(np.float64), s)
    with np.errstate(invalid="ignore", divide="ignore"):
        img = np.where(valid, num / den, np.nan)

    bh, bw = img.shape
    by, bx = np.indices((bh, bw))
    dc = bx * size + 0.5 * (size - 1) - beam_col
    dr = (by * size + 0.5 * (size - 1) - beam_row) * row_scale
    radius = np.hypot(dc, dr) / size
    ring = radius.astype(np.int64)
    n_rings = int(ring.max()) + 1
    ring_median = np.full(n_rings, np.inf)
    use = np.isfinite(img) & ~shadow
    for k in np.unique(ring[use]):
        ring_median[k] = _cpp_percentile(img[use & (ring == k)], 50.0)
    window = max(2, HALO_WINDOW_PX // size)
    lo = window // 2 - (window + 1) // 2
    ks = np.arange(n_rings)[:, None] + lo + np.arange(window + 1)[None, :]
    normal = ring_median[_reflect(ks, n_rings)].min(axis=1)
    normal[~np.isfinite(normal)] = np.nan
    nl = normal[ring]

    with np.errstate(invalid="ignore"):
        excess = (img - nl) * kernel_px
        noise = np.sqrt(np.maximum(nl, 0.0) * kernel_px)
        bright = (
            np.isfinite(img) & np.isfinite(nl) & ~shadow
            & (img > HALO_FACTOR * nl)
            & (excess > HALO_SIGNIFICANCE * np.maximum(noise, 1.0))
        )
    labels, n = ndimage.label(bright)
    if n == 0:
        return empty, 0.0
    reach = max(2, math.ceil(HALO_REACH_PX / size))
    near_shadow = ndimage.binary_dilation(shadow, iterations=reach) if shadow.any() else shadow
    at_beam_ids = np.unique(labels[bright & (radius <= max(1.5, HALO_BEAM_PX / size))])
    by_shadow_ids = np.unique(labels[bright & near_shadow])
    keep = np.zeros(n + 1, dtype=np.uint8)
    keep[by_shadow_ids] = 1
    keep[at_beam_ids] = 2
    keep[0] = 0
    bin_keep = keep[labels]

    rows = np.minimum(np.arange(height) // size, bh - 1)
    cols = np.minimum(np.arange(width) // size, bw - 1)
    pixel_keep = bin_keep[np.ix_(rows, cols)]
    halo = usable & (pixel_keep > 0)
    radius_px = 0.0
    at_beam = usable & (pixel_keep == 2)
    if at_beam.any():
        rr, cc = np.nonzero(at_beam)
        radius_px = float(np.hypot(cc - beam_col, (rr - beam_row) * row_scale).max())
    return halo, radius_px


def classify_pixels(
    sums: PixelSums,
    beam_col: float,
    beam_row: float,
    *,
    row_scale: float = 1.0,
    find_defects: bool = True,
) -> tuple[np.ndarray, int, float]:
    """Classify every pixel as ProcessingLab's ``run_defpix`` does.

    No ``UNTRUSTED`` regions or excluded resolution ranges (the viewer has
    none).

    :param sums: Statistics over the sampled frames.
    :param beam_col: Beam position along the fast axis (pixels).
    :param beam_row: Beam position along the slow axis (pixels).
    :param row_scale: Slow pixel size over fast pixel size.
    :param find_defects: Look for defective and hot pixels (PROCESS skips
        them when the sampled frames span under 5°).
    :returns: ``(status, block_px, halo_radius_px)``: PixelStatus per pixel,
        the shadow block edge (0 if not run) and the halo's radius.
    """
    nf = sums.n_frames
    h, w = sums.sum.shape
    n_valid, n_over = sums.n_valid, sums.n_overloaded
    background = sums.background()
    status = np.full((h, w), TRUSTED, dtype=np.uint8)
    status[(n_valid == 0) & (n_over == 0)] = DETECTOR_MASKED
    detect = nf >= MIN_DETECT_FRAMES
    # Overloaded in every sampled frame: a gap or dead pixel at the saturation
    # value (16-bit Eiger gaps read 65535), rejected even without the search.
    if detect:
        status[(status == TRUSTED) & (n_over == nf)] = DEFECTIVE
    if find_defects and detect:
        defective = (status == TRUSTED) & (n_over >= 2) & ((2 * n_over >= nf) | (sums.max < HOT_MIN_COUNTS))
        status[defective] = DEFECTIVE
    status[(status == TRUSTED) & (n_valid == 0)] = DETECTOR_MASKED  # overloaded throughout
    if not detect:
        return status, 0, 0.0

    usable = (status == TRUSTED) & (n_valid >= 2)
    if not usable.any():
        return status, 0, 0.0
    mean_b = _sequential_mean(background[usable])
    s = int(np.clip(math.ceil(math.sqrt(SHADOW_MIN_COUNTS / max(mean_b * nf, 1e-6))), MIN_BLOCK_PX, MAX_BLOCK_PX))

    rr, cc = np.indices((h, w))
    radius_px = np.hypot(cc - beam_col, (rr - beam_row) * row_scale)
    n_rings = int(math.hypot(w, h) / s) + 2
    ring_px = np.minimum((radius_px / s).astype(np.int64), n_rings - 1)
    ring_n = np.bincount(ring_px[usable], minlength=n_rings)
    ring_sum = np.bincount(ring_px[usable], weights=background[usable], minlength=n_rings)
    ring_mean = np.where(ring_n > 0, ring_sum / np.maximum(ring_n, 1), 0.0)

    if find_defects:
        hot = (
            (status == TRUSTED) & (n_valid == nf)
            & (sums.min >= np.maximum(HOT_MIN_COUNTS, HOT_FACTOR * ring_mean[ring_px]))
        )
        status[hot] = HOT

    # Shadow: each block's mean background, pixels clipped at 10× their ring.
    bh, bw = -(-h // s), -(-w // s)
    by, bx = np.indices((bh, bw))
    centre_r = np.minimum(by * s + s // 2, h - 1)
    centre_c = np.minimum(bx * s + s // 2, w - 1)
    block_ring = np.minimum(
        (np.hypot(centre_c - beam_col, (centre_r - beam_row) * row_scale) / s).astype(np.int64), n_rings - 1
    )
    cap = 10.0 * ring_mean[block_ring] + 2.0
    use = (status == TRUSTED) & (n_valid >= 2)
    clipped = np.minimum(
        _block_view(np.where(use, background, 0.0).astype(np.float64), s, 0.0), cap[:, None, :, None]
    )
    n_block = _block_view(use, s, False).sum(axis=(1, 3))
    judged = 2 * n_block >= s * s
    with np.errstate(invalid="ignore", divide="ignore"):
        value = np.where(judged, _block_sums(clipped) / np.maximum(n_block, 1), 0.0)

    ring_median = np.zeros(n_rings)
    for k in np.unique(block_ring[judged]):
        ring_median[k] = _cpp_median(value[judged & (block_ring == k)])
    reach = max(1, SHADED_RING_REACH_PX // s)
    ks = np.minimum(np.arange(n_rings)[:, None] + np.arange(reach + 1)[None, :], n_rings - 1)
    outer = ring_median[ks].max(axis=1)
    expected = np.where(10000.0 * ring_median < VALUE_RANGE_LOW * outer, outer, ring_median)
    e = expected[block_ring]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = 10000.0 * value / e
    shaded_block = (
        judged & (e > 0) & (e * nf * n_block >= SHADOW_MIN_COUNTS)
        & ((ratio < VALUE_RANGE_LOW) | (ratio > VALUE_RANGE_HIGH))
    )
    shaded_px = np.repeat(np.repeat(shaded_block, s, axis=0), s, axis=1)[:h, :w]
    status[shaded_px & (status == TRUSTED)] = SHADED

    # Halo on the counts summed over the sampled frames (background × (n − 1)).
    usable = (status == TRUSTED) & (n_valid >= 2)
    counts = np.where(
        usable & (background >= 0), background * (n_valid - 1).astype(np.float32), np.float32(-1.0)
    ).astype(np.float32)
    halo, halo_radius = find_beam_halo(counts, usable, status == SHADED, beam_col, beam_row, row_scale)
    status[halo & (status == TRUSTED)] = HALO
    return status, s, halo_radius


def detect_beamstop(
    frames: Sequence[np.ndarray],
    beam_x: float,
    beam_y: float,
    *,
    saturation: float | None = None,
    row_scale: float = 1.0,
    find_defects: bool = True,
) -> Beamstop | None:
    """Return DEFPIX's shaded and halo pixels for *frames*.

    :param frames: The sampled frames, e.g. :data:`DEFPIX_FRAMES` picked by
        :func:`defpix_frames`; at least :data:`MIN_DETECT_FRAMES`.
    :param beam_x: Beam position along the fast axis (pixels), as
        ProcessingLab's ``BEAM`` x ÷ pixel size.
    :param beam_y: Beam position along the slow axis (pixels).
    :param saturation: The detector's overload level, or ``None`` if unknown.
    :param row_scale: Slow pixel size over fast pixel size.
    :param find_defects: Exclude defective and hot pixels from the test.
    :returns: The shaded and halo pixels, or ``None`` if there are neither.
    """
    sums = pixel_sums(frames, saturation)
    if sums.n_frames < MIN_DETECT_FRAMES:
        logger.info("Beamstop: DEFPIX needs %d frames, got %d", MIN_DETECT_FRAMES, sums.n_frames)
        return None
    status, block_px, halo_radius = classify_pixels(
        sums, beam_x, beam_y, row_scale=row_scale, find_defects=find_defects
    )
    mask = status == SHADED
    halo = status == HALO
    if not mask.any() and not halo.any():
        logger.info("Beamstop: no shaded or halo pixels")
        return None
    ys, xs = np.nonzero(mask | halo)
    found = Beamstop(
        mask=mask,
        n_pixels=int(mask.sum()),
        bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
        halo=halo,
        n_halo=int(halo.sum()),
        block_px=block_px,
        halo_radius_px=halo_radius,
    )
    x0, y0, x1, y1 = found.bbox
    logger.info(
        "Beamstop: %d shaded and %d halo pixels, x %d–%d, y %d–%d (%d frames, %d px blocks)",
        found.n_pixels, found.n_halo, x0, x1, y0, y1, sums.n_frames, block_px,
    )
    return found


def sweep_files(path: Path) -> list[Path]:
    """Return the images of *path*'s sweep, in frame order.

    The sweep is the files sharing *path*'s name with a different frame
    number (the last run of digits in the stem), *path* included.

    :param path: One image of the sweep.
    :returns: Image paths sorted by name.
    """
    match = re.search(r"(\d+)(?!.*\d)", path.stem)
    if match is None:
        return [path]
    prefix, suffix = path.stem[: match.start()], path.stem[match.end():]
    pattern = re.compile(rf"{re.escape(prefix)}\d{{{len(match.group(1))}}}{re.escape(suffix)}")
    files = sorted(p for p in path.parent.glob(f"{prefix}*{path.suffix}") if pattern.fullmatch(p.stem))
    return files or [path]
