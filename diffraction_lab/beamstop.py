"""Beamstop shadow detection.

MOSFLM's ``BACKSTOP AUTO`` keeps the lowest value of each pixel over a few
images (spots move, the shadow does not), smooths, finds the edge of a
circular backstop and then a straight arm. Real shadows are not always a disc
on an arm (a rod across the detector casts a band), so the shadow here is
found as XDS DEFPIX finds shaded regions: pixels much darker than the lit part
of their resolution ring. The shadow is the connected region that covers the
beam, bridged across detector gaps. Frames are combined by their median
rather than MOSFLM's minimum: it removes spots just as well, but keeps the
level of a weak background that the minimum drives to zero.

Fine-sliced data (an Eiger frame may hold 0.03 counts per pixel) has too few
counts per pixel to tell shadow from background, so consecutive frames are
first summed into groups (:func:`frame_groups`), and the median of the groups
is binned until the smoothing kernel holds :data:`TARGET_KERNEL_COUNTS`.

The opposite happens beside the stop: where the direct beam is not fully
blocked it flares into a halo, pixels far brighter than the rest of their
ring. The halo is the bright region, joined to the shadow or the beam, above
:data:`HALO_FACTOR` × its ring's normal level (the median of its unshadowed
pixels) by :data:`HALO_SIGNIFICANCE` σ. ProcessingLab's DEFPIX uses the same
test (``find_beam_halo``) and leaves the halo out of integration.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)

#: Gaussian smoothing of the combined image (pixels).
SMOOTH_SIGMA_PX = 3.0
#: A pixel is in shadow below this fraction of its ring's lit level.
SHADOW_FRACTION = 0.4
#: Percentile of a resolution ring taken as its lit level.
LIT_PERCENTILE = 75.0
#: Rings this far out (pixels) also set a ring's lit level, so rings wholly
#: inside the shadow still compare against lit pixels.
LIT_WINDOW_PX = 100
#: Shadow pixels within this distance (pixels) bridge a detector gap.
GAP_BRIDGE_PX = 10
#: The shadow must come within this distance (pixels) of the beam centre.
BEAM_SEARCH_PX = 15.0
#: Bin the combined image until the smoothing kernel (its 2σ disc) holds
#: about this many counts.
TARGET_KERNEL_COUNTS = 100.0
#: Largest bin (pixels on a side).
MAX_BIN_PX = 16
#: A pixel can only be shadow if its ring's lit level puts this many counts
#: in the smoothing kernel.
MIN_LIT_COUNTS = 30.0
#: A pixel is in the halo above this multiple of its ring's normal level.
HALO_FACTOR = 2.0
#: …and its excess counts in the smoothing kernel must exceed this many σ.
HALO_SIGNIFICANCE = 5.0
#: The halo must lie within this distance (pixels) of the shadow or the beam;
#: the step from shadow to halo spans a smoothing width or so.
HALO_REACH_PX = 18
#: Sum consecutive frames into a group until it averages this many counts per
#: pixel, or :data:`GROUP_MAX_FRAMES` frames.
GROUP_TARGET_COUNTS = 1.0
GROUP_MAX_FRAMES = 16


@dataclass(frozen=True)
class Beamstop:
    """A detected beamstop shadow and beam halo (image-pixel coordinates)."""

    #: ``True`` on shadowed pixels, the shape of the image.
    mask: np.ndarray
    #: Shadowed pixel count.
    n_pixels: int
    #: Bounding box ``(x0, y0, x1, y1)`` of shadow and halo, inclusive.
    bbox: tuple[int, int, int, int]
    #: ``True`` where the unblocked beam flares beside the stop (the halo).
    halo: np.ndarray | None = None
    #: Halo pixel count.
    n_halo: int = 0


def valid_pixels(frame: np.ndarray) -> np.ndarray:
    """Return ``True`` where a pixel holds data.

    Negative values (Pilatus gaps and bad pixels) and the integer dtype's
    maximum (Eiger module gaps) are invalid.

    :param frame: Raw detector frame.
    :returns: Boolean mask of measured pixels.
    """
    valid = np.isfinite(frame) & (frame >= 0)
    if np.issubdtype(frame.dtype, np.integer):
        valid &= frame < np.iinfo(frame.dtype).max
    return valid


def median_frame(frames: Iterable[np.ndarray]) -> np.ndarray:
    """Return the per-pixel median over *frames*, with invalid pixels as NaN.

    :param frames: Raw frames of one shape.
    :returns: Float image; NaN where any frame is invalid.
    :raises ValueError: If *frames* is empty.
    """
    stack = [np.where(valid_pixels(f), f, np.nan).astype(np.float32) for f in frames]
    if not stack:
        raise ValueError("No frames to combine")
    return np.median(np.stack(stack), axis=0).astype(np.float64)


def sum_frames(frames: Iterable[np.ndarray]) -> np.ndarray:
    """Return the per-pixel sum of *frames*, with invalid pixels as NaN.

    :param frames: Raw frames of one shape.
    :returns: Float image; NaN where any frame is invalid.
    :raises ValueError: If *frames* is empty.
    """
    total: np.ndarray | None = None
    for frame in frames:
        data = np.where(valid_pixels(frame), frame, np.nan).astype(np.float32)
        total = data if total is None else total + data
    if total is None:
        raise ValueError("No frames to sum")
    return total


def background_level(image: np.ndarray) -> float:
    """Return the mean counts per pixel, ignoring the brightest 1% (spots).

    Taken on every 7th pixel in each direction, which is plenty and keeps it
    fast on 16M-pixel frames.

    :param image: Float image, NaN where invalid.
    :returns: Mean of the valid sampled pixels at or below their 99th percentile.
    """
    sample = image[::7, ::7]
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return 0.0
    return float(sample[sample <= np.percentile(sample, 99)].mean())


def frame_groups(
    read_frame: Callable[[int], np.ndarray],
    n_frames: int,
    starts: Iterable[int],
    *,
    target: float = GROUP_TARGET_COUNTS,
    max_frames: int = GROUP_MAX_FRAMES,
) -> list[np.ndarray]:
    """Sum runs of consecutive frames into groups with enough counts.

    Each group starts at one of *starts* and adds frames until its
    :func:`background_level` reaches *target*, *max_frames* frames, or the
    sweep ends.

    :param read_frame: Returns the raw frame at a 0-based index.
    :param n_frames: Number of frames in the sweep.
    :param starts: First frame of each group.
    :param target: Mean counts per pixel to stop at.
    :param max_frames: Most frames in one group.
    :returns: One summed image per group (NaN where invalid).
    """
    groups = []
    for start in starts:
        total: np.ndarray | None = None
        for index in range(start, min(start + max_frames, n_frames)):
            frame = sum_frames([read_frame(index)])
            total = frame if total is None else total + frame
            if background_level(total) >= target:
                break
        if total is not None:
            groups.append(total)
    return groups


def bin_image(image: np.ndarray, size: int) -> np.ndarray:
    """Average *image* over ``size`` × ``size`` bins, ignoring NaN pixels.

    :param image: Float image, NaN where invalid.
    :param size: Bin edge (pixels).
    :returns: Binned image; NaN where under half a bin is valid.
    """
    if size <= 1:
        return image
    h, w = image.shape
    padded = np.full((-(-h // size) * size, -(-w // size) * size), np.nan)
    padded[:h, :w] = image
    blocks = padded.reshape(padded.shape[0] // size, size, padded.shape[1] // size, size)
    count = np.isfinite(blocks).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nansum(blocks, axis=(1, 3)) / count
    mean[count < size * size / 2] = np.nan
    return mean


def _smooth(image: np.ndarray, sigma: float = SMOOTH_SIGMA_PX) -> np.ndarray:
    """Gaussian-smooth *image*, ignoring NaN pixels (normalised convolution)."""
    valid = np.isfinite(image)
    num = ndimage.gaussian_filter(np.where(valid, image, 0.0), sigma)
    den = ndimage.gaussian_filter(valid.astype(np.float64), sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~valid] = np.nan
    return out


def ring_level(
    image: np.ndarray,
    radius: np.ndarray,
    percentile: float,
    *,
    window: int,
    use: np.ndarray | None = None,
    reduce: str = "max",
) -> np.ndarray:
    """Return each pixel's ring level: a percentile of its 1-pixel ring.

    Each ring's level is then the maximum (or minimum) over it and the rings
    up to *window* further out, so rings filled by the shadow (or the halo)
    take their level from clearer rings outside.

    :param image: Smoothed image, NaN where invalid.
    :param radius: Distance of each pixel from the beam (pixels).
    :param percentile: Percentile of each ring's pixels.
    :param window: Rings further out that also count (pixels).
    :param use: Pixels to include (default: every valid pixel).
    :param reduce: ``"max"`` or ``"min"`` over the window.
    :returns: Level per pixel (NaN where no ring data).
    """
    ring = radius.astype(np.int64)
    valid = np.isfinite(image) if use is None else np.isfinite(image) & use
    n_rings = int(ring.max()) + 1
    level = np.full(n_rings, np.nan)
    order = np.argsort(ring[valid], kind="stable")
    values = image[valid][order]
    bounds = np.searchsorted(ring[valid][order], np.arange(n_rings + 1))
    for r in range(n_rings):
        chunk = values[bounds[r]: bounds[r + 1]]
        if chunk.size:
            level[r] = np.percentile(chunk, percentile)
    # Running extreme over a window of rings further out.
    if reduce == "max":
        filled = np.nan_to_num(level, nan=-np.inf)
        windowed = ndimage.maximum_filter1d(filled, size=window + 1, origin=-(window // 2))
    else:
        filled = np.nan_to_num(level, nan=np.inf)
        windowed = ndimage.minimum_filter1d(filled, size=window + 1, origin=-(window // 2))
    windowed[~np.isfinite(windowed)] = np.nan
    return windowed[ring]


def lit_level(image: np.ndarray, radius: np.ndarray, window: int = LIT_WINDOW_PX) -> np.ndarray:
    """Return each pixel's lit level: the :data:`LIT_PERCENTILE` of its ring.

    The largest level over the ring and those up to *window* further out.

    :param image: Smoothed image, NaN where invalid.
    :param radius: Distance of each pixel from the beam (pixels).
    :param window: Rings further out that also count (pixels).
    :returns: Lit level per pixel (NaN where no ring data).
    """
    return ring_level(image, radius, LIT_PERCENTILE, window=window, reduce="max")


def find_halo(
    image: np.ndarray,
    radius: np.ndarray,
    shadow: np.ndarray,
    *,
    kernel_px: float,
    window: int,
    reach: int,
    beam_disc: np.ndarray,
) -> np.ndarray:
    """Return the halo: bright regions joined to the shadow or the beam.

    A ring's normal level is the median of its unshadowed pixels, taking the
    smallest over the window further out (rings near the beam are halo). A
    pixel is halo above :data:`HALO_FACTOR` × that level when its excess in the
    smoothing kernel exceeds :data:`HALO_SIGNIFICANCE` σ. ProcessingLab's
    ``find_beam_halo`` (DEFPIX) is the same test.

    :param image: Binned, smoothed counts, NaN where invalid.
    :param radius: Distance of each bin from the beam (bins).
    :param shadow: Shadowed bins.
    :param kernel_px: Pixels in the smoothing kernel (its 2σ disc).
    :param window: Rings further out that set a ring's normal level.
    :param reach: Bins from the shadow a halo region may start.
    :param beam_disc: Bins near enough the beam to anchor a halo region.
    :returns: Halo bins.
    """
    valid = np.isfinite(image)
    normal = ring_level(image, radius, 50.0, window=window, use=~shadow, reduce="min")
    with np.errstate(invalid="ignore"):
        excess = (image - normal) * kernel_px
        noise = np.sqrt(np.maximum(normal, 0.0) * kernel_px)
        bright = (
            valid & ~shadow & (image > HALO_FACTOR * normal)
            & (excess > HALO_SIGNIFICANCE * np.maximum(noise, 1.0))
        )
    labels, n = ndimage.label(bright)
    if n == 0:
        return bright
    anchor = ndimage.binary_dilation(shadow, iterations=reach) | beam_disc
    ids = np.unique(labels[anchor & bright])
    return np.isin(labels, ids[ids > 0])


def bin_size(level: float) -> int:
    """Return the bin edge that gives the smoothing kernel enough counts.

    Smoothing with σ = :data:`SMOOTH_SIGMA_PX` already averages a 2σ disc, so
    bins smaller than σ add nothing; above that the kernel is one bin wide.

    :param level: Mean counts per pixel of the combined image.
    :returns: Bin edge in pixels, 1 when no binning is needed.
    """
    needed = math.sqrt(TARGET_KERNEL_COUNTS / (4.0 * math.pi * max(level, 1e-9)))
    if needed <= SMOOTH_SIGMA_PX:
        return 1
    return int(min(MAX_BIN_PX, math.ceil(needed)))


def detect_beamstop(
    frames: Sequence[np.ndarray],
    beam_x: float,
    beam_y: float,
) -> Beamstop | None:
    """Detect the beamstop shadow, and the beam halo past it, at the beam centre.

    :param frames: One or more raw frames, or summed frame groups from
        :func:`frame_groups` (more remove more spots).
    :param beam_x: Beam centre x (pixels).
    :param beam_y: Beam centre y (pixels).
    :returns: The shadow and halo, or ``None`` if neither is at the beam.
    """
    combined = median_frame(frames)
    full_shape = combined.shape
    level = float(np.nanmean(combined)) if np.isfinite(combined).any() else 0.0
    size = bin_size(level)
    kernel_px = 4.0 * math.pi * max(SMOOTH_SIGMA_PX, size) ** 2
    image = _smooth(bin_image(combined, size), max(1.0, SMOOTH_SIGMA_PX / size))
    valid = np.isfinite(image)
    yy, xx = np.indices(image.shape)
    # Bin centres, in binned pixels.
    radius = np.hypot(xx + 0.5 - beam_x / size, yy + 0.5 - beam_y / size)
    lit = lit_level(image, radius, max(2, LIT_WINDOW_PX // size))
    with np.errstate(invalid="ignore"):
        shadow = valid & (image < SHADOW_FRACTION * lit) & (lit * kernel_px >= MIN_LIT_COUNTS)
    shadow = ndimage.binary_opening(shadow, iterations=2 if size < 4 else 1)
    # Gap pixels next to shadow join it, so a band crossing a gap stays whole.
    near = ndimage.binary_dilation(shadow, iterations=max(2, math.ceil(GAP_BRIDGE_PX / size)))
    labels, n = ndimage.label(shadow | (~valid & near))
    beam_disc = radius <= max(1.5, BEAM_SEARCH_PX / size)
    ids = np.unique(labels[beam_disc & shadow]) if n else np.array([], dtype=int)
    ids = ids[ids > 0]
    # Holes (pixels invalid in some frame, the attenuated direct beam) are
    # part of the shadow too.
    mask = ndimage.binary_fill_holes(np.isin(labels, ids) & valid) if ids.size else np.zeros_like(valid)
    halo = find_halo(
        image, radius, mask,
        kernel_px=kernel_px,
        window=max(2, LIT_WINDOW_PX // size),
        reach=max(2, math.ceil(HALO_REACH_PX / size)),
        beam_disc=beam_disc,
    )
    mask &= ~halo
    if not mask.any() and not halo.any():
        logger.info("Beamstop: no shadow or halo within %.0f px of the beam", BEAM_SEARCH_PX)
        return None
    if size > 1:
        mask, halo = (
            np.repeat(np.repeat(m, size, axis=0), size, axis=1)[: full_shape[0], : full_shape[1]]
            for m in (mask, halo)
        )
    ys, xs = np.nonzero(mask | halo)
    found = Beamstop(
        mask=mask,
        n_pixels=int(mask.sum()),
        bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
        halo=halo,
        n_halo=int(halo.sum()),
    )
    x0, y0, x1, y1 = found.bbox
    logger.info(
        "Beamstop: %d shadowed and %d halo pixels, x %d–%d, y %d–%d (%.2f counts/pixel, %d px bins)",
        found.n_pixels, found.n_halo, x0, x1, y0, y1, level, size,
    )
    return found


def sibling_frames(path: Path, count: int = 5) -> list[Path]:
    """Return up to *count* evenly spaced images from *path*'s sweep.

    The sweep is the files sharing *path*'s name with a different frame
    number (the last run of digits in the stem); *path* is always included.

    :param path: One image of the sweep.
    :param count: Maximum number of images to return.
    :returns: Image paths, *path* first.
    """
    match = re.search(r"(\d+)(?!.*\d)", path.stem)
    if match is None or count <= 1:
        return [path]
    prefix, suffix = path.stem[: match.start()], path.stem[match.end():]
    pattern = re.compile(rf"{re.escape(prefix)}\d{{{len(match.group(1))}}}{re.escape(suffix)}")
    others = sorted(
        p for p in path.parent.glob(f"{prefix}*{path.suffix}") if p != path and pattern.fullmatch(p.stem)
    )
    if not others:
        return [path]
    picks = np.linspace(len(others) - 1, 0, num=min(count - 1, len(others)))
    return [path, *(others[int(round(i))] for i in picks)]
