"""Pixel-intensity histogram of a diffraction image.

Detector images are photon counts: whole numbers, mostly 0, 1 or 2, with a
long tail of strong pixels (the Bragg spots) up to the overload. A histogram
that shows this needs:

- **Every real count**, including 0 (the commonest value on a photon-counting
  detector), but not detector gaps or masked pixels: negative values
  (Pilatus gaps −1, bad pixels −2) and the type's maximum (Eiger marks module
  gaps with 2¹⁶−1 / 2³²−1), nor values above the detector's overload, which
  no real pixel can read (e.g. a stuck Pilatus pixel at its 20-bit counter
  maximum, 2²⁰−1).
- **Unit bins for small counts** (one per integer) and logarithmic bins
  above, up to the brightest pixel, so the background and the spots both
  show.
- **Logarithmic axes**, plotted with :func:`plot_pixel_histogram`: pixel
  counts span seven orders of magnitude.

Images that are not integer counts (e.g. JPEG/TIFF converted to float) get
linear bins over their full range.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PixelHistogram:
    """A histogram of one frame, with how its pixels were classified.

    :param edges: Bin edges (counts); ``len(edges) == len(counts) + 1``.
    :param counts: Pixels per bin.
    :param integer: Whether the frame holds integer counts (unit + log bins).
    :param linear_limit: Unit bins run up to here (integer frames).
    :param n_total: Every pixel.
    :param n_valid: Pixels histogrammed (zeros and overloads included).
    :param n_zero: Valid pixels with 0 counts.
    :param n_gap: Negative pixels: detector gaps and bad pixels.
    :param n_masked: Masked pixels (the type's maximum) and non-finite values.
    :param n_overloaded: Valid pixels at ``overload`` (saturated).
    :param overload: The detector's overload count, if known.
    :param median: Median of the valid pixels.
    :param mean: Mean of the valid pixels.
    :param p999: 99.9th percentile of the valid pixels.
    :param maximum: Largest valid pixel.
    :param n_above_cutoff: Pixels above ``overload``: not real counts
        (stuck or defective pixels), so excluded.
    """

    edges: np.ndarray
    counts: np.ndarray
    integer: bool
    linear_limit: int
    n_total: int
    n_valid: int
    n_zero: int
    n_gap: int
    n_masked: int
    n_overloaded: int
    overload: float | None
    median: float
    mean: float
    p999: float
    maximum: float
    n_above_cutoff: int = 0

    @property
    def statistics(self) -> str:
        """The key numbers of the valid pixels, on one line."""
        return " · ".join([f"{self.n_valid:,} pixels", f"{self.n_zero / max(self.n_valid, 1):.1%} zero",
                           f"median {self.median:g}", f"mean {self.mean:.3g}", f"99.9% {self.p999:g}",
                           f"max {self.maximum:g}"])

    @property
    def notes(self) -> str:
        """Overloaded pixels and what was excluded (empty if nothing)."""
        parts = [f"{self.n_overloaded:,} overloaded"] if self.n_overloaded else []
        excluded = [f"{n:,} {what}" for n, what in ((self.n_gap, "gap/bad"), (self.n_masked, "masked"),
                                                    (self.n_above_cutoff, "above cutoff")) if n]
        if excluded:
            parts.append("excluded " + ", ".join(excluded))
        return " · ".join(parts)

    @property
    def summary(self) -> str:
        """Statistics and notes on one line (for a status bar)."""
        return " · ".join(s for s in (self.statistics, self.notes) if s)


def pixel_histogram(frame: np.ndarray, *, overload: float | None = None, linear_limit: int = 20,
                    bins_per_decade: int = 20, float_bins: int = 256) -> PixelHistogram:
    """Histogram a detector frame.

    :param frame: The raw frame, in its native dtype (so masked values are
        recognisable).
    :param overload: The detector's overload count, or ``None``. Pixels at it
        are kept and counted as overloaded; pixels above it are excluded.
    :param linear_limit: Integer frames get one bin per count up to here.
    :param bins_per_decade: Logarithmic bins per factor of 10 above it.
    :param float_bins: Bins for frames that are not integer counts.
    :returns: The histogram.
    :raises ValueError: If no pixel is valid.
    """
    data = np.asarray(frame)
    flat = data.ravel()
    n_total = int(flat.size)
    masked = ~np.isfinite(flat) if np.issubdtype(flat.dtype, np.floating) else np.zeros(flat.shape, bool)
    if np.issubdtype(flat.dtype, np.unsignedinteger):
        masked |= flat == np.iinfo(flat.dtype).max  # Eiger module gaps / masked pixels
    with np.errstate(invalid="ignore"):
        gap = (flat < 0) & ~masked
    valid = flat[~masked & ~gap].astype(np.float64, copy=False)
    n_above_cutoff = 0
    if overload is not None:
        above = valid > overload
        n_above_cutoff = int(np.count_nonzero(above))
        if n_above_cutoff:
            valid = valid[~above]
    if valid.size == 0:
        raise ValueError(f"no valid pixels ({n_total:,} total: {int(gap.sum()):,} gap/bad, "
                         f"{int(masked.sum()):,} masked, {n_above_cutoff:,} above cutoff)")

    integer = bool(np.issubdtype(flat.dtype, np.integer)) or bool(
        np.all(np.mod(valid[:: max(1, valid.size // 100_000)], 1.0) == 0.0))
    maximum = float(valid.max())
    if integer:
        top = max(maximum, 0.0)
        edges = np.arange(-0.5, min(top, linear_limit) + 1.0, 1.0)
        if top > linear_limit:
            decades = np.log10((top + 1.0) / (linear_limit + 0.5))
            n_log = max(1, int(np.ceil(decades * bins_per_decade)))
            log_edges = np.geomspace(linear_limit + 0.5, top + 1.0, n_log + 1)
            edges = np.concatenate([edges[:-1], log_edges])
    else:
        low = float(valid.min())
        edges = np.linspace(low, maximum if maximum > low else low + 1.0, float_bins + 1)
    counts, edges = np.histogram(valid, bins=edges)

    return PixelHistogram(
        edges=edges,
        counts=counts,
        integer=integer,
        linear_limit=linear_limit,
        n_total=n_total,
        n_valid=int(valid.size),
        n_zero=int(np.count_nonzero(valid == 0.0)),
        n_gap=int(gap.sum()),
        n_masked=int(masked.sum()),
        n_overloaded=int(np.count_nonzero(valid >= overload)) if overload else 0,
        overload=overload,
        median=float(np.median(valid)),
        mean=float(valid.mean()),
        p999=float(np.percentile(valid, 99.9)),
        maximum=maximum,
        n_above_cutoff=n_above_cutoff,
    )


def plot_pixel_histogram(ax, hist: PixelHistogram, title: str = "Pixel intensity histogram") -> None:
    """Draw *hist* on matplotlib axes *ax*.

    Integer frames: a count axis that is linear up to ``linear_limit`` and
    logarithmic above (``symlog``), so 0 is shown; pixels per bin on a log
    axis. The overload and the 99.9th percentile are marked.

    :param ax: Matplotlib axes.
    :param hist: From :func:`pixel_histogram`.
    :param title: Plot title (the summary goes underneath).
    """
    nonzero = np.where(hist.counts > 0, hist.counts, np.nan)
    ax.stairs(nonzero, hist.edges, fill=True, color="#4C78A8", alpha=0.85)
    if hist.integer and hist.maximum > hist.linear_limit:
        ax.set_xscale("symlog", linthresh=hist.linear_limit + 0.5, linscale=1.0)
    ax.set_yscale("log")
    ax.set_xlim(hist.edges[0], hist.edges[-1])
    ax.set_xlabel("Counts per pixel" if hist.integer else "Pixel value")
    ax.set_ylabel("Pixels")
    if hist.p999 > hist.edges[0]:
        ax.axvline(hist.p999, color="0.35", linestyle=":", linewidth=1, label=f"99.9%: {hist.p999:g}")
    if hist.overload is not None and hist.edges[0] < hist.overload <= hist.edges[-1] * 1.5:
        ax.axvline(hist.overload, color="#dc3545", linestyle="--", linewidth=1,
                   label=f"overload {hist.overload:g}")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title("\n".join(s for s in (title, hist.statistics, hist.notes) if s), fontsize=9)
