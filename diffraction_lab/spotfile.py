"""Mosflm spot-file I/O and a simple peak finder for indexing.

Spot-file layout matches ``Indexwizard::getSpotfileFirstImage`` /
``Spot::write`` in the iMosflm TCL sources.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class Spot:
    """One diffraction spot in millimetres (Mosflm convention)."""

    x_mm: float
    y_mm: float
    intensity: float
    sigma: float


@dataclass
class SpotList:
    """Spots from one image / phi wedge."""

    spots: list[Spot] = field(default_factory=list)
    phi_start: float = 0.0
    phi_end: float = 0.0

    @property
    def mid_phi(self) -> float:
        return 0.5 * (float(self.phi_start) + float(self.phi_end))

    @property
    def osc_range(self) -> float:
        return abs(float(self.phi_end) - float(self.phi_start))


@dataclass
class SpotFileGeometry:
    """Header geometry written at the top of a Mosflm spotfile."""

    image_width: int
    image_height: int
    pixel_size_mm: float
    beam_x_mm: float
    beam_y_mm: float
    yscale: float = 1.0
    detector_omega: float = 0.0
    invert_x: bool = False


def write_spot_line(
    x_mm: float,
    y_mm: float,
    osc_range: float,
    mid_phi: float,
    intensity: float,
    sigma: float,
) -> str:
    """Format one spot (or sentinel) line as in ``Spot::write``."""
    return (
        f"{x_mm:11.2f}{y_mm:10.2f}{osc_range:9.3f}{mid_phi:9.3f}"
        f"{intensity:12.1f}{sigma:10.1f}"
    )


def write_merged_spotfile(
    path: Path,
    geometry: SpotFileGeometry,
    spotlists: list[SpotList],
) -> Path:
    """Write a temporary merged spotfile for ``autoindex dps``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write(
            f"{geometry.image_width:12d}{geometry.image_height:12d}"
            f"{geometry.pixel_size_mm:11.8f}{geometry.yscale:12.6f}"
            f"{geometry.detector_omega:12.6f}\n"
        )
        handle.write(
            f"{int(geometry.invert_x):12d}{1:12d}\n"
        )
        handle.write(
            f"{geometry.beam_x_mm:11.5f}{geometry.beam_y_mm:11.5f}\n"
        )
        for index, spotlist in enumerate(spotlists):
            if index > 0:
                handle.write(write_spot_line(-99, -99, -99, -99, -99, -99) + "\n")
            for spot in spotlist.spots:
                handle.write(
                    write_spot_line(
                        spot.x_mm,
                        spot.y_mm,
                        spotlist.osc_range,
                        spotlist.mid_phi,
                        spot.intensity,
                        spot.sigma,
                    )
                    + "\n"
                )
        handle.write(write_spot_line(-999, -999, -999, -999, -999, -999) + "\n")
        handle.write(
            f"{geometry.image_width:7d}{geometry.image_height:6d}"
            f"{geometry.pixel_size_mm:10.4f}{int(geometry.yscale):5d}\n"
        )
    return path


def new_temp_spotfilename(directory: Path) -> Path:
    """Create an exclusive temp spotfile path like TCL ``msf#####.tmp``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for _ in range(50):
        candidate = directory / f"msf{random.randint(0, 99999):05d}.tmp"
        try:
            with candidate.open("x", encoding="ascii"):
                pass
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create temporary spotfile in {directory}")


def copy_spotfile(source: Path, destination: Path) -> Path:
    """Copy a Mosflm ``.spt`` (or temp spotfile) to ``destination``.

    If ``destination`` has no ``.spt`` / ``.tmp`` suffix, ``.spt`` is appended.
    """
    source = Path(source)
    destination = Path(destination)
    if destination.suffix.lower() not in {".spt", ".tmp"}:
        destination = destination.with_suffix(".spt")
    if not source.is_file():
        raise FileNotFoundError(f"Spot file not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())
    return destination


def find_spots_in_frame(
    frame: np.ndarray,
    *,
    pixel_size_mm: float,
    threshold_sigma: float = 5.0,
    min_distance: int = 8,
    max_spots: int = 500,
    border: int = 20,
) -> list[Spot]:
    """Detect strong peaks in a detector frame and return spots in mm.

    This is a lightweight stand-in for Mosflm ``findspots`` when driving
    indexing from frames already loaded in DiffractionLab (e.g. HDF5). When the
    Mosflm binary can read the images directly, prefer its findspots path.
    """
    import cv2  # local: keep spotfile I/O usable without OpenCV installed

    data = np.asarray(frame, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"Expected 2-D frame, got shape {data.shape}")
    # Mask PAD dead/hot sentinels (Eiger uses 2^16-1 / 2^32-1; zeros are gaps).
    valid = (
        np.isfinite(data)
        & (data > 0)
        & (data < 60000)
        & (data < 1.0e8)
    )
    if not np.any(valid):
        return []
    sample = data[valid]
    # Background from a robust low-intensity sample (spots are sparse).
    bg_cut = float(np.percentile(sample, 80.0))
    background = sample[sample <= bg_cut]
    if background.size < 100:
        background = sample
    med = float(np.median(background))
    mad = float(np.median(np.abs(background - med))) + 1e-6
    sigma = max(1.4826 * mad, 0.5)
    cutoff = med + threshold_sigma * sigma
    # Absolute floor so very flat frames still pick real peaks.
    cutoff = max(cutoff, med + 3.0)

    work = np.zeros_like(data, dtype=np.float32)
    work[valid] = data[valid].astype(np.float32)
    # Local maxima via dilation.
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(work, kernel)
    maxima = (work >= dilated) & (work >= cutoff) & valid
    maxima[:border, :] = False
    maxima[-border:, :] = False
    maxima[:, :border] = False
    maxima[:, -border:] = False
    ys, xs = np.nonzero(maxima)
    if ys.size == 0:
        return []
    intensities = work[ys, xs]
    order = np.argsort(-intensities)
    picked: list[tuple[int, int, float]] = []
    min_dist2 = float(min_distance * min_distance)
    for idx in order:
        y = int(ys[idx])
        x = int(xs[idx])
        inten = float(intensities[idx])
        if any((x - px) * (x - px) + (y - py) * (y - py) < min_dist2 for px, py, _ in picked):
            continue
        picked.append((x, y, inten))
        if len(picked) >= max_spots:
            break

    spots: list[Spot] = []
    for x, y, inten in picked:
        sigma_i = max(sigma, np.sqrt(max(inten, 1.0)))
        spots.append(
            Spot(
                x_mm=float(x) * float(pixel_size_mm),
                y_mm=float(y) * float(pixel_size_mm),
                intensity=inten,
                sigma=float(sigma_i),
            )
        )
    return spots
