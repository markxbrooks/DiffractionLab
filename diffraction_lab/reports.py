"""Per-image reports for a ProcessingLab run, in the spirit of autoPROC's.

autoPROC plots the spots found on each image and Aimless's Rmerge against
batch (image) number. Here:

* **Spots per image** comes from ProcessingLab's XDS ``INTEGRATE.LP``-style
  per-image table in ``process.log`` (``IMAGE IER SCALE NBKG NOVL NEWALD
  NSTRONG …``), which covers every integrated frame, not only the
  spot-finding wedge. ``NSTRONG`` is drawn with ``NEWALD`` (reflections
  predicted on the image) for reference. The table is printed once per
  refinement block; the last row for each image wins.
* **R_meas by image** comes from ProcessingLab's ``IMAGE_RMEAS`` table in
  ``process.log``. It is computed from all integrated observations before
  scaling, grouped by image.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

logger = logging.getLogger(__name__)

#: Images per block in the R_meas chart (as XDS / ProcessingLab batches).
BLOCK_IMAGES = 25

_IMAGE_HEADER = re.compile(r"^\s*IMAGE\s+IER\s+")
_IMAGE_RMEAS_HEADER = re.compile(r"^\s*IMAGE_RMEAS\s+columns=(.+)$")
_IMAGE_RMEAS_END = "IMAGE_RMEAS_END"


@dataclass(frozen=True)
class ImageTable:
    """ProcessingLab's per-image integration table (one row per image)."""

    #: Image numbers, ascending.
    images: np.ndarray
    #: Column name (``NSTRONG``, ``NEWALD``, ``SIGMAR``, …) → values per image.
    columns: dict[str, np.ndarray]

    def column(self, name: str) -> np.ndarray | None:
        """Values of column *name* per image, or ``None`` if the table lacks it."""
        return self.columns.get(name)


def read_image_table(path: Path) -> ImageTable | None:
    """Read the per-image table(s) in a ProcessingLab ``process.log``.

    :param path: ``process.log``.
    :returns: The table (the last row printed for each image), or ``None``
        if the log is unreadable or has none.
    """
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    rows: dict[int, list[float]] = {}
    names: list[str] = []
    header: list[str] | None = None
    for line in lines:
        if _IMAGE_HEADER.match(line):
            header = line.split()
            names = names or header
            continue
        if header is None:
            continue
        parts = line.split()
        if not parts:
            header = None  # a blank line ends the table
            continue
        # The block's SIGMAR / refinement lines interrupt the table before
        # its last image, so other lines are skipped, not taken as its end.
        if len(parts) != len(header) or not parts[0].isdigit():
            continue
        try:
            rows[int(parts[0])] = [float(v) for v in parts]
        except ValueError:
            continue
    if not rows or not names:
        return None
    images = np.array(sorted(rows), dtype=int)
    table = np.array([rows[i] for i in images], dtype=float)
    columns = {name: table[:, j] for j, name in enumerate(names) if name != "IMAGE"}
    return ImageTable(images=images, columns=columns)


@dataclass(frozen=True)
class ImageRmeasTable:
    """ProcessingLab's R_meas statistics for each integrated image."""

    images: np.ndarray
    n_accepted: np.ndarray
    n_compared: np.ndarray
    rmeas_pct: np.ndarray


def read_image_rmeas(path: Path) -> ImageRmeasTable | None:
    """Read the ``IMAGE_RMEAS`` table from a ProcessingLab ``process.log``."""
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    rows: dict[int, tuple[int, int, float]] = {}
    in_table = False
    for line in lines:
        header = _IMAGE_RMEAS_HEADER.match(line)
        if header:
            in_table = header.group(1).strip() == "image,n_accepted,n_compared,rmeas_pct"
            continue
        if not in_table:
            continue
        if line.strip() == _IMAGE_RMEAS_END:
            in_table = False
            continue
        fields = line.split(",")
        if len(fields) != 4:
            continue
        try:
            image, accepted, compared = map(int, fields[:3])
            rmeas = float(fields[3])
        except ValueError:
            continue
        rows[image] = (accepted, compared, rmeas)
    if not rows:
        return None
    images = np.array(sorted(rows), dtype=int)
    values = [rows[i] for i in images]
    return ImageRmeasTable(
        images=images,
        n_accepted=np.array([row[0] for row in values], dtype=int),
        n_compared=np.array([row[1] for row in values], dtype=int),
        rmeas_pct=np.array([row[2] for row in values], dtype=float),
    )


@dataclass(frozen=True)
class RmeasByImage:
    """Per-image R_meas contributions from scaled unmerged observations."""

    batches: np.ndarray
    numerator: np.ndarray
    denominator: np.ndarray
    n_obs: np.ndarray

    def __post_init__(self) -> None:
        arrays = {
            "batches": np.array(self.batches, copy=True),
            "numerator": np.array(self.numerator, dtype=float, copy=True),
            "denominator": np.array(self.denominator, dtype=float, copy=True),
            "n_obs": np.array(self.n_obs, copy=True),
        }

        shapes = {a.shape for a in arrays.values()}
        if len(shapes) != 1:
            raise ValueError("All arrays must have the same shape")

        batches = arrays["batches"]
        if batches.ndim != 1:
            raise ValueError("Arrays must be one-dimensional")

        if not np.issubdtype(batches.dtype, np.integer):
            raise TypeError("batches must contain integer image numbers")

        if np.any(batches < 1):
            raise ValueError("Image numbers must be positive and one-based")

        if np.any(np.diff(batches) <= 0):
            raise ValueError("batches must be strictly increasing")

        for name, array in arrays.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @staticmethod
    def _ratio(
        numerator: np.ndarray,
        denominator: np.ndarray,
    ) -> np.ndarray:
        valid = (
            np.isfinite(numerator)
            & np.isfinite(denominator)
            & (denominator > 0)
        )
        result = np.full(numerator.shape, np.nan, dtype=float)
        np.divide(numerator, denominator, out=result, where=valid)
        return result

    @property
    def rmeas(self) -> np.ndarray:
        """R_meas per image; NaN for invalid denominators or inputs."""
        return self._ratio(self.numerator, self.denominator)

    @property
    def overall(self) -> float:
        """Pooled R_meas across all images."""
        numerator = float(self.numerator.sum())
        denominator = float(self.denominator.sum())

        if not np.isfinite(numerator) or not np.isfinite(denominator):
            return float("nan")
        if denominator <= 0:
            return float("nan")

        return numerator / denominator

    def blocks(
        self,
        size: int = BLOCK_IMAGES,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pool consecutive one-based image numbers into blocks."""
        if not isinstance(size, (int, np.integer)) or isinstance(
            size, (bool, np.bool_)
        ):
            raise TypeError("size must be an integer")

        if size <= 0:
            raise ValueError("size must be positive")

        block_index = (self.batches - 1) // size
        keys, inverse = np.unique(block_index, return_inverse=True)

        num = np.bincount(
            inverse, weights=self.numerator, minlength=len(keys)
        )
        den = np.bincount(
            inverse, weights=self.denominator, minlength=len(keys)
        )

        return (
            keys * size + 1,
            keys * size + size,
            self._ratio(num, den),
        )


@dataclass(frozen=True)
class RmeasByImageOld:
    """R_meas per image (batch) from scaled unmerged observations."""

    #: Batch (image) numbers, ascending.
    batches: np.ndarray
    #: ``Σ √(n/(n−1))·|I − ⟨I⟩|`` per batch.
    numerator: np.ndarray
    #: ``Σ I`` per batch (observations with *n* ≥ 2).
    denominator: np.ndarray
    #: Observations per batch with *n* ≥ 2.
    n_obs: np.ndarray

    @property
    def rmeas(self) -> np.ndarray:
        """R_meas per batch (NaN where ``Σ I`` ≤ 0)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.denominator > 0, self.numerator / self.denominator, np.nan)

    @property
    def overall(self) -> float:
        """R_meas over every batch."""
        den = float(self.denominator.sum())
        return float(self.numerator.sum()) / den if den > 0 else float("nan")

    def blocks(self, size: int = BLOCK_IMAGES) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pool consecutive images into blocks of *size* (images 1–size, …).

        :param size: Images per block.
        :returns: ``(first, last, rmeas)`` per block that holds observations.
        """
        index = (self.batches - 1) // size
        keys = np.unique(index)
        num = np.array([self.numerator[index == k].sum() for k in keys])
        den = np.array([self.denominator[index == k].sum() for k in keys])
        with np.errstate(invalid="ignore", divide="ignore"):
            values = np.where(den > 0, num / den, np.nan)
        return keys * size + 1, keys * size + size, values


def rmeas_from_observations(
    hkl: np.ndarray,
    isym: np.ndarray,
    batch: np.ndarray,
    intensity: np.ndarray,
    *,
    anomalous: bool = False,
) -> RmeasByImage:
    """R_meas per batch from unmerged observations.

    :param hkl: ``(n, 3)`` reduced (asymmetric-unit) indices.
    :param isym: Symmetry numbers (``M/ISYM`` mod 256); odd is I+, even I−.
    :param batch: Batch (image) number of each observation.
    :param intensity: Scaled intensities.
    :param anomalous: Keep I+ and I− apart, as Aimless with ``ANOMALOUS ON``.
    :returns: Per-batch sums (see :class:`RmeasByImage`).
    """
    keys = np.column_stack([hkl, isym % 2]) if anomalous else np.asarray(hkl)
    _, group = np.unique(keys, axis=0, return_inverse=True)
    group = group.ravel()
    n = np.bincount(group)
    mean = np.bincount(group, weights=intensity) / n
    multiple = n[group] >= 2
    factor = np.sqrt(n / np.maximum(n - 1, 1))[group]
    deviation = np.where(multiple, factor * np.abs(intensity - mean[group]), 0.0)
    counted = np.where(multiple, intensity, 0.0)
    batches, slot = np.unique(batch, return_inverse=True)
    return RmeasByImage(
        batches=batches.astype(int),
        numerator=np.bincount(slot, weights=deviation),
        denominator=np.bincount(slot, weights=counted),
        n_obs=np.bincount(slot, weights=multiple.astype(float)).astype(int),
    )


def rmeas_by_image(unmerged_mtz: Path, *, anomalous: bool = False) -> RmeasByImage:
    """R_meas per image from Aimless's unmerged output.

    :param unmerged_mtz: ``aimless_unmerged.mtz`` (``OUTPUT MERGED UNMERGED``).
    :param anomalous: Keep I+ and I− apart.
    :returns: Per-batch sums.
    :raises OSError: If the file cannot be read.
    :raises KeyError: If it lacks H, K, L, M/ISYM, BATCH or I.
    """
    import gemmi

    mtz = gemmi.read_mtz_file(str(unmerged_mtz))
    data = np.array(mtz, copy=False)
    col = {label: i for i, label in enumerate(mtz.column_labels())}
    hkl = data[:, [col["H"], col["K"], col["L"]]].astype(int)
    isym = data[:, col["M/ISYM"]].astype(int) % 256
    keep = np.isfinite(data[:, col["I"]])
    return rmeas_from_observations(
        hkl[keep], isym[keep], data[keep, col["BATCH"]].astype(int), data[keep, col["I"]].astype(float),
        anomalous=anomalous,
    )


def unmerged_path(merged_mtz: Path) -> Path:
    """Where Aimless writes the unmerged file for *merged_mtz* (``…_unmerged.mtz``)."""
    merged_mtz = Path(merged_mtz)
    return merged_mtz.with_name(f"{merged_mtz.stem}_unmerged{merged_mtz.suffix}")


class _Chart(FigureCanvasQTAgg):
    """A one-axes matplotlib chart with a waiting message."""

    def __init__(self, message: str, parent: QWidget | None = None) -> None:
        self.figure = Figure(figsize=(9, 2.6), layout="constrained")
        super().__init__(self.figure)
        self.setParent(parent)
        self.clear(message)

    def clear(self, message: str = "") -> None:
        """Show an empty chart with *message*."""
        self.figure.clear()
        self.figure.text(0.5, 0.5, message, ha="center", va="center", color="0.4")
        self.draw_idle()


class SpotsPerImageCanvas(_Chart):
    """Strong spots (and predicted reflections) on each image."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Spots per image: integrate first", parent)

    def show_table(self, table: ImageTable) -> None:
        """Draw ``NSTRONG`` (and ``NEWALD``) against image number."""
        self.figure.clear()
        ax = self.figure.subplots()
        strong = table.column("NSTRONG")
        predicted = table.column("NEWALD")
        if predicted is not None:
            ax.plot(table.images, predicted, color="0.65", lw=0.8, label="predicted (NEWALD)")
        if strong is not None:
            ax.plot(table.images, strong, color="C0", lw=1.0, label="strong spots (NSTRONG)")
        ax.set_xlabel("image")
        ax.set_ylabel("spots per image")
        ax.set_xlim(table.images.min(), table.images.max())
        ax.set_ylim(bottom=0)
        ax.legend(loc="best", fontsize=7, framealpha=0.6)
        ax.set_title("Spots per image (ProcessingLab integration)", fontsize=9)
        self.draw_idle()


class RmeasPerImageCanvas(_Chart):
    """R_meas per image, with blocks of :data:`BLOCK_IMAGES` and the overall value."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("R_meas by image: integrate first", parent)

    def show_rmeas(self, result: ImageRmeasTable) -> None:
        """Draw ProcessingLab's per-image R_meas values."""
        self.figure.clear()
        ax = self.figure.subplots()
        per_image = result.rmeas_pct
        ax.plot(result.images, per_image, color="C3", lw=0.8, label="per image")
        finite = per_image[np.isfinite(per_image) & (per_image >= 0)]
        top = np.percentile(finite, 99) if finite.size else 0.0
        ax.set_ylim(0, 1.1 * top if top > 0 else 1.0)
        ax.set_xlim(result.images.min() - 0.5, result.images.max() + 0.5)
        ax.set_xlabel("image")
        ax.set_ylabel("R_meas (%)")
        ax.legend(loc="best", fontsize=7, framealpha=0.6)
        ax.set_title("R_meas by image (ProcessingLab)", fontsize=9)
        self.draw_idle()


class ReportsWidget(QWidget):
    """The Reports tab: spots per image and R_meas by image."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.spots_chart = SpotsPerImageCanvas(self)
        self.rmeas_chart = RmeasPerImageCanvas(self)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.spots_chart, stretch=1)
        layout.addWidget(self.rmeas_chart, stretch=1)
        layout.addWidget(self.note)
        self._loaded: dict[str, tuple] = {}

    @staticmethod
    def _stamp(path: Path | None) -> tuple:
        try:
            return (str(path), path.stat().st_mtime_ns) if path else (None, None)
        except OSError:
            return (str(path), None)

    def load(self, process_log: Path | None, unmerged_mtz: Path | None = None, *, anomalous: bool = False) -> None:
        """Redraw from a run's ``process.log``.

        Files already drawn (same path and modification time) are not re-read.
        ``unmerged_mtz`` and ``anomalous`` remain accepted for older callers.
        """
        notes: list[str] = []
        key = self._stamp(process_log)
        if key != self._loaded.get("log"):
            self._loaded["log"] = key
            table = read_image_table(process_log) if process_log else None
            if table is None:
                self.spots_chart.clear("Spots per image: integrate first")
            else:
                self.spots_chart.show_table(table)
        if key != self._loaded.get("image_rmeas"):
            self._loaded["image_rmeas"] = key
            table = read_image_rmeas(process_log) if process_log else None
            if table is None:
                self.rmeas_chart.clear("R_meas by image: integrate first")
            else:
                self.rmeas_chart.show_rmeas(table)
        self.note.setText(" ".join(notes))
