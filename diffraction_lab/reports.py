"""Per-image reports for a ProcessingLab run, in the spirit of autoPROC's.

autoPROC plots the spots found on each image and Aimless's Rmerge against
batch (image) number. Here:

* **Spots per image** comes from ProcessingLab's XDS ``INTEGRATE.LP``-style
  per-image table in ``process.log`` (``IMAGE IER SCALE NBKG NOVL NEWALD
  NSTRONG …``), which covers every integrated frame, not only the
  spot-finding wedge. ``NSTRONG`` is drawn with ``NEWALD`` (reflections
  predicted on the image) for reference. The table is printed once per
  refinement block; the last row for each image wins.
* **R_meas by image** is computed from Aimless's scaled, unmerged output
  (``OUTPUT MERGED UNMERGED`` → ``aimless_unmerged.mtz``): each observation
  contributes ``√(n/(n−1))·|I − ⟨I⟩|`` (⟨I⟩ over its *n* symmetry
  equivalents in the whole data set, *n* ≥ 2) to its image, divided by the
  image's ``Σ I``. Blocks of images pool numerators and denominators, so a
  block's value is exactly the R_meas of its observations. Aimless itself
  reports only Rmerge per batch.
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
class RmeasByImage:
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
        super().__init__("R_meas by image: scale with Aimless first", parent)

    def show_rmeas(self, result: RmeasByImage, block: int = BLOCK_IMAGES) -> None:
        """Draw per-image R_meas, its block values and the overall line."""
        self.figure.clear()
        ax = self.figure.subplots()
        per_image = result.rmeas
        ax.plot(result.batches, per_image, color="C3", lw=0.6, alpha=0.45, label="per image")
        first, last, values = result.blocks(block)
        steps_x = np.column_stack([first - 0.5, last + 0.5]).ravel()
        ax.plot(steps_x, np.repeat(values, 2), color="C3", lw=2.0, label=f"blocks of {block} images")
        overall = result.overall
        ax.axhline(overall, color="0.3", ls="--", lw=1.0, label=f"overall {overall:.3f}")
        finite = per_image[np.isfinite(per_image)]
        top = max(np.nanmax(values) if np.isfinite(values).any() else 0.0,
                  np.percentile(finite, 99) if finite.size else 0.0)
        ax.set_ylim(0, 1.1 * top if top > 0 else 1.0)
        ax.set_xlim(result.batches.min() - 0.5, result.batches.max() + 0.5)
        ax.set_xlabel("image (batch)")
        ax.set_ylabel("R_meas")
        ax.legend(loc="best", fontsize=7, framealpha=0.6)
        ax.set_title("R_meas by image (Aimless-scaled, unmerged)", fontsize=9)
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

    def load(self, process_log: Path | None, unmerged_mtz: Path | None, *, anomalous: bool = False) -> None:
        """Redraw from a run's ``process.log`` and Aimless's unmerged MTZ (either may be ``None``).

        Files already drawn (same path and modification time) are not re-read.
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
        key = (*self._stamp(unmerged_mtz), anomalous)
        if key != self._loaded.get("mtz"):
            self._loaded["mtz"] = key
            if unmerged_mtz is None or not unmerged_mtz.is_file():
                self.rmeas_chart.clear("R_meas by image: scale with Aimless first")
                if unmerged_mtz is not None:
                    notes.append(f"No {unmerged_mtz.name}: re-run Aimless to write the unmerged data.")
            else:
                try:
                    result = rmeas_by_image(unmerged_mtz, anomalous=anomalous)
                    self.rmeas_chart.show_rmeas(result)
                except (OSError, KeyError, ValueError, RuntimeError) as ex:
                    logger.warning("R_meas by image failed: %s", ex)
                    self.rmeas_chart.clear(f"R_meas by image: {ex}")
        self.note.setText(" ".join(notes))
