"""Live integration charts for ProcessingLab: spot profile and resolution bins.

During PROCESS, ProcessingLab prints two machine-readable blocks after every
batch of frames (``PROCESSING OF IMAGES f1 ... f2``)::

    PROFILE_MAP side=15 nphi=9 scale=100      average 3D spot profile:
    # phi=-4                                  one side×side slice per image
       0    0   12   40 …                     offset, values 0…scale
    PROFILE_MAP_END

    RESOLUTION_BINS nshells=9 columns=d_limit_A,observed,…
    # comment lines
         9.99   36   35  285  12.3 …          one row per shell, then "total"
    RESOLUTION_BINS_END

:class:`LogBlockParser` reads them from the output as it streams (the same
formats as ProcessingLab's ``doc/read_profile_maps.py`` and
``doc/resolution_bins.py``). :class:`IntegrationMonitor` turns them into Qt
signals, and :class:`IntegrationChartsWidget` redraws the matplotlib charts
when each block completes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtWidgets import QLabel, QSplitter, QVBoxLayout, QWidget

_BATCH = re.compile(r"PROCESSING OF IMAGES\s+(\d+)\s+\.\.\.\s+(\d+)")


@dataclass(frozen=True)
class BatchRange:
    """A batch of frames PROCESS started (``first``–``last``)."""

    first: int
    last: int


@dataclass(frozen=True)
class ProfileMap:
    """Average 3D spot profile: ``volume[k]`` is the slice at ``phi_offsets[k]`` images.

    Values run from 0 to ``scale`` (the peak).
    """

    volume: np.ndarray
    phi_offsets: tuple[int, ...]
    scale: int


@dataclass(frozen=True)
class ResolutionBins:
    """One resolution table: per-shell rows and the total, keyed by column name."""

    columns: tuple[str, ...]
    shells: tuple[dict[str, float], ...]
    total: dict[str, float] | None = None

    def column(self, name: str) -> np.ndarray:
        """A column over the shells (NaN for XDS's −99 "undefined" markers)."""
        values = np.array([row.get(name, np.nan) for row in self.shells], dtype=float)
        values[values <= -99.0] = np.nan
        return values


ChartEvent = Union[BatchRange, ProfileMap, ResolutionBins]


@dataclass
class LogBlockParser:
    """Incremental parser: :meth:`feed` one output line at a time."""

    _block: str | None = None
    _meta: dict[str, str] = field(default_factory=dict)
    _rows: list[list[str]] = field(default_factory=list)
    _slices: list[tuple[int, list[list[int]]]] = field(default_factory=list)

    def feed(self, line: str) -> list[ChartEvent]:
        """Consume one line; return the events it completes (usually none).

        :param line: One line of ProcessingLab output (without the newline).
        :returns: Completed :class:`BatchRange`, :class:`ProfileMap` or
            :class:`ResolutionBins` events.
        """
        if self._block is None:
            if line.startswith("PROFILE_MAP "):
                self._start("profile", line)
            elif line.startswith("RESOLUTION_BINS "):
                self._start("bins", line)
            elif (m := _BATCH.search(line)) is not None:
                return [BatchRange(int(m.group(1)), int(m.group(2)))]
            return []
        if self._block == "profile":
            return self._feed_profile(line)
        return self._feed_bins(line)

    def _start(self, kind: str, header: str) -> None:
        self._block = kind
        self._meta = dict(tok.split("=", 1) for tok in header.split()[1:] if "=" in tok)
        self._rows, self._slices = [], []

    def _feed_profile(self, line: str) -> list[ChartEvent]:
        if line.strip() == "PROFILE_MAP_END":
            self._block = None
            return [self._profile()] if self._slices else []
        if line.startswith("# phi="):
            self._slices.append((int(line.split("=", 1)[1]), []))
        elif line.strip() and self._slices:
            try:
                self._slices[-1][1].append([int(x) for x in line.split()])
            except ValueError:
                pass
        return []

    def _profile(self) -> ProfileMap:
        side = int(self._meta.get("side", len(self._slices[0][1])))
        volume = np.zeros((len(self._slices), side, side), dtype=float)
        for k, (_phi, rows) in enumerate(self._slices):
            for r, row in enumerate(rows[:side]):
                volume[k, r, :len(row)] = row[:side]
        return ProfileMap(volume=volume, phi_offsets=tuple(phi for phi, _ in self._slices),
                          scale=int(self._meta.get("scale", 100)))

    def _feed_bins(self, line: str) -> list[ChartEvent]:
        if line.strip() == "RESOLUTION_BINS_END":
            self._block = None
            return [self._bins()] if self._rows else []
        if line.strip() and not line.startswith("#"):
            self._rows.append(line.split())
        return []

    def _bins(self) -> ResolutionBins:
        columns = tuple(self._meta.get("columns", "").split(","))
        shells: list[dict[str, float]] = []
        total: dict[str, float] | None = None
        for row in self._rows:
            values: dict[str, float] = {}
            for name, token in zip(columns, row):
                try:
                    values[name] = float(token)
                except ValueError:
                    values[name] = np.nan
            if row and row[0] == "total":
                total = values
            else:
                shells.append(values)
        return ResolutionBins(columns=columns, shells=tuple(shells), total=total)


def read_log_charts(path: Path) -> tuple[ProfileMap | None, ResolutionBins | None, BatchRange | None]:
    """The last profile, resolution table and batch in a finished run's log.

    :param path: ``process.log``.
    :returns: ``(profile, bins, batch)``; ``None`` for any the log lacks.
    """
    parser = LogBlockParser()
    profile = bins = batch = None
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None, None, None
    for line in lines:
        for event in parser.feed(line):
            if isinstance(event, ProfileMap):
                profile = event
            elif isinstance(event, ResolutionBins):
                bins = event
            else:
                batch = event
    return profile, bins, batch


class IntegrationMonitor(QObject):
    """Turns streamed ProcessingLab output into chart signals."""

    batchStarted = Signal(int, int)
    profileReady = Signal(object)  # ProfileMap
    binsReady = Signal(object)  # ResolutionBins

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._parser = LogBlockParser()

    @Slot(str)
    def feed(self, line: str) -> None:
        """Consume one output line; emit a signal for each completed block."""
        for event in self._parser.feed(line):
            if isinstance(event, BatchRange):
                self.batchStarted.emit(event.first, event.last)
            elif isinstance(event, ProfileMap):
                self.profileReady.emit(event)
            else:
                self.binsReady.emit(event)

    def reset(self) -> None:
        """Forget any half-read block (a new run is starting)."""
        self._parser = LogBlockParser()


class ProfileMapCanvas(FigureCanvasQTAgg):
    """The average spot profile, one panel per φ slice."""

    def __init__(self, parent: QWidget | None = None) -> None:
        self.figure = Figure(figsize=(9, 1.8), layout="constrained")
        super().__init__(self.figure)
        self.setParent(parent)
        self.clear("Average spot profile: waiting for the first batch")

    def clear(self, message: str = "") -> None:
        """Show an empty chart with *message*."""
        self.figure.clear()
        self.figure.text(0.5, 0.5, message, ha="center", va="center", color="0.4")
        self.draw_idle()

    def show_profile(self, profile: ProfileMap, title: str = "") -> None:
        """Draw *profile* (slices left to right, φ offset above each)."""
        self.figure.clear()
        n = profile.volume.shape[0]
        axes = self.figure.subplots(1, n, sharey=True, squeeze=False)[0]
        for ax, sl, phi in zip(axes, profile.volume, profile.phi_offsets):
            ax.imshow(sl, cmap="magma", vmin=0, vmax=profile.scale, origin="upper")
            ax.set_title(f"φ {phi:+d}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        if title:
            self.figure.suptitle(title, fontsize=9)
        self.draw_idle()


class ResolutionBinsCanvas(FigureCanvasQTAgg):
    """Completeness and I/σ, and CC½ and R_meas, against resolution."""

    def __init__(self, parent: QWidget | None = None) -> None:
        self.figure = Figure(figsize=(9, 2.6), layout="constrained")
        super().__init__(self.figure)
        self.setParent(parent)
        self.clear("Resolution bins: waiting for the first batch")

    def clear(self, message: str = "") -> None:
        """Show an empty chart with *message*."""
        self.figure.clear()
        self.figure.text(0.5, 0.5, message, ha="center", va="center", color="0.4")
        self.draw_idle()

    def show_bins(self, bins: ResolutionBins, title: str = "") -> None:
        """Draw *bins*: resolution runs high to low left to right."""
        self.figure.clear()
        left, right = self.figure.subplots(1, 2)
        d = bins.column("d_limit_A")
        if d.size == 0:
            self.clear("Resolution bins: no shells")
            return
        lo, hi = np.nanmax(d), np.nanmin(d)

        left.plot(d, bins.column("completeness_pct"), "o-", color="C0", label="completeness %")
        left.set_xlim(lo, hi)
        left.set_xlabel("d (Å)")
        left.set_ylabel("completeness %")
        twin = left.twinx()
        twin.plot(d, bins.column("mean_isigma"), "s--", color="C1", label="I/σ")
        twin.set_ylabel("I/σ")
        handles = left.get_lines() + twin.get_lines()
        left.legend(handles, [h.get_label() for h in handles], loc="best", fontsize=7, framealpha=0.6)

        right.plot(d, bins.column("cc_half_pct"), "o-", color="C2", label="CC½ %")
        right.set_xlim(lo, hi)
        right.set_ylim(-5, 105)
        right.set_xlabel("d (Å)")
        right.set_ylabel("CC½ %")
        twin2 = right.twinx()
        twin2.plot(d, bins.column("rmeas_pct"), "s--", color="C3", label="R_meas %")
        twin2.set_ylabel("R_meas %")
        handles = right.get_lines() + twin2.get_lines()
        right.legend(handles, [h.get_label() for h in handles], loc="best", fontsize=7, framealpha=0.6)
        if title:
            self.figure.suptitle(title, fontsize=9)
        self.draw_idle()


class StrategyCanvas(FigureCanvasQTAgg):
    """Best completeness against total rotation (``RunStep.STRATEGY``)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        self.figure = Figure(figsize=(6, 2.8), layout="constrained")
        super().__init__(self.figure)
        self.setParent(parent)
        self.clear("Completeness against rotation: calculate a strategy")

    def clear(self, message: str = "") -> None:
        """An empty chart with *message*."""
        self.figure.clear()
        self.figure.text(0.5, 0.5, message, ha="center", va="center", color="0.4")
        self.draw_idle()

    def show_curve(self, curve, *, target_pct: float, anomalous: bool,
                   chosen: tuple[float, float] | None = None, title: str = "") -> None:
        """Plot ``(rotation, start, completeness %, anomalous %)`` points.

        :param target_pct: The completeness asked for (a dashed line).
        :param anomalous: Whether the target is anomalous completeness.
        :param chosen: ``(rotation, completeness %)`` of the chosen segment.
        """
        self.figure.clear()
        ax = self.figure.subplots()
        if curve:
            rot = [p[0] for p in curve]
            ax.plot(rot, [p[2] for p in curve], color="#4C78A8", label="unique")
            ax.plot(rot, [p[3] for p in curve], color="#F58518", label="anomalous pairs")
        ax.axhline(target_pct, color="0.4", linestyle="--", linewidth=1,
                   label=f"target {target_pct:g}%" + (" (anomalous)" if anomalous else ""))
        if chosen is not None:
            ax.plot([chosen[0]], [chosen[1]], "o", color="#E45756", label=f"{chosen[0]:g}°")
        ax.set_xlabel("Total rotation (°)")
        ax.set_ylabel("Completeness (%)")
        ax.set_ylim(0, 101)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
        if title:
            ax.set_title(title, fontsize=9)
        self.draw_idle()


class IntegrationChartsWidget(QWidget):
    """Live profile and resolution charts, fed with ProcessingLab output lines."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.monitor = IntegrationMonitor(self)
        self._batch = ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.batch_label = QLabel("Integration charts update after every batch of frames.")
        layout.addWidget(self.batch_label)
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.profile_canvas = ProfileMapCanvas(self)
        self.bins_canvas = ResolutionBinsCanvas(self)
        splitter.addWidget(self.profile_canvas)
        splitter.addWidget(self.bins_canvas)
        splitter.setSizes([170, 260])
        layout.addWidget(splitter, stretch=1)
        self.monitor.batchStarted.connect(self._on_batch)
        self.monitor.profileReady.connect(self._on_profile)
        self.monitor.binsReady.connect(self._on_bins)

    @Slot(str)
    def feed(self, line: str) -> None:
        """One line of ProcessingLab output (connect a worker's line signal here)."""
        self.monitor.feed(line)

    def reset(self) -> None:
        """Clear the charts for a new integration."""
        self.monitor.reset()
        self._batch = ""
        self.batch_label.setText("Integrating: charts appear after the first batch of frames.")
        self.profile_canvas.clear("Average spot profile: waiting for the first batch")
        self.bins_canvas.clear("Resolution bins: waiting for the first batch")

    def load_log(self, path: Path) -> bool:
        """Show the last charts of a finished run.

        :param path: Its ``process.log``.
        :returns: Whether the log had any chart data.
        """
        profile, bins, batch = read_log_charts(path)
        if batch is not None:
            self._on_batch(batch.first, batch.last)
        if profile is not None:
            self._on_profile(profile)
        if bins is not None:
            self._on_bins(bins)
        return profile is not None or bins is not None

    def _on_batch(self, first: int, last: int) -> None:
        self._batch = f"frames {first}–{last}"
        self.batch_label.setText(f"Integrating {self._batch}…")

    def _on_profile(self, profile: ProfileMap) -> None:
        self.profile_canvas.show_profile(profile, f"Average spot profile (to {self._batch})"
                                         if self._batch else "Average spot profile")

    def _on_bins(self, bins: ResolutionBins) -> None:
        total = bins.total or {}
        summary = ""
        if total:
            summary = (f": {total.get('observed', float('nan')):.0f} observed, "
                       f"completeness {total.get('completeness_pct', float('nan')):.1f}%, "
                       f"I/σ {total.get('mean_isigma', float('nan')):.2f}, "
                       f"CC½ {total.get('cc_half_pct', float('nan')):.1f}%")
        self.bins_canvas.show_bins(bins, f"Resolution bins, unscaled (to {self._batch}){summary}"
                                   if self._batch else f"Resolution bins, unscaled{summary}")
        if self._batch:
            self.batch_label.setText(f"Integrated {self._batch}{summary}")
