"""Matplotlib ``Summary vs resolution`` plot (DIALS integrate table)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from imosflm.dials_summary import (
    ResolutionBinSummary,
    find_integrate_log,
    parse_summary_vs_resolution_log,
    parse_summary_vs_resolution_text,
)


def plot_summary_vs_resolution(
    bins: list[ResolutionBinSummary],
    *,
    figure: Figure | None = None,
    title: str = "Summary vs resolution",
) -> Figure:
    """Draw a 2×2 matplotlib summary of DIALS integrate resolution bins.

    Panels: I/σI (sum & prf), CC prf, RMSD XY, reflection counts.
    High resolution (small ``d_min``) is on the left.
    """
    if not bins:
        raise ValueError("No resolution bins to plot")

    # Sort high-res → low-res for left→right axis.
    ordered = sorted(bins, key=lambda b: b.d_min)
    d_min = [b.d_min for b in ordered]
    i_sum = [b.i_sigi_sum for b in ordered]
    i_prf = [b.i_sigi_prf for b in ordered]
    cc = [b.cc_prf for b in ordered]
    rmsd = [b.rmsd_xy for b in ordered]
    n_full = [b.n_full for b in ordered]
    n_sum = [b.n_sum for b in ordered]
    n_prf = [b.n_prf for b in ordered]

    fig = figure or Figure(figsize=(10, 7), constrained_layout=True)
    fig.clear()
    axes = fig.subplots(2, 2)
    fig.suptitle(title)

    ax = axes[0, 0]
    ax.plot(d_min, i_sum, "o-", label="I/σI (sum)", color="#4C78A8")
    ax.plot(d_min, i_prf, "s-", label="I/σI (prf)", color="#F58518")
    ax.set_ylabel("I / σ(I)")
    ax.set_xlabel("d min (Å)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(d_min, cc, "o-", color="#54A24B")
    ax.set_ylabel("CC prf")
    ax.set_xlabel("d min (Å)")
    ax.set_ylim(0.0, max(1.05, max(cc) * 1.05 if cc else 1.05))
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(d_min, rmsd, "o-", color="#E45756")
    ax.set_ylabel("RMSD XY (px)")
    ax.set_xlabel("d min (Å)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(d_min, n_full, "o-", label="# full", color="#4C78A8")
    ax.plot(d_min, n_sum, "^-", label="# sum", color="#72B7B2")
    ax.plot(d_min, n_prf, "s-", label="# prf", color="#F58518")
    ax.set_ylabel("Reflections")
    ax.set_xlabel("d min (Å)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    for ax in axes.ravel():
        ax.invert_xaxis()  # high-res (small d) on the left

    return fig


def show_summary_vs_resolution(
    bins: list[ResolutionBinSummary],
    *,
    title: str = "Summary vs resolution",
) -> None:
    """Open a standalone matplotlib window for *bins*."""
    fig = plot_summary_vs_resolution(bins, title=title)
    fig.show()
    plt.show(block=False)


class SummaryVsResolutionDialog(QDialog):
    """Qt dialog embedding the Summary vs resolution matplotlib figure."""

    def __init__(
        self,
        bins: list[ResolutionBinSummary],
        parent: QWidget | None = None,
        *,
        title: str = "Summary vs resolution",
        source: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(900, 680)
        self._bins = list(bins)

        layout = QVBoxLayout(self)
        if source:
            caption = QLabel(source)
            caption.setWordWrap(True)
            caption.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            layout.addWidget(caption)

        self._figure = Figure(figsize=(9.5, 6.5), constrained_layout=True)
        plot_summary_vs_resolution(self._bins, figure=self._figure, title=title)
        self._canvas = FigureCanvas(self._figure)
        layout.addWidget(self._canvas, stretch=1)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save PNG…")
        save_btn.clicked.connect(self._on_save_png)
        btn_row.addWidget(save_btn)
        btn_row.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        close = buttons.button(QDialogButtonBox.StandardButton.Close)
        if close is not None:
            close.clicked.connect(self.reject)
        btn_row.addWidget(buttons)
        layout.addLayout(btn_row)

    def _on_save_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Summary vs resolution",
            "summary_vs_resolution.png",
            "PNG (*.png)",
        )
        if path:
            self._figure.savefig(path, dpi=150)


def open_summary_vs_resolution_from_log(
    log_path: Path | str,
    parent: QWidget | None = None,
) -> SummaryVsResolutionDialog:
    """Parse *log_path* and show the Summary vs resolution dialog."""
    path = Path(log_path).expanduser()
    bins = parse_summary_vs_resolution_log(path)
    if not bins:
        raise ValueError(f"No 'Summary vs resolution' table in {path}")
    dialog = SummaryVsResolutionDialog(
        bins,
        parent=parent,
        source=str(path),
    )
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    return dialog


def open_summary_vs_resolution_from_work_dir(
    work_dir: Path | str,
    parent: QWidget | None = None,
) -> SummaryVsResolutionDialog:
    """Locate ``dials.integrate.log`` under *work_dir* and show the plot."""
    log = find_integrate_log(work_dir)
    if log is None:
        raise FileNotFoundError(
            f"No dials.integrate.log in {Path(work_dir).expanduser()}"
        )
    return open_summary_vs_resolution_from_log(log, parent=parent)


__all__ = [
    "SummaryVsResolutionDialog",
    "open_summary_vs_resolution_from_log",
    "open_summary_vs_resolution_from_work_dir",
    "parse_summary_vs_resolution_log",
    "parse_summary_vs_resolution_text",
    "plot_summary_vs_resolution",
    "show_summary_vs_resolution",
]
