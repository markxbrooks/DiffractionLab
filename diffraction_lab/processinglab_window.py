"""ProcessingLab Processing window: spots → index → integrate → symmetry → scale.

A processing window built only on ProcessingLab and CCP4, beside the
DIALS / MOSFLM Processing window (:mod:`diffraction_lab.processing_wizard`):

1. **Setup**: the sweep, the session folder, the tools found, pipeline status.
2. **Find spots**: ProcessingLab FINDSPOTS over a frame wedge; spots can be
   overlaid on the image.
3. **Index**: ProcessingLab DPS AUTOINDEX; every lattice solution with its
   penalty, cell and possible space groups. Another solution can be chosen.
4. **Integrate**: ProcessingLab integration in an initial space group chosen
   from the solution; re-integration from the refined cell and detector
   parameters (``--from``). The average spot profile and the resolution
   bins are charted live, after every batch of frames
   (:mod:`diffraction_lab.processinglab_charts`).
5. **Symmetry**: CCP4 Pointless (Laue / space group, reindexing, L-test for
   twinning).
6. **Scale**: CCP4 Aimless (scale and merge).

Each stage is a ProcessingLab run in the session folder
(:mod:`diffraction_lab.processinglab_stages`); Pointless and Aimless write
into its ``04-ccp4`` folder.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt, QThread, QUrl
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from diffraction_lab.aimless_runner import AimlessRunResult, AimlessSummary, find_aimless, run_aimless
from diffraction_lab.pointless_runner import (
    PointlessRunResult,
    find_pointless,
    is_identity_reindex,
    run_pointless,
)
from diffraction_lab.processing_sweep import Frames, ProcessingSweep
from diffraction_lab.processinglab_charts import IntegrationChartsWidget
from diffraction_lab.processinglab_client import find_integrationlab
from diffraction_lab.processinglab_panel import (
    CallWorker,
    default_work_parent,
    format_record,
    parse_cell_text,
)
from diffraction_lab.processinglab_run import REFINABLE_PARAMETERS, RunRecord, rerun_options, run_processinglab
from diffraction_lab.processinglab_stages import (
    STAGE_CCP4,
    STAGE_INTEGRATE,
    IndexResult,
    LatticeSolution,
    autoindex,
    find_spots,
    integrate,
    new_session_dir,
    read_spt,
    scan_of,
    spot_overlay,
    stage_dir,
)
from diffraction_lab.reflection_viewer_launch import launch_reflection_viewer

logger = logging.getLogger(__name__)

_REFINE_LABELS = {
    "orientation": "Orientation",
    "cell": "Unit cell",
    "distance": "Distance",
    "beam": "Beam centre",
    "tilt_twist": "Tilt / twist",
    "mosaicity": "Mosaicity",
}
_SOLUTION_COLUMNS = ("No", "Lattice", "Penalty", "a", "b", "c", "α", "β", "γ", "Metric dev %", "Space groups")


def format_pointless(p: PointlessRunResult) -> str:
    """HTML summary of a Pointless run: solution, confidences, twinning."""
    kind = p.solution_kind or "space group"
    lines = [
        f"<b>Best solution:</b> {kind} <b>{p.space_group or '—'}</b>",
        f"Reindex operator: {p.reindex_operator or '—'}",
        "Space-group confidence: " + ("—" if p.confidence is None else f"{p.confidence:.3f}")
        + ("" if p.laue_confidence is None else f"; Laue-group confidence: {p.laue_confidence:.3f}"),
    ]
    if kind == "point group":
        lines.append("<i>Only the point group was determined: too few axial reflections to test "
                     "screw axes. Set the space group yourself, or collect more data.</i>")
    if not is_identity_reindex(p.reindex_operator):
        lines.append(f"<i>Pointless reindexes the data ({p.reindex_operator}); re-integrating in its "
                     "space group would need that reindexing, which ProcessingLab does not apply.</i>")
    t = p.twinning
    if t is not None:
        colour = "#b00" if t.suggests_twinning else "#070"
        lines.append(
            "<b>Twinning (L-test):</b> ⟨|L|⟩ " + ("—" if t.mean_abs_l is None else f"{t.mean_abs_l:.3f}")
            + " (0.5 untwinned, 0.375 perfect twin), ⟨L²⟩ " + ("—" if t.mean_l2 is None else f"{t.mean_l2:.3f}")
            + ", twin fraction ≈ " + ("—" if t.twin_fraction is None else f"{t.twin_fraction:.3f}")
            + (f"<br><span style='color:{colour}'>{t.verdict}</span>" if t.verdict else "")
        )
    return "<br>".join(lines)


def format_aimless(s: AimlessSummary) -> str:
    """HTML table of Aimless's overall / inner / outer statistics."""
    rows = [("Low resolution (Å)", "Low resolution limit", ".2f"),
            ("High resolution (Å)", "High resolution limit", ".2f"),
            ("R<sub>merge</sub>", "Rmerge (all I+ and I-)", ".3f"),
            ("R<sub>meas</sub>", "Rmeas (all I+ & I-)", ".3f"),
            ("R<sub>pim</sub>", "Rpim (all I+ & I-)", ".3f"),
            ("Observations", "Total number of observations", ".0f"),
            ("Unique", "Total number unique", ".0f"),
            ("⟨I/σ(I)⟩", "Mean((I)/sd(I))", ".1f"),
            ("CC½", "Mn(I) half-set correlation CC(1/2)", ".3f"),
            ("Completeness (%)", "Completeness", ".1f"),
            ("Multiplicity", "Multiplicity", ".1f")]

    def cell(label: str, shell: str, fmt: str) -> str:
        v = s.value(label, shell)
        return "—" if v is None else format(v, fmt)

    body = "".join(
        f"<tr><td>{name}</td>" + "".join(f"<td align=right>&nbsp;{cell(label, sh, f)}</td>"
                                         for sh in ("overall", "inner", "outer")) + "</tr>"
        for name, label, f in rows if label in s.rows
    )
    html = (f"<b>Space group</b> {s.space_group or '—'}"
            + ("" if s.cell is None else "; cell " + " ".join(f"{v:.2f}" for v in s.cell))
            + f"<table><tr><th></th><th>Overall</th><th>Inner</th><th>Outer</th></tr>{body}</table>")
    if s.resolution_estimates:
        html += "<b>Resolution estimates:</b><br>" + "<br>".join(
            f"{crit}: {lim:.2f} Å" for crit, lim in s.resolution_estimates.items())
    return html


@dataclass
class _Session:
    """What this window has produced so far (latest of each stage)."""

    folder: Path | None = None
    spots: RunRecord | None = None
    index: IndexResult | None = None
    integrated: RunRecord | None = None
    pointless: PointlessRunResult | None = None
    aimless: AimlessRunResult | None = None


class ProcessingLabWindow(QDialog):
    """Processing with ProcessingLab, Pointless and Aimless only."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Processing — ProcessingLab")
        self.setMinimumSize(860, 640)
        self.setModal(False)
        self._sweep: ProcessingSweep | None = None
        self._s = _Session()
        self._thread: QThread | None = None
        self._worker: CallWorker | None = None
        self._overlay_shown = False  # whether this window drew the spot overlay
        self._slider_connected = False
        self._build_ui()
        self.refresh_dataset()

    # ================================================================ UI

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(50000)
        self.log_view.setFont(QFont("Menlo", 10))
        splitter.addWidget(self.log_view)
        splitter.setSizes([520, 160])
        layout.addWidget(splitter, stretch=1)
        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self._build_setup_tab()
        self._build_spots_tab()
        self._build_index_tab()
        self._build_integrate_tab()
        self._build_symmetry_tab()
        self._build_scale_tab()

    def _build_setup_tab(self) -> None:
        page = QWidget()
        v = QVBoxLayout(page)
        box = QGroupBox("Sweep and session")
        form = QFormLayout(box)
        self.image_label = QLabel("(no image loaded)")
        self.image_label.setWordWrap(True)
        form.addRow("Image", self.image_label)
        self.sweep_first = QSpinBox()
        self.sweep_last = QSpinBox()
        for spin in (self.sweep_first, self.sweep_last):
            spin.setRange(1, 999999)
        form.addRow("Frames to process", _row(self.sweep_first, QLabel("to"), self.sweep_last))
        self.sweep_label = QLabel("")
        form.addRow("Sweep on disk", self.sweep_label)
        self.work_edit = QLineEdit()
        self.work_edit.setToolTip("Sessions are created here as session_YYYYMMDD_HHMMSS")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._on_browse_work)
        form.addRow("Work folder", _row(self.work_edit, browse, stretch_first=True))
        self.session_label = QLabel("(none yet: created by the first step)")
        self.session_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        new_btn = QPushButton("New session")
        new_btn.clicked.connect(self._on_new_session)
        reveal_btn = QPushButton("Reveal")
        reveal_btn.clicked.connect(lambda: _open(self._s.folder))
        form.addRow("Session", _row(self.session_label, new_btn, reveal_btn, stretch_first=True))
        v.addWidget(box)

        tools = QGroupBox("Programs")
        tools_form = QFormLayout(tools)
        self.tool_labels = {name: QLabel() for name in ("ProcessingLab", "Pointless", "Aimless")}
        for name, label in self.tool_labels.items():
            tools_form.addRow(name, label)
        v.addWidget(tools)

        status = QGroupBox("Pipeline")
        sv = QVBoxLayout(status)
        self.pipeline_labels = {}
        for key, text in (("spots", "Find spots (ProcessingLab)"), ("index", "Index (ProcessingLab)"),
                          ("integrate", "Integrate (ProcessingLab)"), ("symmetry", "Symmetry (Pointless)"),
                          ("scale", "Scale (Aimless)")):
            label = QLabel(f"○ {text}")
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.pipeline_labels[key] = (label, text)
            sv.addWidget(label)
        self.run_all_btn = QPushButton("Run all steps with these settings")
        self.run_all_btn.setToolTip("Find spots → index (suggested solution) → integrate (its first space group) "
                                    "→ Pointless → Aimless")
        self.run_all_btn.clicked.connect(self._on_run_all)
        sv.addWidget(self.run_all_btn)
        v.addWidget(status)
        v.addStretch(1)
        self.tabs.addTab(page, "1. Setup")

    def _build_spots_tab(self) -> None:
        page = QWidget()
        v = QVBoxLayout(page)
        box = QGroupBox("Spot finding (ProcessingLab FINDSPOTS)")
        form = QFormLayout(box)
        self.spot_first = QSpinBox()
        self.spot_last = QSpinBox()
        for spin in (self.spot_first, self.spot_last):
            spin.setRange(1, 999999)
        form.addRow("Frames", _row(self.spot_first, QLabel("to"), self.spot_last))
        self.spot_threshold = QDoubleSpinBox()
        self.spot_threshold.setRange(1.0, 100.0)
        self.spot_threshold.setValue(5.0)
        form.addRow("Threshold (σ)", self.spot_threshold)
        self.spot_max = QSpinBox()
        self.spot_max.setRange(50, 100000)
        self.spot_max.setValue(1500)
        form.addRow("Maximum spots", self.spot_max)
        self.find_spots_btn = QPushButton("Find spots")
        self.find_spots_btn.clicked.connect(self._on_find_spots)
        self.show_spots_check = QCheckBox("Show spots on the image (current frame)")
        self.show_spots_check.toggled.connect(lambda _on: self._update_spot_overlay())
        form.addRow(_row(self.find_spots_btn, self.show_spots_check))
        v.addWidget(box)
        self.spots_label = QLabel("")
        self.spots_label.setWordWrap(True)
        v.addWidget(self.spots_label)
        v.addStretch(1)
        self.tabs.addTab(page, "2. Find spots")

    def _build_index_tab(self) -> None:
        page = QWidget()
        v = QVBoxLayout(page)
        box = QGroupBox("Autoindexing (ProcessingLab DPS)")
        form = QFormLayout(box)
        self.index_threshold = QDoubleSpinBox()
        self.index_threshold.setRange(1.0, 200.0)
        self.index_threshold.setValue(15.0)
        form.addRow("Spot I/σ threshold", self.index_threshold)
        self.index_dmax = QDoubleSpinBox()
        self.index_dmax.setRange(0.0, 2000.0)
        self.index_dmax.setSpecialValueText("200 Å (default)")
        self.index_dmax.setSuffix(" Å")
        self.index_dmax.setToolTip("Longest cell edge searched; raise it for very long axes")
        form.addRow("Maximum cell edge", self.index_dmax)
        self.index_btn = QPushButton("Index")
        self.index_btn.clicked.connect(lambda: self._on_index(None))
        form.addRow(self.index_btn)
        v.addWidget(box)

        self.solutions_table = QTableWidget(0, len(_SOLUTION_COLUMNS))
        self.solutions_table.setHorizontalHeaderLabels(_SOLUTION_COLUMNS)
        self.solutions_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.solutions_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.solutions_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.solutions_table.horizontalHeader().setSectionResizeMode(
            len(_SOLUTION_COLUMNS) - 1, QHeaderView.ResizeMode.Stretch)
        self.solutions_table.itemSelectionChanged.connect(self._update_enabled)
        v.addWidget(self.solutions_table, stretch=1)
        self.index_label = QLabel("")
        self.index_label.setWordWrap(True)
        v.addWidget(self.index_label)
        self.reindex_btn = QPushButton("Use the highlighted solution (re-index)")
        self.reindex_btn.setToolTip("Re-run AUTOINDEX choosing this lattice (AUTOINDEX CHARACTER n)")
        self.reindex_btn.clicked.connect(self._on_reindex_selected)
        v.addWidget(self.reindex_btn)
        self.tabs.addTab(page, "3. Index")

    def _build_integrate_tab(self) -> None:
        controls = QWidget()
        v = QVBoxLayout(controls)
        box = QGroupBox("Integration (ProcessingLab)")
        form = QFormLayout(box)
        self.space_group_combo = QComboBox()
        self.space_group_combo.setEditable(True)
        self.space_group_combo.setToolTip("Initial space group: any group of the chosen lattice. Pointless "
                                          "(Symmetry tab) tests it after integration.")
        form.addRow("Initial space group", self.space_group_combo)
        self.integrate_btn = QPushButton("Integrate")
        self.integrate_btn.clicked.connect(self._on_integrate)
        form.addRow(self.integrate_btn)
        v.addWidget(box)

        result = QGroupBox("Result")
        rv = QVBoxLayout(result)
        self.integrate_label = QLabel("Not integrated yet.")
        self.integrate_label.setWordWrap(True)
        self.integrate_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        rv.addWidget(self.integrate_label)
        self.report_btn = QPushButton("Open report")
        self.report_btn.clicked.connect(lambda: _open(self._s.integrated.directory / "process.md"
                                                      if self._s.integrated else None))
        self.mtz_view_btn = QPushButton("Open MTZ in viewer")
        self.mtz_view_btn.clicked.connect(lambda: self._view_mtz(self._s.integrated.mtz
                                                                if self._s.integrated else None))
        rv.addLayout(_row(self.report_btn, self.mtz_view_btn))
        v.addWidget(result)

        rerun = QGroupBox("Re-integrate with the refined unit cell and detector parameters")
        rr = QVBoxLayout(rerun)
        checks = QHBoxLayout()
        self.refine_checks: dict[str, QCheckBox] = {}
        for name in REFINABLE_PARAMETERS:
            box_ = QCheckBox(_REFINE_LABELS[name])
            box_.setChecked(name in ("orientation", "mosaicity"))
            self.refine_checks[name] = box_
            checks.addWidget(box_)
        checks.addStretch(1)
        rr.addWidget(QLabel("Refine again (unticked parameters stay at their refined values):"))
        rr.addLayout(checks)
        rform = QFormLayout()
        self.rerun_space_group = QLineEdit()
        self.rerun_space_group.setPlaceholderText("optional; filled from Pointless when it is safe")
        rform.addRow("Space group", self.rerun_space_group)
        self.rerun_cell = QLineEdit()
        self.rerun_cell.setPlaceholderText("optional a b c α β γ (default: the refined cell)")
        rform.addRow("Unit cell", self.rerun_cell)
        rr.addLayout(rform)
        self.rerun_btn = QPushButton("Re-integrate")
        self.rerun_btn.clicked.connect(self._on_reintegrate)
        rr.addWidget(self.rerun_btn)
        v.addWidget(rerun)
        v.addStretch(1)

        # Controls above, live charts (profile, resolution bins) below.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        self.charts = IntegrationChartsWidget()
        page = QSplitter(Qt.Orientation.Vertical)
        page.addWidget(scroll)
        page.addWidget(self.charts)
        page.setSizes([300, 380])
        self.tabs.addTab(page, "4. Integrate")

    def _build_symmetry_tab(self) -> None:
        page = QWidget()
        v = QVBoxLayout(page)
        box = QGroupBox("Space group and twinning (CCP4 Pointless)")
        bv = QVBoxLayout(box)
        bv.addWidget(QLabel("Tests the Laue group from symmetry-related intensities, screw axes from "
                            "systematic absences, and twinning (L-test), on the integrated reflections."))
        self.pointless_btn = QPushButton("Run Pointless")
        self.pointless_btn.clicked.connect(self._on_pointless)
        bv.addWidget(self.pointless_btn)
        self.pointless_label = QLabel("")
        self.pointless_label.setWordWrap(True)
        self.pointless_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        bv.addWidget(self.pointless_label)
        self.use_sg_btn = QPushButton("Re-integrate in this space group…")
        self.use_sg_btn.clicked.connect(self._on_use_pointless_space_group)
        self.pointless_log_btn = QPushButton("Open log")
        self.pointless_log_btn.clicked.connect(lambda: _open(self._s.pointless.log if self._s.pointless else None))
        bv.addLayout(_row(self.use_sg_btn, self.pointless_log_btn))
        v.addWidget(box)
        v.addStretch(1)
        self.tabs.addTab(page, "5. Symmetry")

    def _build_scale_tab(self) -> None:
        page = QWidget()
        v = QVBoxLayout(page)
        box = QGroupBox("Scaling and merging (CCP4 Aimless)")
        form = QFormLayout(box)
        self.aimless_dmin = QDoubleSpinBox()
        self.aimless_dmin.setRange(0.0, 20.0)
        self.aimless_dmin.setDecimals(2)
        self.aimless_dmin.setSpecialValueText("all data")
        self.aimless_dmin.setSuffix(" Å")
        form.addRow("High-resolution cutoff", self.aimless_dmin)
        self.aimless_anomalous = QCheckBox("Anomalous (keep I+ and I− separate)")
        form.addRow(self.aimless_anomalous)
        self.aimless_btn = QPushButton("Run Aimless")
        self.aimless_btn.clicked.connect(self._on_aimless)
        form.addRow(self.aimless_btn)
        v.addWidget(box)
        self.aimless_label = QLabel("")
        self.aimless_label.setWordWrap(True)
        self.aimless_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.aimless_label)
        self.merged_view_btn = QPushButton("Open merged MTZ in viewer")
        self.merged_view_btn.clicked.connect(lambda: self._view_mtz(self._s.aimless.hklout
                                                                   if self._s.aimless else None))
        self.aimless_log_btn = QPushButton("Open log")
        self.aimless_log_btn.clicked.connect(lambda: _open(self._s.aimless.log if self._s.aimless else None))
        v.addLayout(_row(self.merged_view_btn, self.aimless_log_btn))
        v.addStretch(1)
        self.tabs.addTab(page, "6. Scale")

    # ========================================================= dataset

    def refresh_dataset(self) -> None:
        """Read the open image and frame count from the main window."""
        app = self.parentWidget()
        current = getattr(app, "current_file", None) if app is not None else None
        self.set_dataset(Path(current) if current else None)
        slider = getattr(app, "frame_slider", None)
        if slider is not None and not self._slider_connected:
            slider.valueChanged.connect(self._update_spot_overlay)
            self._slider_connected = True

    def set_dataset(self, image: Path | None) -> None:
        """Point the window at a sweep (any frame, or an HDF5 master).

        The frame range is the whole sweep as ProcessingLab sees it (every
        data file of a master; the image viewer counts one at a time).

        :param image: The image, or ``None``.
        """
        new = Path(image).expanduser().resolve() if image else None
        self.image_label.setText(str(new) if new else "(no image loaded)")
        if new != (self._sweep.image if self._sweep else None):
            # A different sweep: a fresh session, ranges over the whole sweep.
            self._sweep = ProcessingSweep.from_image(new) if new else None
            if new is not None and self._sweep is None:
                self._sweep = ProcessingSweep(new, Frames(1, 1))  # unreadable master: let ProcessingLab report it
            frames = self._sweep.frames if self._sweep else Frames(1, 1)
            for spin in (self.sweep_first, self.sweep_last, self.spot_first, self.spot_last):
                spin.setRange(frames.start, frames.end)
            self._show_frames(frames, self._sweep.spot_wedge() if self._sweep else frames)
            self.sweep_label.setText(f"frames {frames} ({frames.count} images)"
                                     + (f" in {len(self._sweep.data_files)} data files"
                                        if self._sweep and self._sweep.data_files else ""))
            self._s = _Session()
            self._show_all()
        if new is not None and not self.work_edit.text().strip():
            self.work_edit.setText(str(default_work_parent(new)))
        self._update_tools()
        self._update_enabled()

    def _show_frames(self, integrate: Frames, spots: Frames) -> None:
        """Put frames into the Setup and Find-spots range boxes."""
        self.sweep_first.setValue(integrate.start)
        self.sweep_last.setValue(integrate.end)
        self.spot_first.setValue(spots.start)
        self.spot_last.setValue(spots.end)

    def _integrate_frames(self) -> Frames:
        """The frames to integrate (Setup), always passed explicitly (``--range``)."""
        return Frames.between(self.sweep_first.value(), self.sweep_last.value())

    def _spot_frames(self) -> Frames:
        """The spot-search wedge (Find spots)."""
        return Frames.between(self.spot_first.value(), self.spot_last.value())

    def _session(self) -> Path:
        if self._s.folder is None:
            parent = Path(self.work_edit.text().strip() or default_work_parent(self._sweep.image))
            self._s.folder = new_session_dir(parent)
            self.session_label.setText(str(self._s.folder))
        return self._s.folder

    # ========================================================= actions

    def _on_browse_work(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Work folder", self.work_edit.text())
        if folder:
            self.work_edit.setText(folder)

    def _on_new_session(self) -> None:
        self._s = _Session()
        self.session_label.setText("(none yet: created by the first step)")
        self._show_all()

    def _on_find_spots(self) -> None:
        sweep, session, wedge = self._sweep, self._session(), self._spot_frames()
        threshold, max_spots = self.spot_threshold.value(), self.spot_max.value()
        self._start("Finding spots…", lambda emit: find_spots(sweep, session, wedge, threshold=threshold,
                                                              max_spots=max_spots, on_line=emit),
                    self._spots_done)

    def _on_index(self, character: int | None) -> None:
        spots = self._s.spots
        if spots is None or "spot_list" not in spots.outputs:
            return
        sweep, session = self._sweep, self._session()
        threshold = self.index_threshold.value()
        dmax = self.index_dmax.value() or None
        spt = spots.outputs["spot_list"]
        what = "Indexing…" if character is None else f"Re-indexing with lattice character {character}…"
        self._start(what, lambda emit: autoindex(sweep, session, spt, threshold=threshold, dmax=dmax,
                                                 character=character, on_line=emit), self._index_done)

    def _on_reindex_selected(self) -> None:
        solution = self._highlighted_solution()
        if solution is not None:
            self._on_index(solution.character)

    def _on_integrate(self) -> None:
        index = self._s.index
        space_group = self.space_group_combo.currentText().strip()
        if index is None or not space_group:
            return
        sweep, session, frames = self._sweep, self._session(), self._integrate_frames()
        self._start(f"Integrating frames {frames} in {space_group}…",
                    lambda emit: integrate(sweep, session, index, space_group, frames=frames, on_line=emit),
                    self._integrate_done, charts=True)

    def _on_reintegrate(self) -> None:
        record = self._s.integrated
        if record is None:
            return
        refine = tuple(n for n, box in self.refine_checks.items() if box.isChecked())
        try:
            options = rerun_options(record, refine=refine,
                                    space_group=self.rerun_space_group.text().strip() or None,
                                    cell=parse_cell_text(self.rerun_cell.text()),
                                    output_dir=stage_dir(self._session(), STAGE_INTEGRATE))
        except ValueError as ex:
            QMessageBox.warning(self, "Re-integrate", str(ex))
            return
        self._start(f"Re-integrating from {record.directory.name} (refining {', '.join(refine) or 'nothing'})…",
                    lambda emit: run_processinglab(options, on_line=emit), self._integrate_done, charts=True)

    def _on_pointless(self) -> None:
        record = self._s.integrated
        if record is None or record.mtz is None:
            return
        out = stage_dir(self._session(), STAGE_CCP4)
        mtz = record.mtz

        def task(emit: Callable[[str], None]) -> PointlessRunResult:
            emit(f"Pointless on {mtz} → {out}")
            return run_pointless(mtz, out / "pointless.mtz", work_dir=out, choose_space_group=True)

        self._start("Running Pointless…", task, self._pointless_done)

    def _on_use_pointless_space_group(self) -> None:
        p = self._s.pointless
        if p is None or not p.space_group:
            return
        self.rerun_space_group.setText(p.space_group)
        self.tabs.setCurrentIndex(3)
        self.status_label.setText(f"Space group {p.space_group} set for re-integration: choose what to refine, "
                                  "then Re-integrate.")

    def _on_aimless(self) -> None:
        p = self._s.pointless
        if p is None:
            return
        hklin, work = p.hklout, p.hklout.parent
        dmin = self.aimless_dmin.value() or None
        anomalous = self.aimless_anomalous.isChecked()

        def task(emit: Callable[[str], None]) -> AimlessRunResult:
            emit(f"Aimless on {hklin}")
            return run_aimless(hklin, work / "aimless.mtz", work_dir=work, resolution_high=dmin,
                               anomalous=anomalous)

        self._start("Running Aimless…", task, self._aimless_done)

    def _on_run_all(self) -> None:
        """Every step with the current settings, each solution / group as suggested."""
        sweep, session = self._sweep, self._session()
        frames, wedge = self._integrate_frames(), self._spot_frames()
        threshold, max_spots = self.spot_threshold.value(), self.spot_max.value()
        ix_threshold, dmax = self.index_threshold.value(), self.index_dmax.value() or None
        dmin, anomalous = self.aimless_dmin.value() or None, self.aimless_anomalous.isChecked()

        def task(emit: Callable[[str], None]) -> _Session:
            s = _Session(folder=session)
            emit("=== Find spots")
            s.spots = find_spots(sweep, session, wedge, threshold=threshold, max_spots=max_spots,
                                 on_line=emit)
            emit("=== Index")
            s.index = autoindex(sweep, session, s.spots.outputs["spot_list"], threshold=ix_threshold, dmax=dmax,
                                on_line=emit)
            chosen = s.index.selected
            space_group = chosen.space_groups[0] if chosen and chosen.space_groups else "1"
            emit(f"=== Integrate in {space_group}")
            s.integrated = integrate(sweep, session, s.index, space_group, frames=frames, on_line=emit)
            if s.integrated.mtz is not None and find_pointless() is not None:
                out = stage_dir(session, STAGE_CCP4)
                emit("=== Pointless")
                s.pointless = run_pointless(s.integrated.mtz, out / "pointless.mtz", work_dir=out)
                if find_aimless() is not None:
                    emit("=== Aimless")
                    s.aimless = run_aimless(s.pointless.hklout, out / "aimless.mtz", work_dir=out,
                                            resolution_high=dmin, anomalous=anomalous)
            return s

        self._start("Running all steps…", task, self._all_done, charts=True)

    # ========================================================= results

    def _spots_done(self, record: RunRecord) -> None:
        self._s.spots, self._s.index, self._s.integrated = record, None, None
        self._s.pointless = self._s.aimless = None
        self._show_all()
        self.tabs.setCurrentIndex(2)
        self.status_label.setText("Spots found: index next.")

    def _index_done(self, result: IndexResult) -> None:
        self._s.index, self._s.integrated, self._s.pointless, self._s.aimless = result, None, None, None
        self._show_all()
        self.status_label.setText("Indexed: check the solution, choose an initial space group, integrate.")

    def _integrate_done(self, record: RunRecord) -> None:
        self.charts.load_log(record.directory / "process.log")  # the final batch's charts
        self._s.integrated, self._s.pointless, self._s.aimless = record, None, None
        self._show_all()
        self.tabs.setCurrentIndex(3)
        self.status_label.setText(f"Integrated: {record.directory}. Run Pointless next.")

    def _pointless_done(self, result: PointlessRunResult) -> None:
        self._s.pointless, self._s.aimless = result, None
        self._show_all()
        self.tabs.setCurrentIndex(4)
        self.status_label.setText("Pointless finished: check the space group and twinning, then scale.")

    def _aimless_done(self, result: AimlessRunResult) -> None:
        self._s.aimless = result
        self._show_all()
        self.tabs.setCurrentIndex(5)
        self.status_label.setText(f"Merged: {result.hklout}")

    def _all_done(self, session: _Session) -> None:
        if session.integrated is not None:
            self.charts.load_log(session.integrated.directory / "process.log")
        self._s = session
        self._show_all()
        self.tabs.setCurrentIndex(5 if session.aimless else 3)
        self.status_label.setText("All steps finished." if session.aimless else
                                  "Integrated (CCP4 Pointless / Aimless not found).")

    def _show_all(self) -> None:
        """Refresh every tab from the session state."""
        s = self._s
        self.session_label.setText(str(s.folder) if s.folder else "(none yet: created by the first step)")
        n_spots = None
        if s.spots is not None:
            spt = s.spots.outputs.get("spot_list")
            try:
                n_spots = len(read_spt(spt)[1]) if spt else None
            except (OSError, ValueError):
                n_spots = None
            self.spots_label.setText(f"{n_spots if n_spots is not None else '?'} spots → {spt}")
        else:
            self.spots_label.setText("")
        self._show_solutions()
        if s.integrated is not None:
            self.integrate_label.setText(format_record(s.integrated))
        else:
            self.integrate_label.setText("Not integrated yet.")
        self.pointless_label.setText(format_pointless(s.pointless) if s.pointless else "")
        self.aimless_label.setText(format_aimless(s.aimless.summary) if s.aimless else "")

        marks = {
            "spots": (s.spots, f"{n_spots} spots" if n_spots is not None else ""),
            "index": (s.index, f"{s.index.selected.lattice} (character {s.index.selected.character})"
                      if s.index and s.index.selected else ""),
            "integrate": (s.integrated, f"{s.integrated.integrated_reflections} reflections, space group "
                          f"{s.integrated.space_group_number}" if s.integrated else ""),
            "symmetry": (s.pointless, f"{s.pointless.solution_kind} {s.pointless.space_group}"
                         if s.pointless else ""),
            "scale": (s.aimless, f"{s.aimless.summary.high_resolution} Å, CC½ {s.aimless.summary.cc_half}"
                      if s.aimless else ""),
        }
        for key, (done, detail) in marks.items():
            label, text = self.pipeline_labels[key]
            label.setText(f"{'●' if done else '○'} {text}" + (f" — {detail}" if done and detail else ""))
        self._update_spot_overlay()
        self._update_enabled()

    def _show_solutions(self) -> None:
        index = self._s.index
        table = self.solutions_table
        table.setRowCount(0)
        self.space_group_combo.clear()
        if index is None:
            self.index_label.setText("")
            return
        bold = QFont()
        bold.setBold(True)
        for row, sol in enumerate(index.solutions):
            table.insertRow(row)
            values = [str(sol.rank) + (" ★" if sol.rank == index.lastgood_rank else ""), sol.lattice,
                      f"{sol.penalty:.0f}", *(f"{v:.2f}" for v in sol.cell[:3]),
                      *(f"{v:.1f}" for v in sol.cell[3:]),
                      "—" if sol.metric_deviation_pct is None else f"{sol.metric_deviation_pct:.3f}",
                      ", ".join(sol.space_groups)]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, sol.character)
                if sol.selected:
                    item.setFont(bold)
                table.setItem(row, col, item)
            if sol.selected:
                table.selectRow(row)
        chosen = index.selected
        if chosen is not None:
            self.space_group_combo.addItems(list(chosen.space_groups))
        frac = "—" if index.indexed_fraction is None else f"{100 * index.indexed_fraction:.0f}%"
        rmsd = "—" if index.rmsd_hkl is None else f"{index.rmsd_hkl:.3f}"
        self.index_label.setText(
            f"Selected (bold): <b>{chosen.lattice if chosen else '—'}</b>, {frac} of spots indexed, "
            f"RMSD {rmsd} (hkl). ★ marks the last solution before the largest penalty jump. "
            "Highlight another row and re-index to choose a different lattice.")

    def _highlighted_solution(self) -> LatticeSolution | None:
        index = self._s.index
        rows = self.solutions_table.selectionModel().selectedRows() if index else []
        if not rows:
            return None
        character = self.solutions_table.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)
        return next((s for s in index.solutions if s.character == character), None)

    def _update_spot_overlay(self, *_args: Any) -> None:
        app = self.parentWidget()
        if app is None or not hasattr(app, "set_indexing_spot_overlay"):
            return
        spots = self._s.spots
        if not self.show_spots_check.isChecked() or spots is None or "spot_list" not in spots.outputs:
            if self._overlay_shown:  # clear only what this window drew
                app.set_indexing_spot_overlay(None)
                self._overlay_shown = False
            return
        scan = scan_of(spots)
        try:
            pixel_mm, spt = read_spt(spots.outputs["spot_list"])
        except (OSError, ValueError):
            return
        if scan is None:
            return
        first_frame, phi0, osc = scan
        frame = self._displayed_frame(first_frame)
        app.set_indexing_spot_overlay(spot_overlay(spt, pixel_mm, frame=frame, first_frame=first_frame,
                                                   phi_start_deg=phi0, osc_deg=osc))
        self._overlay_shown = True

    def _displayed_frame(self, first_frame: int) -> int:
        """The sweep frame the main window shows.

        HDF5: the slider counts within the data file shown, so add the frames
        of the data files before it. CBF/SMV: the frame number of the file.
        """
        app = self.parentWidget()
        slider = getattr(app, "frame_slider", None)
        if self._sweep is not None and self._sweep.is_hdf5 and slider is not None:
            combo = getattr(app, "dataset_combo", None)
            return self._sweep.sweep_frame(int(slider.value()), combo.currentText() if combo is not None else None)
        digits = "".join(ch if ch.isdigit() else " " for ch in Path(getattr(app, "current_file", "") or "").stem)
        numbers = digits.split()
        return int(numbers[-1]) if numbers else first_frame

    # ========================================================= plumbing

    def _start(self, message: str, fn: Callable[[Callable[[str], None]], Any],
               on_finished: Callable[[Any], None], *, charts: bool = False) -> None:
        """Run *fn* in a worker thread; its output lines go to the log (and,
        with *charts*, to the live integration charts, which redraw after each
        batch of frames)."""
        if self._busy():
            QMessageBox.information(self, "ProcessingLab", "A step is already running.")
            return
        if self._sweep is None:
            QMessageBox.warning(self, "ProcessingLab", "Open an image first.")
            return
        self.status_label.setText(message)
        self.log_view.appendPlainText(f"\n=== {datetime.now():%H:%M:%S} {message}")
        self._thread = QThread(self)
        self._worker = CallWorker(fn)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.line.connect(self.log_view.appendPlainText)
        if charts:
            self.charts.reset()
            self._worker.line.connect(self.charts.feed)
            self.tabs.setCurrentIndex(3)
        self._worker.finished.connect(on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._update_enabled)
        self._thread.start()
        self._update_enabled()

    def _on_failed(self, message: str) -> None:
        self.status_label.setText(f"Failed: {message}")
        self.log_view.appendPlainText(f"FAILED: {message}")
        QMessageBox.warning(self, "ProcessingLab", message)

    def _busy(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def _update_tools(self) -> None:
        for name, path in (("ProcessingLab", find_integrationlab()), ("Pointless", find_pointless()),
                           ("Aimless", find_aimless())):
            self.tool_labels[name].setText(str(path) if path else "<span style='color:#b00'>not found</span>")

    def _update_enabled(self) -> None:
        busy, s = self._busy(), self._s
        have_pl = find_integrationlab() is not None
        have_image = self._sweep is not None
        have_pointless, have_aimless = find_pointless() is not None, find_aimless() is not None
        self.find_spots_btn.setEnabled(not busy and have_pl and have_image)
        self.run_all_btn.setEnabled(not busy and have_pl and have_image)
        self.index_btn.setEnabled(not busy and have_pl and s.spots is not None)
        highlighted = self._highlighted_solution()
        self.reindex_btn.setEnabled(not busy and s.spots is not None and highlighted is not None
                                    and not highlighted.selected)
        self.integrate_btn.setEnabled(not busy and have_pl and s.index is not None)
        self.rerun_btn.setEnabled(not busy and have_pl and s.integrated is not None and s.integrated.can_rerun)
        self.report_btn.setEnabled(s.integrated is not None)
        self.mtz_view_btn.setEnabled(s.integrated is not None and s.integrated.mtz is not None)
        self.pointless_btn.setEnabled(not busy and have_pointless and s.integrated is not None
                                      and s.integrated.mtz is not None)
        p = s.pointless
        self.use_sg_btn.setEnabled(not busy and p is not None and p.solution_kind != "point group"
                                   and bool(p.space_group) and is_identity_reindex(p.reindex_operator))
        self.pointless_log_btn.setEnabled(p is not None)
        self.aimless_btn.setEnabled(not busy and have_aimless and p is not None)
        self.merged_view_btn.setEnabled(s.aimless is not None)
        self.aimless_log_btn.setEnabled(s.aimless is not None)

    def _view_mtz(self, mtz: Path | None) -> None:
        if mtz is None:
            return
        try:
            launch_reflection_viewer(mtz)
        except Exception as ex:  # noqa: BLE001 - surfaced to the user
            QMessageBox.warning(self, "Reflection viewer", f"Could not open {mtz}:\n{ex}")

    def showEvent(self, event) -> None:  # noqa: N802
        """Pick up a newly opened image when the window is shown again."""
        super().showEvent(event)
        self.refresh_dataset()


def _row(*widgets: QWidget, stretch_first: bool = False) -> QHBoxLayout:
    row = QHBoxLayout()
    for i, w in enumerate(widgets):
        row.addWidget(w, stretch=1 if stretch_first and i == 0 else 0)
    if not stretch_first:
        row.addStretch(1)
    return row


def _open(path: Path | None) -> None:
    if path is not None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
