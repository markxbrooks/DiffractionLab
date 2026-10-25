"""ProcessingLab Processing window: spots → index → integrate → symmetry → scale.

A processing window built only on ProcessingLab and CCP4, beside the
DIALS / MOSFLM Processing window (:mod:`diffraction_lab.processing_wizard`):

1. **Setup**: the sweep, the session folder, the tools found, pipeline status.
2. **Find spots**: ProcessingLab FINDSPOTS over a frame wedge; spots can be
   overlaid on the image.
3. **Index**: ProcessingLab DPS AUTOINDEX; every lattice solution with its
   penalty, cell and possible space groups. Another solution can be chosen.
4. **Strategy** (``RunStep.STRATEGY``): the smallest rotation, and so the
   number of images, that records a target completeness (MOSFLM's method),
   within Setup's frames to process (then integrate just those) or on a full turn.
5. **Integrate**: ProcessingLab integration in an initial space group chosen
   from the solution; re-integration from the refined cell and detector
   parameters (``--from``). The average spot profile and the resolution
   bins are charted live, after every batch of frames
   (:mod:`diffraction_lab.processinglab_charts`). Untrusted regions and
   resolution ranges for DEFPIX can be added, and the pixels it ignored
   (``defpix.cbf``) shown on the image.
6. **Symmetry**: CCP4 Pointless (Laue / space group, reindexing, L-test for
   twinning).
7. **Scale**: CCP4 Aimless (scale and merge).
8. **Dimple**: CCP4 Dimple refines a known model against the merged data
   and finds unmodelled blobs (:mod:`diffraction_lab.dimple_tab`).

Each stage is a ProcessingLab run in the session folder
(:mod:`diffraction_lab.processinglab_stages`); Pointless and Aimless write
into its ``04-ccp4`` folder, Dimple into ``05-dimple``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
    QMessageBox,
    QPlainTextEdit,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from diffraction_lab.aimless_runner import AimlessRunResult, AimlessSummary, find_aimless, run_aimless
from diffraction_lab.dimple_tab import DimpleTab
from diffraction_lab.pointless_runner import (
    PointlessRunResult,
    find_pointless,
    is_identity_reindex,
    run_pointless,
)
from diffraction_lab.jobs.jobs_tab import JobsTab
from diffraction_lab.jobs.service import SCHEDARRAY_DB_NAME, JobService, use_schedarray
from diffraction_lab.processing_sweep import Frames, ProcessingSweep
from diffraction_lab.processinglab_charts import IntegrationChartsWidget, StrategyCanvas
from diffraction_lab.processinglab_specs import (
    ProcessingLabSpecs,
    create_button,
    create_checkbox,
    create_editable_combo,
    create_optional_double_spinbox,
    create_text_field,
    processinglab_specs,
)
from diffraction_lab.processinglab_client import find_integrationlab
from diffraction_lab.processing_window_base import ProcessingEngine, ProcessingWindowBase
from diffraction_lab.processinglab_format import (
    default_work_parent,
    format_record,
    parse_cell_text,
)
from diffraction_lab.processinglab_run import (
    JobQueue,
    PixelExclusions,
    RunRecord,
    rerun_options,
    run_processinglab,
)
from diffraction_lab.processinglab_stages import (
    STAGE_CCP4,
    STAGE_DIMPLE,
    STAGE_INTEGRATE,
    IndexResult,
    LatticeSolution,
    StrategyPlan,
    autoindex,
    find_spots,
    integrate,
    new_session_dir,
    read_spt,
    scan_of,
    spot_overlay,
    stage_dir,
    strategy,
)
from diffraction_lab.reflection_viewer_launch import launch_reflection_viewer
from diffraction_lab.terminal_log_viewer import TerminalLogViewer
from picoui.helpers.groups import build_group
from picoui.helpers.layout import create_layout_with_items, create_splitter_with_items, create_widget_with_layout
from picoui.widget.helper import create_double_spinbox_from_spec, create_spinbox_from_spec

logger = logging.getLogger(__name__)

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
    strategy: StrategyPlan | None = None
    integrated: RunRecord | None = None
    pointless: PointlessRunResult | None = None
    aimless: AimlessRunResult | None = None


class ProcessingLabWindow(ProcessingWindowBase):
    """Processing with ProcessingLab, Pointless and Aimless only
    (:attr:`ProcessingEngine.PROCESSINGLAB`)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, ProcessingEngine.PROCESSINGLAB, minimum_size=(860, 640))
        self._sweep: ProcessingSweep | None = None
        self._s = _Session()
        self._overlay_shown = False  # whether this window drew the spot overlay
        self._defpix_shown: Path | None = None  # defpix.cbf this window shows
        self._jobs_service: JobService | None = None
        self._terminal: TerminalLogViewer | None = None
        self._slider_connected = False
        #: Every control's range, default, text and slot (one group per tab).
        self.specs: ProcessingLabSpecs = processinglab_specs(self)
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
        self._build_strategy_tab()
        self._build_integrate_tab()
        self._build_symmetry_tab()
        self._build_scale_tab()
        self._build_dimple_tab()
        self._build_jobs_tab()

    def _build_setup_tab(self) -> None:
        spec = self.specs.setup
        self.image_label = QLabel("(no image loaded)")
        self.image_label.setWordWrap(True)
        self.sweep_first = create_spinbox_from_spec(spec.frames)
        self.sweep_last = create_spinbox_from_spec(spec.frames)
        self.sweep_label = QLabel("")
        self.work_edit = create_text_field(spec.work_folder)
        browse = create_button(spec.browse)
        self.session_label = QLabel("(none yet: created by the first step)")
        self.session_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        new_btn = create_button(spec.new_session)
        reveal_btn = create_button(spec.reveal)
        self.tool_labels = {name: QLabel() for name in ("ProcessingLab", "Pointless", "Aimless")}
        self.pipeline_labels = {}
        for key, text in (("spots", "Find spots (ProcessingLab)"), ("index", "Index (ProcessingLab)"),
                          ("strategy", "Strategy (ProcessingLab, optional)"),
                          ("integrate", "Integrate (ProcessingLab)"), ("symmetry", "Symmetry (Pointless)"),
                          ("scale", "Scale (Aimless)")):
            label = QLabel(f"○ {text}")
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.pipeline_labels[key] = (label, text)
        self.run_all_btn = create_button(spec.run_all)

        box = build_group("Sweep and session", _form(
            ("Image", self.image_label),
            ("Frames to process", _row(self.sweep_first, QLabel("to"), self.sweep_last)),
            ("Sweep on disk", self.sweep_label),
            ("Work folder", _row(self.work_edit, browse, stretch_first=True)),
            ("Session", _row(self.session_label, new_btn, reveal_btn, stretch_first=True)),
        ))
        tools = build_group("Programs", _form(*self.tool_labels.items()))
        status = build_group("Pipeline", _column(*(label for label, _text in self.pipeline_labels.values()),
                                                 self.run_all_btn))
        self.tabs.addTab(create_widget_with_layout(_column(box, tools, status, stretch=True)), "1. Setup")

    def _build_spots_tab(self) -> None:
        spec = self.specs.spots
        # Widgets
        self.spot_first = create_spinbox_from_spec(spec.frames)
        self.spot_last = create_spinbox_from_spec(spec.frames)
        self.spot_threshold = create_double_spinbox_from_spec(spec.threshold)
        self.spot_max = create_spinbox_from_spec(spec.max_spots)
        self.find_spots_btn = create_button(spec.find)
        self.show_spots_check = create_checkbox(spec.show_spots)
        self.spots_label = QLabel("")
        self.spots_label.setWordWrap(True)

        # Layout: the settings form in a group, the result below it.
        box = build_group("Spot finding (ProcessingLab FINDSPOTS)", _form(
            ("Frames", _row(self.spot_first, QLabel("to"), self.spot_last)),
            ("Threshold (σ)", self.spot_threshold),
            ("Maximum spots", self.spot_max),
            _row(self.find_spots_btn, self.show_spots_check),
        ))
        self.tabs.addTab(create_widget_with_layout(_column(box, self.spots_label, stretch=True)), "2. Find spots")

    def _build_index_tab(self) -> None:
        spec = self.specs.index
        self.index_threshold = create_double_spinbox_from_spec(spec.threshold)
        self.index_dmax = create_optional_double_spinbox(spec.max_cell_edge)
        self.index_btn = create_button(spec.index)
        self.solutions_table = QTableWidget(0, len(_SOLUTION_COLUMNS))
        self.solutions_table.setHorizontalHeaderLabels(_SOLUTION_COLUMNS)
        self.solutions_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.solutions_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.solutions_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.solutions_table.horizontalHeader().setSectionResizeMode(
            len(_SOLUTION_COLUMNS) - 1, QHeaderView.ResizeMode.Stretch)
        self.solutions_table.itemSelectionChanged.connect(self._update_enabled)
        self.index_label = QLabel("")
        self.index_label.setWordWrap(True)
        self.reindex_btn = create_button(spec.reindex)

        box = build_group("Autoindexing (ProcessingLab DPS)", _form(
            ("Spot I/σ threshold", self.index_threshold),
            ("Maximum cell edge", self.index_dmax),
            self.index_btn,
        ))
        v = _column(box, self.solutions_table, self.index_label, self.reindex_btn)
        v.setStretchFactor(self.solutions_table, 1)  # the table takes the spare height
        self.tabs.addTab(create_widget_with_layout(v), "3. Index")

    def _build_strategy_tab(self) -> None:
        spec = self.specs.strategy
        note = QLabel("From the indexing solution, the smallest rotation that records the completeness "
                      "wanted, as MOSFLM's STRATEGY does. Within Setup's frames to process it says which of "
                      "them to integrate; on a full turn it plans a collection.")
        note.setWordWrap(True)
        self.strategy_sg_combo = create_editable_combo(spec.space_group)
        self.strategy_target = create_double_spinbox_from_spec(spec.target)
        self.strategy_anomalous = create_checkbox(spec.anomalous)
        self.strategy_within = create_checkbox(spec.within)
        self.strategy_dmin = create_optional_double_spinbox(spec.resolution)
        self.strategy_btn = create_button(spec.calculate)
        self.strategy_label = QLabel("No strategy yet.")
        self.strategy_label.setWordWrap(True)
        self.strategy_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.strategy_shells = QTableWidget(0, 6)
        self.strategy_shells.setHorizontalHeaderLabels(
            ["d (Å)", "Possible", "Observed", "Complete", "Multiplicity", "Anomalous"])
        self.strategy_shells.verticalHeader().setVisible(False)
        self.strategy_shells.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.strategy_shells.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.strategy_shells.setMaximumHeight(220)
        self.strategy_use_btn = create_button(spec.use_frames)
        self.strategy_chart = StrategyCanvas()

        box = build_group("Strategy (RunStep.STRATEGY): rotation for a target completeness", _form(
            note,
            ("Space group", self.strategy_sg_combo),
            ("Completeness wanted", self.strategy_target),
            self.strategy_anomalous,
            self.strategy_within,
            ("High-resolution limit", self.strategy_dmin),
            self.strategy_btn,
        ))
        result = build_group("Result", _column(self.strategy_label, self.strategy_shells,
                                               _row(self.strategy_use_btn)))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(create_widget_with_layout(_column(box, result, stretch=True)))
        page = create_splitter_with_items([scroll, self.strategy_chart], [420, 280])
        self.tabs.addTab(page, "4. Strategy")

    def _build_integrate_tab(self) -> None:
        spec = self.specs.integrate
        self.space_group_combo = create_editable_combo(spec.space_group)
        self.integrate_btn = create_button(spec.integrate)
        note = QLabel("Shaded (beamstop), defective and hot pixels are found automatically. "
                      "Add regions and resolution ranges to ignore as well:")
        note.setWordWrap(True)
        self.untrusted_edit = create_text_field(spec.untrusted)
        self.exclude_res_edit = create_text_field(spec.exclude_resolution)
        self.exclude_ice_check = create_checkbox(spec.exclude_ice)
        self.integrate_label = QLabel("Not integrated yet.")
        self.integrate_label.setWordWrap(True)
        self.integrate_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.report_btn = create_button(spec.report)
        self.mtz_view_btn = create_button(spec.view_mtz)
        self.show_defpix_check = create_checkbox(spec.show_defpix)
        self.refine_checks = {name: create_checkbox(check) for name, check in spec.refine.items()}
        self.rerun_space_group = create_text_field(spec.rerun_space_group)
        self.rerun_cell = create_text_field(spec.rerun_cell)
        self.rerun_btn = create_button(spec.rerun)
        self.charts = IntegrationChartsWidget()

        box = build_group("Integration (ProcessingLab)", _form(
            ("Initial space group", self.space_group_combo),
            self.integrate_btn,
        ))
        defpix = build_group("Ignored pixels (DEFPIX)", _form(
            note,
            ("Untrusted regions", self.untrusted_edit),
            ("Exclude resolution", self.exclude_res_edit),
            self.exclude_ice_check,
        ))
        result = build_group("Result", _column(self.integrate_label, _row(self.report_btn, self.mtz_view_btn),
                                               self.show_defpix_check))
        rerun = build_group("Re-integrate with the refined unit cell and detector parameters", _column(
            QLabel("Refine again (unticked parameters stay at their refined values):"),
            _row(*self.refine_checks.values()),
            _form(("Space group", self.rerun_space_group), ("Unit cell", self.rerun_cell)),
            self.rerun_btn,
        ))
        # Controls above, live charts (profile, resolution bins) below.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(create_widget_with_layout(_column(box, defpix, result, rerun, stretch=True)))
        page = create_splitter_with_items([scroll, self.charts], [300, 380])
        self.tabs.addTab(page, "5. Integrate")

    def _build_symmetry_tab(self) -> None:
        spec = self.specs.symmetry
        about = QLabel("Tests the Laue group from symmetry-related intensities, screw axes from "
                       "systematic absences, and twinning (L-test), on the integrated reflections.")
        self.pointless_btn = create_button(spec.run)
        self.pointless_label = QLabel("")
        self.pointless_label.setWordWrap(True)
        self.pointless_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.use_sg_btn = create_button(spec.use_space_group)
        self.pointless_log_btn = create_button(spec.log)

        box = build_group("Space group and twinning (CCP4 Pointless)", _column(
            about, self.pointless_btn, self.pointless_label, _row(self.use_sg_btn, self.pointless_log_btn)))
        self.tabs.addTab(create_widget_with_layout(_column(box, stretch=True)), "6. Symmetry")

    def _build_scale_tab(self) -> None:
        spec = self.specs.scale
        self.aimless_dmin = create_optional_double_spinbox(spec.resolution)
        self.aimless_anomalous = create_checkbox(spec.anomalous)
        self.aimless_btn = create_button(spec.run)
        self.aimless_label = QLabel("")
        self.aimless_label.setWordWrap(True)
        self.aimless_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.merged_view_btn = create_button(spec.view_merged)
        self.aimless_log_btn = create_button(spec.log)

        box = build_group("Scaling and merging (CCP4 Aimless)", _form(
            ("High-resolution cutoff", self.aimless_dmin),
            self.aimless_anomalous,
            self.aimless_btn,
        ))
        self.tabs.addTab(create_widget_with_layout(
            _column(box, self.aimless_label, _row(self.merged_view_btn, self.aimless_log_btn), stretch=True)),
            "7. Scale")

    def _build_dimple_tab(self) -> None:
        self.dimple_tab = DimpleTab(
            self,
            mtz_provider=lambda: self._s.aimless.hklout if self._s.aimless else None,
            folder_provider=lambda: self._s.folder / STAGE_DIMPLE if self._s.folder else None,
            job_service_provider=self._job_service,
            session_id_provider=lambda: (queue.session_id if (queue := self._job_queue()) else None),
            view_mtz=self._view_mtz,
            show_log=False,  # Dimple's output goes to the window's log
        )
        self.tabs.addTab(self.dimple_tab, "8. Dimple")

    def _build_jobs_tab(self) -> None:
        """SchedArray history of this dataset's ProcessingLab, Pointless and Aimless jobs."""
        self.jobs_tab = JobsTab(
            self,
            service_provider=self._job_service,
            work_dir_provider=lambda: self._s.folder,
            tool_filters=(("All", None), ("ProcessingLab", "processinglab"), ("Pointless", "pointless"),
                          ("Aimless", "aimless"), ("Dimple", "dimple")),
        )
        self.jobs_tab.work_dir_only.setText("This session only")
        self.jobs_tab.open_log_requested.connect(self._show_job_log)
        self.tabs.addTab(self.jobs_tab, "Jobs")

    # ========================================================= dataset

    def refresh_dataset(self) -> None:
        """Read the open image and frame count from the main window."""
        app = self.viewer() or self.parentWidget()
        self.set_dataset(self.current_image_path())
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
        threshold, max_spots, jobs = self.spot_threshold.value(), self.spot_max.value(), self._job_queue()
        self._start("Finding spots…", lambda emit: find_spots(sweep, session, wedge, threshold=threshold,
                                                              max_spots=max_spots, on_line=emit, jobs=jobs),
                    self._spots_done)

    def _on_index(self, character: int | None) -> None:
        spots = self._s.spots
        if spots is None or "spot_list" not in spots.outputs:
            return
        sweep, session = self._sweep, self._session()
        threshold = self.index_threshold.value()
        dmax = self.index_dmax.value() or None
        spt, jobs = spots.outputs["spot_list"], self._job_queue()
        what = "Indexing…" if character is None else f"Re-indexing with lattice character {character}…"
        self._start(what, lambda emit: autoindex(sweep, session, spt, threshold=threshold, dmax=dmax,
                                                 character=character, on_line=emit, jobs=jobs), self._index_done)

    def _on_reindex_selected(self) -> None:
        solution = self._highlighted_solution()
        if solution is not None:
            self._on_index(solution.character)

    def _on_strategy(self) -> None:
        index = self._s.index
        space_group = self.strategy_sg_combo.currentText().strip()
        if index is None or not space_group or self._sweep is None:
            return
        sweep, session = self._sweep, self._session()
        # The frames chosen on Setup ("Frames to process"), not every frame on disk.
        within = self._integrate_frames() if self.strategy_within.isChecked() else None
        target, anomalous = self.strategy_target.value(), self.strategy_anomalous.isChecked()
        dmin = self.strategy_dmin.value() or None
        where = f"frames {within}" if within else "a full turn"
        self._start(f"Strategy for {target:g}% in {space_group} within {where}…",
                    lambda emit, jobs=self._job_queue(): strategy(
                        sweep, session, index, space_group, completeness_pct=target, anomalous=anomalous,
                        within=within, resolution=dmin, on_line=emit, jobs=jobs),
                    self._strategy_done)

    def _strategy_done(self, plan: StrategyPlan) -> None:
        self._s.strategy = plan
        self._show_all()
        self._go_to("Strategy")
        t = plan.target
        self.status_label.setText(
            f"Strategy: {t.images} images" + (f" (frames {t.frames})" if t.frames else "") + " for "
            f"{plan.target_completeness_pct:g}%." if t else
            f"Strategy: {plan.target_completeness_pct:g}% is not reachable (at most "
            f"{plan.max_completeness_pct:g}%).")

    def _on_use_strategy_frames(self) -> None:
        plan = self._s.strategy
        if plan is None or plan.target is None or plan.target.frames is None:
            return
        frames = plan.target.frames
        self.sweep_first.setValue(frames.start)
        self.sweep_last.setValue(frames.end)
        self.space_group_combo.setCurrentText(self.strategy_sg_combo.currentText())
        self._go_to("Integrate")
        self.status_label.setText(f"Frames {frames} set for integration ({plan.target.images} images, "
                                  f"{plan.target.completeness_pct:g}% complete): Integrate next.")

    def _show_strategy(self) -> None:
        plan = self._s.strategy
        self.strategy_shells.setRowCount(0)
        if plan is None:
            self.strategy_label.setText("No strategy yet.")
            self.strategy_chart.clear("Completeness against rotation: calculate a strategy")
            return
        t = plan.target
        lo, hi = plan.resolution
        kind = "anomalous " if plan.anomalous else ""
        where = (f"within frames {plan.frames_collected}" if plan.frames_collected
                 else "on a full turn")
        lines = [f"<b>{plan.target_completeness_pct:g}% {kind}completeness</b> {where}, {lo:.1f}–{hi:.2f} Å "
                 f"({plan.unique_possible:,} unique reflections)."]
        if t is None:
            best = plan.max_anomalous_completeness_pct if plan.anomalous else plan.max_completeness_pct
            lines.append(f"<b style='color:#b00'>Not reachable:</b> at most {best:g}% "
                         "(the blind region, or the collected range, limits it).")
        else:
            frames = f", <b>frames {t.frames}</b>" if t.frames else ""
            lines.append(f"<b>Rotate {t.rotation_deg:g}°</b> from φ {t.start_deg:.1f}° to {t.end_deg:.1f}°: "
                         f"<b>{t.images} images</b> of {plan.oscillation_deg:g}°{frames}.")
            lines.append(f"{t.completeness_pct:g}% complete, {t.anomalous_completeness_pct:g}% of anomalous "
                         f"pairs, multiplicity {t.multiplicity:.2f}.")
            for row, sh in enumerate(t.shells):
                self.strategy_shells.insertRow(row)
                for col, text in enumerate([
                        f"{sh['d_low_angstrom']:.2f}–{sh['d_high_angstrom']:.2f}", str(sh["possible"]),
                        str(sh["observed"]), f"{sh['completeness_pct']:g}%", f"{sh['multiplicity']:.2f}",
                        f"{sh['anomalous_completeness_pct']:g}%"]):
                    self.strategy_shells.setItem(row, col, QTableWidgetItem(text))
        self.strategy_label.setText("<br>".join(lines))
        chosen = None if t is None else (
            t.rotation_deg, t.anomalous_completeness_pct if plan.anomalous else t.completeness_pct)
        self.strategy_chart.show_curve(plan.curve, target_pct=plan.target_completeness_pct,
                                       anomalous=plan.anomalous, chosen=chosen,
                                       title="Best start for each total rotation")

    def _on_integrate(self) -> None:
        index = self._s.index
        space_group = self.space_group_combo.currentText().strip()
        if index is None or not space_group:
            return
        exclusions = self._exclusions()
        if exclusions is None:
            return
        sweep, session, frames = self._sweep, self._session(), self._integrate_frames()
        self._start(f"Integrating frames {frames} in {space_group}…",
                    lambda emit, jobs=self._job_queue(): integrate(sweep, session, index, space_group, frames=frames,
                                                                   on_line=emit, jobs=jobs, exclusions=exclusions),
                    self._integrate_done, charts=True)

    def _on_reintegrate(self) -> None:
        record = self._s.integrated
        if record is None:
            return
        refine = tuple(n for n, box in self.refine_checks.items() if box.isChecked())
        exclusions = self._exclusions()
        if exclusions is None:
            return
        try:
            options = rerun_options(record, refine=refine,
                                    space_group=self.rerun_space_group.text().strip() or None,
                                    cell=parse_cell_text(self.rerun_cell.text()),
                                    output_dir=stage_dir(self._session(), STAGE_INTEGRATE),
                                    exclusions=exclusions)
        except ValueError as ex:
            QMessageBox.warning(self, "Re-integrate", str(ex))
            return
        self._start(f"Re-integrating from {record.directory.name} (refining {', '.join(refine) or 'nothing'})…",
                    lambda emit, jobs=self._job_queue(): run_processinglab(options, on_line=emit, jobs=jobs,
                                                                          step="reintegrate"),
                    self._integrate_done, charts=True)

    def _on_pointless(self) -> None:
        record = self._s.integrated
        if record is None or record.mtz is None:
            return
        out = stage_dir(self._session(), STAGE_CCP4)
        mtz, service = record.mtz, self._job_service()

        def task(emit: Callable[[str], None]) -> PointlessRunResult:
            emit(f"Pointless on {mtz} → {out}")
            return run_pointless(mtz, out / "pointless.mtz", work_dir=out, choose_space_group=True,
                                 job_service=service)

        self._start("Running Pointless…", task, self._pointless_done)

    def _on_use_pointless_space_group(self) -> None:
        p = self._s.pointless
        if p is None or not p.space_group:
            return
        self.rerun_space_group.setText(p.space_group)
        self._go_to("Integrate")
        self.status_label.setText(f"Space group {p.space_group} set for re-integration: choose what to refine, "
                                  "then Re-integrate.")

    def _on_aimless(self) -> None:
        p = self._s.pointless
        if p is None:
            return
        hklin, work = p.hklout, p.hklout.parent
        dmin = self.aimless_dmin.value() or None
        anomalous, service = self.aimless_anomalous.isChecked(), self._job_service()

        def task(emit: Callable[[str], None]) -> AimlessRunResult:
            emit(f"Aimless on {hklin}")
            return run_aimless(hklin, work / "aimless.mtz", work_dir=work, resolution_high=dmin,
                               anomalous=anomalous, job_service=service)

        self._start("Running Aimless…", task, self._aimless_done)

    def _on_run_all(self) -> None:
        """Every step with the current settings, each solution / group as suggested."""
        sweep, session = self._sweep, self._session()
        frames, wedge = self._integrate_frames(), self._spot_frames()
        threshold, max_spots = self.spot_threshold.value(), self.spot_max.value()
        ix_threshold, dmax = self.index_threshold.value(), self.index_dmax.value() or None
        dmin, anomalous = self.aimless_dmin.value() or None, self.aimless_anomalous.isChecked()
        exclusions = self._exclusions()
        if exclusions is None:
            return
        jobs = self._job_queue()
        service = jobs.service if jobs else None

        def task(emit: Callable[[str], None]) -> _Session:
            s = _Session(folder=session)
            emit("=== Find spots")
            s.spots = find_spots(sweep, session, wedge, threshold=threshold, max_spots=max_spots,
                                 on_line=emit, jobs=jobs)
            emit("=== Index")
            s.index = autoindex(sweep, session, s.spots.outputs["spot_list"], threshold=ix_threshold, dmax=dmax,
                                on_line=emit, jobs=jobs)
            chosen = s.index.selected
            space_group = chosen.space_groups[0] if chosen and chosen.space_groups else "1"
            emit(f"=== Integrate in {space_group}")
            s.integrated = integrate(sweep, session, s.index, space_group, frames=frames, on_line=emit,
                                     jobs=jobs, exclusions=exclusions)
            if s.integrated.mtz is not None and find_pointless() is not None:
                out = stage_dir(session, STAGE_CCP4)
                emit("=== Pointless")
                s.pointless = run_pointless(s.integrated.mtz, out / "pointless.mtz", work_dir=out,
                                            job_service=service)
                if find_aimless() is not None:
                    emit("=== Aimless")
                    s.aimless = run_aimless(s.pointless.hklout, out / "aimless.mtz", work_dir=out,
                                            resolution_high=dmin, anomalous=anomalous, job_service=service)
            return s

        self._start("Running all steps…", task, self._all_done, charts=True)

    # ========================================================= results

    def _spots_done(self, record: RunRecord) -> None:
        self._s.spots, self._s.index, self._s.integrated = record, None, None
        self._s.pointless = self._s.aimless = None
        self._show_all()
        self._go_to("Index")
        self.status_label.setText("Spots found: index next.")

    def _index_done(self, result: IndexResult) -> None:
        self._s.index, self._s.integrated, self._s.pointless, self._s.aimless = result, None, None, None
        self._s.strategy = None
        self._show_all()
        self.status_label.setText("Indexed: check the solution, choose an initial space group; work out a "
                                  "strategy (optional), then integrate.")

    def _integrate_done(self, record: RunRecord) -> None:
        self.charts.load_log(record.directory / "process.log")  # the final batch's charts
        self._s.integrated, self._s.pointless, self._s.aimless = record, None, None
        self._show_all()
        self._go_to("Integrate")
        self.status_label.setText(f"Integrated: {record.directory}. Run Pointless next.")

    def _pointless_done(self, result: PointlessRunResult) -> None:
        self._s.pointless, self._s.aimless = result, None
        self._show_all()
        self._go_to("Symmetry")
        self.status_label.setText("Pointless finished: check the space group and twinning, then scale.")

    def _aimless_done(self, result: AimlessRunResult) -> None:
        self._s.aimless = result
        self._show_all()
        self._go_to("Scale")
        self.status_label.setText(f"Merged: {result.hklout}")

    def _all_done(self, session: _Session) -> None:
        if session.integrated is not None:
            self.charts.load_log(session.integrated.directory / "process.log")
        self._s = session
        self._show_all()
        self._go_to("Scale" if session.aimless else "Integrate")
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
        self._show_strategy()
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
            "strategy": (s.strategy, (f"{s.strategy.target.images} images"
                                      + (f", frames {s.strategy.target.frames}" if s.strategy.target.frames else "")
                                      if s.strategy.target else "target not reachable") if s.strategy else ""),
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
        self._update_defpix_overlay()
        self._update_enabled()

    def _go_to(self, name: str) -> None:
        """Show the tab titled ``"<n>. <name>"`` (or *name*)."""
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i).split(". ", 1)[-1] == name:
                self.tabs.setCurrentIndex(i)
                return

    def _exclusions(self) -> PixelExclusions | None:
        """The DEFPIX regions and ranges entered, or ``None`` (after a warning) if they don't parse."""
        try:
            return PixelExclusions.parse(self.untrusted_edit.toPlainText(), self.exclude_res_edit.text(),
                                         ice=self.exclude_ice_check.isChecked())
        except ValueError as ex:
            QMessageBox.warning(self, "Ignored pixels", str(ex))
            self._go_to("Integrate")
            return None

    def _update_defpix_overlay(self, *_args: Any) -> None:
        """Show the latest integration's ``defpix.cbf`` on the main window's image, or clear it."""
        record = self._s.integrated
        wanted = record.defpix_image if record is not None and self.show_defpix_check.isChecked() else None
        if wanted == self._defpix_shown:
            return
        viewer = self.viewer()
        if viewer is None or not hasattr(viewer, "set_defpix_overlay"):
            return
        try:
            n = viewer.set_defpix_overlay(wanted)
        except (OSError, ValueError) as ex:
            self.status_label.setText(f"Cannot show the ignored pixels: {ex}")
            viewer.set_defpix_overlay(None)
            self._defpix_shown = None
            return
        self._defpix_shown = wanted
        if wanted is not None:
            self.status_label.setText(f"{n:,} ignored pixels shown in magenta ({wanted}).")

    def _show_solutions(self) -> None:
        index = self._s.index
        table = self.solutions_table
        table.setRowCount(0)
        self.space_group_combo.clear()
        self.strategy_sg_combo.clear()
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
            # Strategy: the highest symmetry of the lattice by default (the
            # least rotation); Pointless confirms it after integration.
            self.strategy_sg_combo.addItems(list(chosen.space_groups))
            self.strategy_sg_combo.setCurrentIndex(len(chosen.space_groups) - 1)
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
        app = self.viewer()
        if app is None or not hasattr(app, "set_indexing_spot_overlay"):
            return
        spots = self._s.spots
        if not self.show_spots_check.isChecked() or spots is None or "spot_list" not in spots.outputs:
            self.clear_spot_overlay()  # only what this window drew
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
        app = self.viewer()
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
        """Run *fn* as a background job (:meth:`start_job`); its output lines go to
        the log (and, with *charts*, to the live integration charts, which
        redraw after each batch of frames)."""
        if self._sweep is None:
            QMessageBox.warning(self, self.engine.label, "Open an image first.")
            return
        if self.job_running():
            QMessageBox.information(self, self.engine.label, "A step is already running.")
            return  # before resetting the charts of the running job
        self.log_view.appendPlainText(f"\n=== {datetime.now():%H:%M:%S} {message}")
        if charts:
            self.charts.reset()
            self._go_to("Integrate")
        if self.start_job(message, fn, on_finished, on_line=self.charts.feed if charts else None):
            self._update_enabled()

    # ----- ProcessingWindowBase hooks

    def report_status(self, text: str) -> None:
        """Status line under the tabs."""
        self.status_label.setText(text)

    def append_log(self, text: str) -> None:
        """The log pane."""
        self.log_view.appendPlainText(text)

    def on_job_ended(self) -> None:
        """Re-enable the buttons; show the finished job in the Jobs tab."""
        self._update_enabled()
        self.jobs_tab.refresh()

    # ----- SchedArray jobs

    def _job_service(self) -> JobService | None:
        """This window's job service: ``<work folder>/schedarray_scheduler.db``.

        One database per dataset (every session in the work folder), owned by
        this window rather than the shared service the DIALS window rebinds.
        ``None`` when SchedArray is not active: steps then run directly.
        """
        if not use_schedarray():
            return None
        work = self.work_edit.text().strip()
        if not work:
            return self._jobs_service
        db = (Path(work).expanduser() / SCHEDARRAY_DB_NAME).resolve()
        current = self._jobs_service
        if current is not None and (current.db_path == db or self.job_running()):
            return current  # never switch databases under a running job
        try:
            service = JobService(db_path=db)
        except Exception as ex:  # noqa: BLE001 - steps still run without it
            logger.warning("SchedArray job service unavailable: %s", ex)
            return current
        if current is not None:
            current.close()
        self._jobs_service = service
        return service

    def _job_queue(self) -> JobQueue | None:
        """Where steps are submitted (``None``: run directly)."""
        service = self._job_service()
        return None if service is None else JobQueue(service, self._s.folder.name if self._s.folder else None)

    def _show_job_log(self, log_path: object) -> None:
        """Follow a job's log in a terminal window."""
        if self._terminal is None:
            self._terminal = TerminalLogViewer(self, title=f"DiffractionLab Terminal — {self.engine.label}")
        self._terminal.set_log_file(Path(str(log_path)), reload=True)
        self._terminal.show()
        self._terminal.raise_()

    def clear_spot_overlay(self) -> None:
        """Clear the overlay only if this window drew it."""
        if self._overlay_shown:
            super().clear_spot_overlay()
            self._overlay_shown = False

    def closeEvent(self, event) -> None:  # noqa: N802
        """Also clear the ignored-pixel overlay."""
        viewer = self.viewer()
        if self._defpix_shown is not None and viewer is not None:
            viewer.set_defpix_overlay(None)
            self._defpix_shown = None
        super().closeEvent(event)

    def _update_tools(self) -> None:
        for name, path in (("ProcessingLab", find_integrationlab()), ("Pointless", find_pointless()),
                           ("Aimless", find_aimless())):
            self.tool_labels[name].setText(str(path) if path else "<span style='color:#b00'>not found</span>")

    def _update_enabled(self) -> None:
        busy, s = self.job_running(), self._s
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
        self.strategy_btn.setEnabled(not busy and have_pl and s.index is not None)
        self.strategy_use_btn.setEnabled(s.strategy is not None and s.strategy.target is not None
                                         and s.strategy.target.frames is not None)
        self.rerun_btn.setEnabled(not busy and have_pl and s.integrated is not None and s.integrated.can_rerun)
        self.report_btn.setEnabled(s.integrated is not None)
        self.mtz_view_btn.setEnabled(s.integrated is not None and s.integrated.mtz is not None)
        self.show_defpix_check.setEnabled(s.integrated is not None and s.integrated.defpix_image is not None)
        self.pointless_btn.setEnabled(not busy and have_pointless and s.integrated is not None
                                      and s.integrated.mtz is not None)
        p = s.pointless
        self.use_sg_btn.setEnabled(not busy and p is not None and p.solution_kind != "point group"
                                   and bool(p.space_group) and is_identity_reindex(p.reindex_operator))
        self.pointless_log_btn.setEnabled(p is not None)
        self.aimless_btn.setEnabled(not busy and have_aimless and p is not None)
        self.merged_view_btn.setEnabled(s.aimless is not None)
        self.dimple_tab.update_enabled()
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


def _form(*rows: tuple[str, QWidget | QLayout] | QWidget | QLayout) -> QFormLayout:
    """A form layout: each row is ``(label, field)``, or one widget or layout across both columns."""
    form = QFormLayout()
    for row in rows:
        if isinstance(row, tuple):
            form.addRow(*row)
        else:
            form.addRow(row)
    return form


def _column(*items: QWidget | QLayout, stretch: bool = False) -> QVBoxLayout:
    """*items* stacked vertically (PicoUI) with the window's margins and spacing; *stretch*: space below."""
    return create_layout_with_items(items=list(items), vertical=True, start_stretch=False, end_stretch=stretch,
                                    margins=(9, 9, 9, 9), spacing=6)


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
