"""Indexing dialog for DiffractionLab (findspots → DPS → refine)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont, QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from diffraction_lab.dials_engine import (
    DialsEngine,
    DialsImageRange,
    DialsIndexingResult,
    DialsIntegrationResult,
    DialsSettings,
    DialsSpotFindingResult,
    auto_dials_image_range,
    centroids_px_for_image,
    default_dials_work_dir,
    dials_run_root_for_master,
    export_indexed_predictions_npz,
    export_strong_centroids_npz,
    find_dials_prefix,
    latest_dials_run_dir,
)
from diffraction_lab.dials_spec import (
    dials_settings_fields,
    dials_settings_to_values,
    values_to_dials_settings,
)

try:
    from picoui.widget.spec_editor import SpecEditor

    _HAS_SPEC_EDITOR = True
except ImportError:  # pragma: no cover
    SpecEditor = None  # type: ignore[misc, assignment]
    _HAS_SPEC_EDITOR = False
from diffraction_lab.indexing import (
    IndexSolution,
    IndexingEngine,
    IndexingGeometry,
    IndexingResult,
    IndexingSettings,
    IntegrationImageRange,
    IntegrationResult,
    MosaicityEstimation,
    SpotFindingResult,
    heuristic_integration_range,
    match_spots_to_solution,
    normalize_spacegroup,
    preselection_solution,
    spacegroups_for_lattice,
    spots_for_frame,
)
from diffraction_lab.mosflm_client import find_ipmosflm
from diffraction_lab.reflection_viewer_launch import launch_reflection_viewer
from diffraction_lab.summary_vs_resolution import open_summary_vs_resolution_from_work_dir
from diffraction_lab.terminal_log_viewer import TerminalLogViewer

logger = logging.getLogger(__name__)


def _dials_to_index_solution(dials: DialsIndexingResult) -> IndexSolution:
    """Map a DIALS crystal model into the Mosflm-oriented table row type."""
    sol = dials.solution
    return IndexSolution(
        number=1,
        kind="ref" if dials.refined else "reg",
        penalty=0.0,
        lattice=sol.lattice or "aP",
        spacegroup=sol.spacegroup or "P1",
        a=sol.a,
        b=sol.b,
        c=sol.c,
        alpha=sol.alpha,
        beta=sol.beta,
        gamma=sol.gamma,
        matrix=(),  # DIALS A-matrix lives in .expt; Mosflm matrix unused here.
        reflections_used=sol.n_indexed,
    )


class _SpotFindWorker(QObject):
    """Background worker for tutorial §5.1 spot finding."""

    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        frames: list[np.ndarray],
        geometry: IndexingGeometry,
        settings: IndexingSettings,
        phis: list[tuple[float, float]],
        frame_indices: list[int],
        work_dir: Path,
    ) -> None:
        super().__init__()
        self._frames = frames
        self._geometry = geometry
        self._settings = settings
        self._phis = phis
        self._frame_indices = frame_indices
        self._work_dir = work_dir

    def run(self) -> None:
        try:
            self.progress.emit(
                f"Finding spots in {len(self._frames)} frame(s)…"
            )
            engine = IndexingEngine(
                work_dir=self._work_dir,
                settings=self._settings,
            )
            result = engine.find_spots_step(
                self._frames,
                self._geometry,
                phis=self._phis,
                frame_indices=self._frame_indices,
            )
            self.finished.emit(result)
        except Exception as ex:  # noqa: BLE001 - surface to UI
            logger.exception("Spot finding failed")
            self.failed.emit(str(ex))


class _IndexWorker(QObject):
    """Background worker for DPS indexing after spot finding (§5.2)."""

    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        spotfinding: SpotFindingResult,
        settings: IndexingSettings,
        work_dir: Path,
    ) -> None:
        super().__init__()
        self._spotfinding = spotfinding
        self._settings = settings
        self._work_dir = work_dir

    def run(self) -> None:
        try:
            engine = IndexingEngine(
                work_dir=self._work_dir,
                settings=self._settings,
            )
            self.progress.emit("Running Mosflm DPS autoindexing…")
            result = engine.index_spotfile(self._spotfinding, refine=False)
            self.finished.emit(result)
        except Exception as ex:  # noqa: BLE001 - surface to UI
            logger.exception("Indexing failed")
            self.failed.emit(str(ex))


class _DialsSpotFindWorker(QObject):
    """DIALS ``import`` + ``find_spots`` for HDF5 masters."""

    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        master_path: Path,
        image_range: DialsImageRange,
        work_dir: Path,
        settings: DialsSettings | None = None,
        on_chunk=None,
    ) -> None:
        super().__init__()
        self._master_path = master_path
        self._image_range = image_range
        self._work_dir = work_dir
        self._settings = settings or DialsSettings()
        self._on_chunk = on_chunk

    def run(self) -> None:
        try:
            engine = DialsEngine(
                work_dir=self._work_dir,
                settings=self._settings,
                on_chunk=self._on_chunk,
            )
            self.progress.emit("DIALS: importing images…")
            imported = engine.import_images(self._master_path)
            self.progress.emit(
                f"DIALS: finding spots on images "
                f"{self._image_range.first_image}–{self._image_range.last_image}…"
            )
            result = engine.find_spots(imported, self._image_range)
            self.finished.emit(result)
        except Exception as ex:  # noqa: BLE001
            logger.exception("DIALS spot finding failed")
            self.failed.emit(str(ex))


class _DialsIndexWorker(QObject):
    """DIALS ``index`` + ``refine`` after spot finding."""

    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        spots: DialsSpotFindingResult,
        work_dir: Path,
        settings: DialsSettings | None = None,
        on_chunk=None,
    ) -> None:
        super().__init__()
        self._spots = spots
        self._work_dir = work_dir
        self._settings = settings or DialsSettings()
        self._on_chunk = on_chunk

    def run(self) -> None:
        try:
            engine = DialsEngine(
                work_dir=self._work_dir,
                settings=self._settings,
                on_chunk=self._on_chunk,
            )
            self.progress.emit("DIALS: indexing (fft3d)…")
            result = engine.index(
                self._spots.imported_expt,
                self._spots.strong_refl,
                refine=True,
            )
            self.finished.emit(result)
        except Exception as ex:  # noqa: BLE001
            logger.exception("DIALS indexing failed")
            self.failed.emit(str(ex))


class _DialsIntegrateWorker(QObject):
    """DIALS ``integrate`` + ``export`` MTZ."""

    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        indexed: DialsIndexingResult,
        image_range: DialsImageRange,
        work_dir: Path,
        settings: DialsSettings | None = None,
        on_chunk=None,
    ) -> None:
        super().__init__()
        self._indexed = indexed
        self._image_range = image_range
        self._work_dir = work_dir
        self._settings = settings or DialsSettings()
        self._on_chunk = on_chunk

    def run(self) -> None:
        try:
            engine = DialsEngine(
                work_dir=self._work_dir,
                settings=self._settings,
                on_chunk=self._on_chunk,
            )
            self.progress.emit(
                f"DIALS: integrating images "
                f"{self._image_range.first_image}–{self._image_range.last_image}…"
            )
            result = engine.integrate(
                self._indexed.experiments,
                self._indexed.reflections,
                image_range=self._image_range,
            )
            self.finished.emit(result)
        except Exception as ex:  # noqa: BLE001
            logger.exception("DIALS integration failed")
            self.failed.emit(str(ex))


class _DialsFullPipelineWorker(QObject):
    """One-shot ``import → find_spots → index → integrate → export``."""

    finished = Signal(object)  # tuple[spots, indexed, integ]
    failed = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        master_path: Path,
        image_range: DialsImageRange,
        work_dir: Path,
        settings: DialsSettings | None = None,
        on_chunk=None,
    ) -> None:
        super().__init__()
        self._master_path = master_path
        self._image_range = image_range
        self._work_dir = work_dir
        self._settings = settings or DialsSettings()
        self._on_chunk = on_chunk

    def run(self) -> None:
        try:
            engine = DialsEngine(
                work_dir=self._work_dir,
                settings=self._settings,
                on_chunk=self._on_chunk,
            )
            self.progress.emit("DIALS full pipeline: import…")
            imported = engine.import_images(self._master_path)
            self.progress.emit(
                f"DIALS full pipeline: find spots "
                f"{self._image_range.first_image}–{self._image_range.last_image}…"
            )
            spots = engine.find_spots(imported, self._image_range)
            self.progress.emit("DIALS full pipeline: index + refine…")
            indexed = engine.index(imported, spots.strong_refl, refine=True)
            self.progress.emit("DIALS full pipeline: integrate + export MTZ…")
            integ = engine.integrate(
                indexed.experiments,
                indexed.reflections,
                image_range=self._image_range,
            )
            self.finished.emit((spots, indexed, integ))
        except Exception as ex:  # noqa: BLE001
            logger.exception("DIALS full pipeline failed")
            self.failed.emit(str(ex))


class _IntegrationWorker(QObject):
    """Background worker: auto-select images and run Mosflm trial integration."""

    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        geometry: IndexingGeometry,
        solution: IndexSolution,
        spacegroup: str,
        settings: IndexingSettings,
        work_dir: Path,
        *,
        total_images: int,
        mosaicity_deg: float,
        phi_start: float,
        oscillation_deg: float,
        image_range: IntegrationImageRange | None,
    ) -> None:
        super().__init__()
        self._geometry = geometry
        self._solution = solution
        self._spacegroup = spacegroup
        self._settings = settings
        self._work_dir = work_dir
        self._total_images = total_images
        self._mosaicity_deg = mosaicity_deg
        self._phi_start = phi_start
        self._oscillation_deg = oscillation_deg
        self._image_range = image_range

    def run(self) -> None:
        try:
            engine = IndexingEngine(
                work_dir=self._work_dir,
                settings=self._settings,
            )
            image_range = self._image_range
            if image_range is None:
                self.progress.emit("Selecting integration images…")
                image_range = engine.auto_select_integration_range(
                    self._geometry,
                    self._solution,
                    spacegroup=self._spacegroup,
                    mosaicity_deg=self._mosaicity_deg,
                    total_images=self._total_images,
                )
            self.progress.emit(
                f"Integrating Mosflm images "
                f"{image_range.first_image}–{image_range.last_image}…"
            )
            result = engine.integrate_images(
                self._geometry,
                self._solution,
                image_range,
                spacegroup=self._spacegroup,
                mosaicity_deg=self._mosaicity_deg,
                phi_start=self._phi_start,
                oscillation_deg=self._oscillation_deg,
            )
            self.finished.emit(result)
        except Exception as ex:  # noqa: BLE001 - surface to UI
            logger.exception("Integration failed")
            self.failed.emit(str(ex))


class _MosaicityWorker(QObject):
    """Background worker for mosaic spread estimation (§5.6)."""

    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        geometry: IndexingGeometry,
        solution: IndexSolution,
        spacegroup: str,
        settings: IndexingSettings,
        work_dir: Path,
        *,
        phi_start: float,
        phi_end: float,
        phi_scan_end: float,
        image_number: int,
    ) -> None:
        super().__init__()
        self._geometry = geometry
        self._solution = solution
        self._spacegroup = spacegroup
        self._settings = settings
        self._work_dir = work_dir
        self._phi_start = phi_start
        self._phi_end = phi_end
        self._phi_scan_end = phi_scan_end
        self._image_number = image_number

    def run(self) -> None:
        try:
            self.progress.emit("Estimating mosaic spread…")
            engine = IndexingEngine(
                work_dir=self._work_dir,
                settings=self._settings,
            )
            result = engine.estimate_mosaicity(
                self._geometry,
                self._solution,
                spacegroup=self._spacegroup,
                image_number=self._image_number,
                phi_start=self._phi_start,
                phi_end=self._phi_end,
                phi_scan_end=self._phi_scan_end,
            )
            self.finished.emit(result)
        except Exception as ex:  # noqa: BLE001 - surface to UI
            logger.exception("Mosaicity estimation failed")
            self.failed.emit(str(ex))


class _RefineWorker(QObject):
    """Background worker for Mosflm ``index2`` with cell / space group."""

    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        geometry: IndexingGeometry,
        spotfile: Path,
        solution_number: int,
        cell: tuple[float, float, float, float, float, float],
        spacegroup: str,
        settings: IndexingSettings,
        work_dir: Path,
    ) -> None:
        super().__init__()
        self._geometry = geometry
        self._spotfile = spotfile
        self._solution_number = solution_number
        self._cell = cell
        self._spacegroup = spacegroup
        self._settings = settings
        self._work_dir = work_dir

    def run(self) -> None:
        try:
            self.progress.emit(
                f"Refining solution #{self._solution_number} "
                f"with {self._spacegroup}…"
            )
            engine = IndexingEngine(
                work_dir=self._work_dir,
                settings=self._settings,
            )
            refined = engine.refine_index_solution(
                self._geometry,
                self._spotfile,
                self._solution_number,
                self._cell,
                self._spacegroup,
                sigma_cutoff=self._settings.sigma_cutoff,
            )
            self.finished.emit(refined)
        except Exception as ex:  # noqa: BLE001 - surface to UI
            logger.exception("Refinement failed")
            self.failed.emit(str(ex))


class IndexingDialog(QDialog):
    """Processing dialog: find spots → index → integrate → MTZ."""

    solutionSelected = Signal(object)  # IndexSolution | None

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("Processing")
        self.setMinimumSize(780, 560)
        self.setModal(False)
        self._result: IndexingResult | None = None
        self._table_solutions: list[IndexSolution] = []
        self._thread: QThread | None = None
        self._worker: QObject | None = None
        self._refine_thread: QThread | None = None
        self._refine_worker: _RefineWorker | None = None
        self._spot_thread: QThread | None = None
        self._spot_worker: _SpotFindWorker | None = None
        self._mosaic_thread: QThread | None = None
        self._mosaic_worker: _MosaicityWorker | None = None
        self._integrate_thread: QThread | None = None
        self._integrate_worker: _IntegrationWorker | None = None
        self._spotfinding: SpotFindingResult | None = None
        self._dials_spots: DialsSpotFindingResult | None = None
        self._dials_index: DialsIndexingResult | None = None
        self._integration_range: IntegrationImageRange | None = None
        self._integration_result: IntegrationResult | None = None
        self._dials_integration: DialsIntegrationResult | None = None
        self._mtz_path: Path | None = None
        self._use_dials: bool = False
        self._selected_frames: list[int] = []
        self._source_geometry: IndexingGeometry | None = None
        self._dials_work_dir: Path = Path.home() / ".qtmosflm" / "dials"
        self._dials_master_path: Path | None = None
        self._last_dials_log: Path | None = None
        self._pipeline_thread: QThread | None = None
        self._pipeline_worker: _DialsFullPipelineWorker | None = None
        self._busy_cancel_requested = False
        self._terminal_viewer: TerminalLogViewer | None = None
        self._build_ui()
        self._populate_defaults()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.pipeline_tabs = QTabWidget()
        layout.addWidget(self.pipeline_tabs, stretch=1)

        # ----- Tab: Setup -----
        setup_page = QWidget()
        setup_layout = QVBoxLayout(setup_page)

        self.dials_input_box = QGroupBox("HDF5 input & work directory")
        dials_input_form = QFormLayout(self.dials_input_box)
        self.dials_master_label = QLabel("(no master loaded)")
        self.dials_master_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.dials_master_label.setWordWrap(True)
        dials_input_form.addRow("Master .h5", self.dials_master_label)

        work_row = QHBoxLayout()
        self.dials_work_dir_edit = QLineEdit()
        self.dials_work_dir_edit.setPlaceholderText(
            "<master_dir>/<master_stem>/YYYY-MM-DD-HH-MM-SS-dials/"
        )
        work_row.addWidget(self.dials_work_dir_edit, stretch=1)
        self.dials_work_browse_btn = QPushButton("Browse…")
        self.dials_work_browse_btn.clicked.connect(self._on_browse_work_dir)
        work_row.addWidget(self.dials_work_browse_btn)
        self.dials_work_reveal_btn = QPushButton("Reveal")
        self.dials_work_reveal_btn.clicked.connect(self._on_reveal_work_dir)
        work_row.addWidget(self.dials_work_reveal_btn)
        dials_input_form.addRow("Work directory", work_row)
        setup_layout.addWidget(self.dials_input_box)

        self.dials_settings_box = QGroupBox("DIALS settings")
        dials_settings_layout = QVBoxLayout(self.dials_settings_box)
        self._dials_spec_editor: SpecEditor | None = None
        # Fallback widgets when PicoUI SpecEditor is unavailable.
        self.dials_nproc: QSpinBox | None = None
        self.dials_d_min: QDoubleSpinBox | None = None
        self.dials_min_spot_size: QSpinBox | None = None
        self.dials_gain: QDoubleSpinBox | None = None
        self.dials_space_group: QLineEdit | None = None
        self.dials_unit_cell: QLineEdit | None = None
        if _HAS_SPEC_EDITOR and SpecEditor is not None:
            self._dials_spec_editor = SpecEditor(
                fields=dials_settings_fields(),
                show_argv_preview=True,
                program="dials",
            )
            self._dials_spec_editor.set_values(
                dials_settings_to_values(DialsSettings())
            )
            dials_settings_layout.addWidget(self._dials_spec_editor)
        else:
            dials_form = QFormLayout()
            self.dials_nproc = QSpinBox()
            self.dials_nproc.setRange(1, 64)
            self.dials_nproc.setValue(4)
            dials_form.addRow("nproc", self.dials_nproc)

            self.dials_d_min = QDoubleSpinBox()
            self.dials_d_min.setRange(0.0, 50.0)
            self.dials_d_min.setDecimals(2)
            self.dials_d_min.setSpecialValueText("auto")
            self.dials_d_min.setValue(0.0)
            dials_form.addRow("d_min (Å)", self.dials_d_min)

            self.dials_min_spot_size = QSpinBox()
            self.dials_min_spot_size.setRange(1, 50)
            self.dials_min_spot_size.setValue(3)
            dials_form.addRow("min spot size", self.dials_min_spot_size)

            self.dials_gain = QDoubleSpinBox()
            self.dials_gain.setRange(0.0, 100.0)
            self.dials_gain.setDecimals(2)
            self.dials_gain.setSpecialValueText("auto")
            self.dials_gain.setValue(0.0)
            dials_form.addRow("gain", self.dials_gain)

            self.dials_space_group = QLineEdit()
            self.dials_space_group.setPlaceholderText("optional, e.g. P212121")
            dials_form.addRow("Known space group", self.dials_space_group)

            self.dials_unit_cell = QLineEdit()
            self.dials_unit_cell.setPlaceholderText("optional a,b,c,α,β,γ")
            dials_form.addRow("Known unit cell", self.dials_unit_cell)
            dials_settings_layout.addLayout(dials_form)
        setup_layout.addWidget(self.dials_settings_box)

        self.dials_pipeline_box = QGroupBox("Pipeline status")
        pipe_layout = QVBoxLayout(self.dials_pipeline_box)
        self.pipeline_import_label = QLabel("○ Import —")
        self.pipeline_spots_label = QLabel("○ Find spots —")
        self.pipeline_index_label = QLabel("○ Index / refine —")
        self.pipeline_integrate_label = QLabel("○ Integrate —")
        self.pipeline_export_label = QLabel("○ Export MTZ —")
        for lab in (
            self.pipeline_import_label,
            self.pipeline_spots_label,
            self.pipeline_index_label,
            self.pipeline_integrate_label,
            self.pipeline_export_label,
        ):
            lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            pipe_layout.addWidget(lab)
        setup_layout.addWidget(self.dials_pipeline_box)

        form_box = QGroupBox("Frames & settings")
        self.frames_settings_box = form_box
        form = QFormLayout(form_box)

        self.frame_start = QSpinBox()
        self.frame_start.setRange(0, 999999)
        self.frame_end = QSpinBox()
        self.frame_end.setRange(0, 999999)
        self.frame_step = QSpinBox()
        self.frame_step.setRange(1, 9999)
        self.frame_step.setValue(90)
        self.frame_step_label = QLabel("Step")

        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Start"))
        frame_row.addWidget(self.frame_start)
        frame_row.addWidget(QLabel("End"))
        frame_row.addWidget(self.frame_end)
        frame_row.addWidget(self.frame_step_label)
        frame_row.addWidget(self.frame_step)
        form.addRow("Frames (0-based)", frame_row)

        self.dials_range_hint = QLabel("")
        self.dials_range_hint.setWordWrap(True)
        form.addRow(self.dials_range_hint)
        self.frame_start.valueChanged.connect(self._update_dials_range_hint)
        self.frame_end.valueChanged.connect(self._update_dials_range_hint)

        self.spot_sigma = QDoubleSpinBox()
        self.spot_sigma.setRange(1.0, 50.0)
        self.spot_sigma.setDecimals(1)
        self.spot_sigma.setSingleStep(0.5)
        self.spot_sigma.setValue(5.0)
        self.spot_sigma_label = QLabel("Spot σ threshold")
        form.addRow(self.spot_sigma_label, self.spot_sigma)

        self.i_sig_i = QDoubleSpinBox()
        self.i_sig_i.setRange(1.0, 200.0)
        self.i_sig_i.setDecimals(1)
        self.i_sig_i.setValue(5.0)
        self.i_sig_i_label = QLabel("DPS I/σI")
        form.addRow(self.i_sig_i_label, self.i_sig_i)

        self.max_penalty = QDoubleSpinBox()
        self.max_penalty.setRange(0.0, 500.0)
        self.max_penalty.setDecimals(1)
        self.max_penalty.setValue(50.0)
        self.max_penalty_label = QLabel("Max penalty to refine")
        form.addRow(self.max_penalty_label, self.max_penalty)

        setup_layout.addWidget(form_box)

        self.backend_label = QLabel("")
        self.backend_label.setWordWrap(True)
        setup_layout.addWidget(self.backend_label)
        setup_layout.addStretch(1)
        self.pipeline_tabs.addTab(setup_page, "1. Setup")

        # ----- Tab: Find spots -----
        spots_page = QWidget()
        spots_layout = QVBoxLayout(spots_page)
        spot_box = QGroupBox("Spot finding")
        spot_layout = QHBoxLayout(spot_box)
        self.find_spots_button = QPushButton("Find spots")
        self.find_spots_button.clicked.connect(self._on_find_spots)
        spot_layout.addWidget(self.find_spots_button)
        self.spot_summary_label = QLabel(
            "Search selected frames, then check spots on the image before indexing."
        )
        self.spot_summary_label.setWordWrap(True)
        spot_layout.addWidget(self.spot_summary_label, stretch=1)
        spots_layout.addWidget(spot_box)

        self.show_spots_checkbox = QCheckBox("Show indexing spots on image")
        self.show_spots_checkbox.setChecked(True)
        self.show_spots_checkbox.toggled.connect(self._on_show_spots_toggled)
        spots_layout.addWidget(self.show_spots_checkbox)
        spots_layout.addStretch(1)
        self.pipeline_tabs.addTab(spots_page, "2. Find spots")

        # ----- Tab: Index -----
        index_page = QWidget()
        index_layout = QVBoxLayout(index_page)
        self.run_button = QPushButton("Run indexing")
        self.run_button.setEnabled(False)
        self.run_button.setToolTip(
            "Requires spots from Find spots (Mosflm DPS or DIALS)."
        )
        self.run_button.clicked.connect(self._on_run)
        index_layout.addWidget(self.run_button)

        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels(
            [
                "Kind",
                "#",
                "Penalty",
                "Lattice",
                "SG",
                "a",
                "b",
                "c",
                "α",
                "β / γ",
            ]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.currentCellChanged.connect(self._on_current_cell_changed)
        index_layout.addWidget(self.table, stretch=1)

        refine_box = QGroupBox("Refine unit cell")
        refine_layout = QHBoxLayout(refine_box)
        refine_layout.addWidget(QLabel("Space group"))
        self.spacegroup_combo = QComboBox()
        self.spacegroup_combo.setEditable(True)
        self.spacegroup_combo.setMinimumWidth(120)
        refine_layout.addWidget(self.spacegroup_combo)
        refine_layout.addWidget(QLabel("σ cutoff"))
        self.refine_sigma = QDoubleSpinBox()
        self.refine_sigma.setRange(0.5, 20.0)
        self.refine_sigma.setDecimals(2)
        self.refine_sigma.setSingleStep(0.25)
        self.refine_sigma.setValue(2.5)
        refine_layout.addWidget(self.refine_sigma)
        self.refine_button = QPushButton("Refine selected solution")
        self.refine_button.setEnabled(False)
        self.refine_button.clicked.connect(self._on_refine)
        refine_layout.addWidget(self.refine_button)
        refine_layout.addStretch(1)
        index_layout.addWidget(refine_box)

        mosaic_box = QGroupBox("Mosaicity estimation (§5.6)")
        self.mosaic_box = mosaic_box
        mosaic_layout = QHBoxLayout(mosaic_box)
        mosaic_layout.addWidget(QLabel("Mosaic spread (°)"))
        self.mosaicity_value = QDoubleSpinBox()
        self.mosaicity_value.setRange(0.0, 20.0)
        self.mosaicity_value.setDecimals(3)
        self.mosaicity_value.setSingleStep(0.05)
        self.mosaicity_value.setValue(0.05)
        mosaic_layout.addWidget(self.mosaicity_value)
        mosaic_layout.addWidget(QLabel("Block size (µm)"))
        self.mosaic_block = QDoubleSpinBox()
        self.mosaic_block.setRange(0.1, 500.0)
        self.mosaic_block.setDecimals(1)
        self.mosaic_block.setValue(100.0)
        mosaic_layout.addWidget(self.mosaic_block)
        self.mosaicity_estimate_button = QPushButton("Estimate")
        self.mosaicity_estimate_button.setEnabled(False)
        self.mosaicity_estimate_button.clicked.connect(self._on_estimate_mosaicity)
        mosaic_layout.addWidget(self.mosaicity_estimate_button)
        mosaic_layout.addStretch(1)
        index_layout.addWidget(mosaic_box)
        self.pipeline_tabs.addTab(index_page, "3. Index")

        # ----- Tab: Integrate -----
        integrate_page = QWidget()
        integrate_page_layout = QVBoxLayout(integrate_page)
        integrate_box = QGroupBox("Integration")
        self.integrate_box = integrate_box
        integrate_layout = QVBoxLayout(integrate_box)
        range_row = QHBoxLayout()
        self.integrate_range_label = QLabel(
            "After indexing, auto-select a short image wedge for integration."
        )
        self.integrate_range_label.setWordWrap(True)
        range_row.addWidget(self.integrate_range_label, stretch=1)
        self.auto_integrate_select_button = QPushButton("Auto-select images")
        self.auto_integrate_select_button.setEnabled(False)
        self.auto_integrate_select_button.clicked.connect(
            self._on_auto_select_integration
        )
        range_row.addWidget(self.auto_integrate_select_button)
        integrate_layout.addLayout(range_row)

        dials_integ_row = QHBoxLayout()
        dials_integ_row.addWidget(QLabel("Integrate first (1-based)"))
        self.integrate_first = QSpinBox()
        self.integrate_first.setRange(1, 999999)
        self.integrate_first.setValue(1)
        dials_integ_row.addWidget(self.integrate_first)
        dials_integ_row.addWidget(QLabel("Last"))
        self.integrate_last = QSpinBox()
        self.integrate_last.setRange(1, 999999)
        self.integrate_last.setValue(40)
        dials_integ_row.addWidget(self.integrate_last)
        dials_integ_row.addStretch(1)
        self.dials_integrate_range_row = QWidget()
        self.dials_integrate_range_row.setLayout(dials_integ_row)
        integrate_layout.addWidget(self.dials_integrate_range_row)

        self.integrate_button = QPushButton("Integrate selected wedge")
        self.integrate_button.setEnabled(False)
        self.integrate_button.clicked.connect(self._on_integrate)
        integrate_layout.addWidget(self.integrate_button)

        self.full_pipeline_button = QPushButton("Run full DIALS pipeline → MTZ")
        self.full_pipeline_button.setToolTip(
            "import → find_spots → index → refine → integrate → export"
        )
        self.full_pipeline_button.clicked.connect(self._on_run_full_pipeline)
        integrate_layout.addWidget(self.full_pipeline_button)
        integrate_page_layout.addWidget(integrate_box)
        integrate_page_layout.addStretch(1)
        self.pipeline_tabs.addTab(integrate_page, "4. Integrate")

        # ----- Tab: MTZ -----
        mtz_page = QWidget()
        mtz_page_layout = QVBoxLayout(mtz_page)
        self.mtz_result_box = QGroupBox("MTZ result")
        mtz_layout = QVBoxLayout(self.mtz_result_box)
        self.mtz_path_label = QLabel("No MTZ yet.")
        self.mtz_path_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.mtz_path_label.setWordWrap(True)
        mtz_layout.addWidget(self.mtz_path_label)

        mtz_btn_row = QHBoxLayout()
        self.mtz_reveal_btn = QPushButton("Reveal")
        self.mtz_reveal_btn.setEnabled(False)
        self.mtz_reveal_btn.clicked.connect(self._on_reveal_mtz)
        mtz_btn_row.addWidget(self.mtz_reveal_btn)
        self.mtz_copy_btn = QPushButton("Copy path")
        self.mtz_copy_btn.setEnabled(False)
        self.mtz_copy_btn.clicked.connect(self._on_copy_mtz_path)
        mtz_btn_row.addWidget(self.mtz_copy_btn)
        self.mtz_open_viewer_btn = QPushButton("Open in Reflection File Viewer")
        self.mtz_open_viewer_btn.setEnabled(False)
        self.mtz_open_viewer_btn.setToolTip(
            "Open the MTZ in ElMo Reflection File Viewer"
        )
        self.mtz_open_viewer_btn.clicked.connect(self._on_open_mtz_in_viewer)
        mtz_btn_row.addWidget(self.mtz_open_viewer_btn)
        self.mtz_summary_btn = QPushButton("Summary vs resolution")
        self.mtz_summary_btn.setEnabled(False)
        self.mtz_summary_btn.setToolTip(
            "Plot DIALS integrate Summary vs resolution (I/σI, CC, RMSD, counts)"
        )
        self.mtz_summary_btn.clicked.connect(self._on_show_summary_vs_resolution)
        mtz_btn_row.addWidget(self.mtz_summary_btn)
        mtz_btn_row.addStretch(1)
        mtz_layout.addLayout(mtz_btn_row)
        mtz_page_layout.addWidget(self.mtz_result_box)
        mtz_page_layout.addStretch(1)
        self._mtz_tab_index = self.pipeline_tabs.addTab(mtz_page, "5. MTZ")
        self._summary_dialog = None

        # ----- Shared footer -----
        self.status_label = QLabel("Ready.")
        layout.addWidget(self.status_label)

        progress_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        progress_row.addWidget(self.progress_bar, stretch=1)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._on_cancel_busy)
        progress_row.addWidget(self.cancel_button)
        self.open_log_button = QPushButton("Open last DIALS log")
        self.open_log_button.setEnabled(False)
        self.open_log_button.clicked.connect(self._on_open_last_dials_log)
        progress_row.addWidget(self.open_log_button)
        self.show_terminal_button = QPushButton("Show terminal")
        self.show_terminal_button.setToolTip(
            "JDXI-style live terminal viewer (tails dials.*.log)"
        )
        self.show_terminal_button.clicked.connect(self.show_terminal_viewer)
        progress_row.addWidget(self.show_terminal_button)
        layout.addLayout(progress_row)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(120)
        self.log.setFont(
            QFont("Menlo", 11) if QFont("Menlo").exactMatch() else QFont("Courier", 11)
        )
        layout.addWidget(self.log)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        close_btn = buttons.button(QDialogButtonBox.StandardButton.Close)
        if close_btn is not None:
            close_btn.clicked.connect(self.reject)
        layout.addWidget(buttons)

    def _populate_defaults(self) -> None:
        app = self.parent()
        n_frames = int(getattr(app, "num_frames", 0) or 0)
        current = 0
        slider = getattr(app, "frame_slider", None)
        if slider is not None:
            current = int(slider.value())
        self.frame_start.setMaximum(max(n_frames - 1, 0))
        self.frame_end.setMaximum(max(n_frames - 1, 0))
        self.frame_start.setValue(current)
        # Default: a few wedges spaced across the sweep (Mosflm-style).
        if n_frames > 1:
            step = max(1, n_frames // 4)
            self.frame_step.setValue(step)
            self.frame_end.setValue(min(n_frames - 1, current + 3 * step))
        else:
            self.frame_end.setValue(current)
            self.frame_step.setValue(1)

        app = self.parentWidget()
        geometry = None
        try:
            if app is not None and hasattr(app, "build_indexing_geometry"):
                geometry = app.build_indexing_geometry()
                self._source_geometry = geometry
        except Exception:
            geometry = None
        self._use_dials = bool(geometry is not None and geometry.is_hdf5_master)
        dials_ok = find_dials_prefix() is not None
        mosflm = find_ipmosflm()

        if self._use_dials:
            self.backend_label.setText(
                "Backend: DIALS (HDF5 / NeXus — Mosflm cannot PROCESS these masters)."
            )
            self.find_spots_button.setText("Find spots (DIALS)")
            self.run_button.setText("Run DIALS indexing")
            self.integrate_button.setText("Integrate with DIALS")
            self.mosaicity_estimate_button.setEnabled(False)
            self.mosaicity_estimate_button.setToolTip(
                "Mosaicity estimate uses Mosflm; DIALS refines mosaicity during index/refine."
            )
            self.refine_button.setEnabled(False)
            self.refine_button.setToolTip(
                "Unit-cell refine with space group uses Mosflm index2; "
                "DIALS refine runs automatically after indexing."
            )
            master = Path(getattr(app, "current_file", "") or "")
            self.dials_master_label.setText(str(master) if master else "(no master)")
            if master.is_file():
                self._dials_master_path = master.resolve()
                # Prefer the newest existing run beside the master; else a fresh stamp.
                existing = latest_dials_run_dir(self._dials_master_path)
                self._dials_work_dir = (
                    existing
                    if existing is not None
                    else default_dials_work_dir(self._dials_master_path)
                )
            else:
                self._dials_master_path = None
                stem = master.stem if master.name else "dataset"
                self._dials_work_dir = Path.home() / ".qtmosflm" / "dials" / stem
            self.dials_work_dir_edit.setText(str(self._dials_work_dir))
            # Contiguous DIALS range: hide Step; seed integrate spinboxes.
            first_1 = int(self.frame_start.value()) + 1
            last_1 = int(self.frame_end.value()) + 1
            if last_1 < first_1:
                first_1, last_1 = last_1, first_1
            self.integrate_first.setMaximum(max(n_frames, 1))
            self.integrate_last.setMaximum(max(n_frames, 1))
            self.integrate_first.setValue(first_1)
            self.integrate_last.setValue(min(max(n_frames, 1), first_1 + 39))
            if not dials_ok:
                self.status_label.setText(
                    "DIALS not found — install CCP4/DIALS for HDF5 processing."
                )
                self.find_spots_button.setEnabled(False)
                self.full_pipeline_button.setEnabled(False)
            else:
                self.status_label.setText(f"Using DIALS in {find_dials_prefix()}")
                self.run_button.setEnabled(False)
                self.full_pipeline_button.setEnabled(True)
        else:
            self.backend_label.setText("Backend: Mosflm (native image templates).")
            self.find_spots_button.setText("Find spots")
            self.run_button.setText("Run DPS indexing")
            self.integrate_button.setText("Integrate selected wedge")
            if mosflm is None:
                self.status_label.setText(
                    "ipmosflm not found — install CCP4 or set MOSFLM_EXEC."
                )
                self.run_button.setEnabled(False)
                self.find_spots_button.setEnabled(False)
            else:
                self.status_label.setText(f"Using {mosflm}")
                self.run_button.setEnabled(False)

        self._apply_dials_chrome(self._use_dials)
        self._reset_pipeline_checklist()
        self._update_dials_range_hint()
        if self._use_dials:
            self._discover_existing_mtz()
            self._update_summary_button_state()

        if app is not None and hasattr(app, "show_indexing_spots"):
            self.show_spots_checkbox.blockSignals(True)
            self.show_spots_checkbox.setChecked(bool(app.show_indexing_spots))
            self.show_spots_checkbox.blockSignals(False)

    def _settings(self) -> IndexingSettings:
        return IndexingSettings(
            spot_threshold_sigma=float(self.spot_sigma.value()),
            i_sig_i=float(self.i_sig_i.value()),
            max_penalty_to_refine=float(self.max_penalty.value()),
            sigma_cutoff=float(self.refine_sigma.value()),
            mosaic_block_size=float(self.mosaic_block.value()),
        )

    def _dials_settings_from_ui(self) -> DialsSettings:
        """Build :class:`DialsSettings` from the DIALS SpecEditor (or fallback form)."""
        if self._dials_spec_editor is not None:
            return values_to_dials_settings(self._dials_spec_editor.values())
        assert self.dials_nproc is not None
        assert self.dials_d_min is not None
        assert self.dials_min_spot_size is not None
        assert self.dials_gain is not None
        assert self.dials_space_group is not None
        assert self.dials_unit_cell is not None
        d_min_val = float(self.dials_d_min.value())
        gain_val = float(self.dials_gain.value())
        sg = self.dials_space_group.text().strip() or None
        cell = self.dials_unit_cell.text().strip() or None
        return DialsSettings(
            nproc=int(self.dials_nproc.value()),
            min_spot_size=int(self.dials_min_spot_size.value()),
            gain=gain_val if gain_val > 0 else None,
            d_min=d_min_val if d_min_val > 0 else None,
            space_group=sg,
            unit_cell=cell,
        )

    def _resolve_dials_work_dir(self) -> Path:
        """Return the work directory from the UI, creating it if needed."""
        text = self.dials_work_dir_edit.text().strip()
        if text:
            path = Path(text).expanduser()
        else:
            path = self._dials_work_dir
        path.mkdir(parents=True, exist_ok=True)
        self._dials_work_dir = path
        self.dials_work_dir_edit.setText(str(path))
        return path

    def _apply_dials_chrome(self, use_dials: bool) -> None:
        """Show DIALS panels and hide Mosflm-only fields when on HDF5."""
        for w in (
            self.dials_input_box,
            self.dials_settings_box,
            self.dials_pipeline_box,
            self.mtz_result_box,
            self.dials_integrate_range_row,
            self.full_pipeline_button,
            self.dials_range_hint,
            self.open_log_button,
        ):
            w.setVisible(use_dials)
        self.frame_step.setVisible(not use_dials)
        self.frame_step_label.setVisible(not use_dials)
        for w in (
            self.spot_sigma,
            self.spot_sigma_label,
            self.i_sig_i,
            self.i_sig_i_label,
            self.max_penalty,
            self.max_penalty_label,
        ):
            w.setVisible(not use_dials)
        if hasattr(self, "mosaic_box"):
            self.mosaic_box.setVisible(not use_dials)
        if use_dials:
            self.setWindowTitle("Processing — DIALS (HDF5 → MTZ)")
            self.setMinimumSize(820, 720)
        else:
            self.setWindowTitle("Processing — Mosflm DPS")

    def _update_dials_range_hint(self) -> None:
        if not self._use_dials:
            self.dials_range_hint.setText("")
            return
        try:
            rng = self._dials_range_from_ui()
            self.dials_range_hint.setText(
                f"DIALS contiguous images (1-based): "
                f"{rng.first_image}–{rng.last_image} "
                f"({rng.n_images} images; Step is ignored)."
            )
        except Exception:
            self.dials_range_hint.setText("Select a valid Start/End frame range.")

    def _reset_pipeline_checklist(self) -> None:
        self.pipeline_import_label.setText("○ Import —")
        self.pipeline_spots_label.setText("○ Find spots —")
        self.pipeline_index_label.setText("○ Index / refine —")
        self.pipeline_integrate_label.setText("○ Integrate —")
        self.pipeline_export_label.setText("○ Export MTZ —")
        self._mtz_path = None
        self.mtz_path_label.setText("No MTZ yet.")
        self.mtz_reveal_btn.setEnabled(False)
        self.mtz_copy_btn.setEnabled(False)
        self.mtz_open_viewer_btn.setEnabled(False)
        self.mtz_summary_btn.setEnabled(False)

    def _mark_pipeline_step(self, label: QLabel, done: bool, detail: str) -> None:
        mark = "●" if done else "○"
        label.setText(f"{mark} {detail}")

    def _set_busy(self, busy: bool) -> None:
        self.progress_bar.setVisible(busy)
        self.cancel_button.setEnabled(busy)
        self._busy_cancel_requested = False
        if busy:
            self.progress_bar.setRange(0, 0)

    def _on_browse_work_dir(self) -> None:
        start = self.dials_work_dir_edit.text().strip() or str(self._dials_work_dir)
        chosen = QFileDialog.getExistingDirectory(
            self, "DIALS work directory", start
        )
        if chosen:
            self.dials_work_dir_edit.setText(chosen)
            self._dials_work_dir = Path(chosen)
            self._discover_existing_mtz()

    def _on_reveal_work_dir(self) -> None:
        path = self._resolve_dials_work_dir()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _on_reveal_mtz(self) -> None:
        if self._mtz_path is None:
            return
        QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(self._mtz_path.parent))
        )

    def _on_copy_mtz_path(self) -> None:
        if self._mtz_path is None:
            return
        QGuiApplication.clipboard().setText(str(self._mtz_path))
        self.status_label.setText("MTZ path copied to clipboard.")

    def _on_open_mtz_in_viewer(self) -> None:
        if self._mtz_path is None:
            QMessageBox.information(
                self,
                "Reflection File Viewer",
                "No MTZ is available yet. Run Integrate or the full pipeline first.",
            )
            return
        try:
            launch_reflection_viewer(self._mtz_path)
            self.status_label.setText(
                f"Launched Reflection File Viewer for {self._mtz_path.name}"
            )
        except Exception as ex:
            QMessageBox.warning(self, "Reflection File Viewer", str(ex))

    def _set_mtz_available(
        self,
        mtz_path: Path,
        *,
        switch_tab: bool = True,
    ) -> None:
        """Enable MTZ actions once *mtz_path* exists on disk."""
        path = Path(mtz_path).expanduser().resolve()
        if not path.is_file():
            return
        self._mtz_path = path
        self.mtz_path_label.setText(str(path))
        self.mtz_reveal_btn.setEnabled(True)
        self.mtz_copy_btn.setEnabled(True)
        self.mtz_open_viewer_btn.setEnabled(True)
        self._update_summary_button_state()
        if switch_tab and hasattr(self, "_mtz_tab_index"):
            self.pipeline_tabs.setCurrentIndex(self._mtz_tab_index)

    def _update_summary_button_state(self) -> None:
        """Enable Summary vs resolution when dials.integrate.log is available."""
        try:
            work = self._resolve_dials_work_dir()
        except Exception:
            self.mtz_summary_btn.setEnabled(False)
            return
        log = work / "dials.integrate.log"
        self.mtz_summary_btn.setEnabled(log.is_file())

    def _on_show_summary_vs_resolution(self) -> None:
        """Open the matplotlib Summary vs resolution dialog from the integrate log."""
        try:
            work = self._resolve_dials_work_dir()
            dialog = open_summary_vs_resolution_from_work_dir(work, parent=self)
            self._summary_dialog = dialog
        except Exception as ex:
            QMessageBox.warning(self, "Summary vs resolution", str(ex))

    def _maybe_show_summary_vs_resolution(self, work_dir: Path) -> None:
        """Auto-open Summary vs resolution after a successful integrate."""
        try:
            dialog = open_summary_vs_resolution_from_work_dir(work_dir, parent=self)
            self._summary_dialog = dialog
            self.mtz_summary_btn.setEnabled(True)
        except Exception as ex:
            logger.info("Summary vs resolution not shown: %s", ex)
            self._update_summary_button_state()

    def _discover_existing_mtz(self) -> None:
        """Enable the MTZ tab if ``integrated.mtz`` (or similar) already exists."""
        search_dirs: list[Path] = []
        try:
            work = self._resolve_dials_work_dir()
            search_dirs.append(work)
        except Exception:
            pass
        if self._dials_master_path is not None:
            latest = latest_dials_run_dir(self._dials_master_path)
            if latest is not None and latest not in search_dirs:
                search_dirs.insert(0, latest)
            root = dials_run_root_for_master(self._dials_master_path)
            if root.is_dir():
                for run in sorted(
                    (p for p in root.glob("*-dials") if p.is_dir()),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                ):
                    if run not in search_dirs:
                        search_dirs.append(run)
        for work in search_dirs:
            for name in ("integrated.mtz", "dials.mtz", "scaled.mtz"):
                candidate = work / name
                if candidate.is_file():
                    self._set_mtz_available(candidate, switch_tab=False)
                    self._mark_pipeline_step(
                        self.pipeline_export_label,
                        True,
                        f"Export MTZ — {candidate.name} (on disk)",
                    )
                    return

    def _on_open_last_dials_log(self) -> None:
        log_path = self._last_dials_log
        if log_path is None or not Path(log_path).is_file():
            # Prefer newest dials.*.log in work dir.
            work = self._resolve_dials_work_dir()
            logs = sorted(work.glob("dials.*.log"), key=lambda p: p.stat().st_mtime)
            if not logs:
                QMessageBox.information(self, "Logs", f"No dials.*.log in {work}")
                return
            log_path = logs[-1]
        self.follow_dials_log(Path(log_path))

    def show_terminal_viewer(self) -> TerminalLogViewer:
        """Show the JDXI-style terminal log viewer (create if needed)."""
        if self._terminal_viewer is None:
            self._terminal_viewer = TerminalLogViewer(
                self,
                title="DiffractionLab Terminal — DIALS",
            )
            self._terminal_viewer.destroyed.connect(self._on_terminal_destroyed)
        self._terminal_viewer.show()
        self._terminal_viewer.raise_()
        self._terminal_viewer.activateWindow()
        return self._terminal_viewer

    def _on_terminal_destroyed(self, *_args) -> None:
        self._terminal_viewer = None

    def follow_dials_log(self, log_path: Path) -> None:
        """Open the terminal viewer and tail *log_path* in real time."""
        viewer = self.show_terminal_viewer()
        viewer.set_log_file(Path(log_path), reload=True)
        self._last_dials_log = Path(log_path)
        self.open_log_button.setEnabled(True)

    def _begin_dials_log(self, log_name: str = "dials.session.log") -> Path:
        """Prepare work-dir log *log_name*, open terminal on it, return path."""
        work = self._resolve_dials_work_dir()
        log_path = work / log_name
        # Truncate session log for a fresh run; touch so the viewer can tail.
        log_path.write_text("", encoding="utf-8")
        self.follow_dials_log(log_path)
        return log_path

    def _session_log_chunker(self, log_path: Path):
        """Return a thread-safe append callback for streaming DIALS output."""

        def _write(text: str) -> None:
            try:
                with open(log_path, "a", encoding="utf-8") as handle:
                    handle.write(text)
                    handle.flush()
            except OSError:
                pass

        return _write

    def _on_cancel_busy(self) -> None:
        self._busy_cancel_requested = True
        self.status_label.setText(
            "Cancel requested — wait for the current DIALS step to finish "
            "(CLI steps are not interruptible mid-command)."
        )

    def _remember_dials_log(self, work_dir: Path, name: str) -> None:
        path = Path(work_dir) / name
        if path.is_file():
            self._last_dials_log = path
            self.open_log_button.setEnabled(True)

    def _update_mtz_result_ui(self, result: DialsIntegrationResult) -> None:
        self._dials_integration = result
        self._set_mtz_available(result.mtz_path, switch_tab=True)
        self._mark_pipeline_step(
            self.pipeline_integrate_label,
            True,
            f"Integrate — {result.experiments.name}",
        )
        self._mark_pipeline_step(
            self.pipeline_export_label,
            True,
            f"Export MTZ — {result.mtz_path.name}",
        )
        self._remember_dials_log(result.work_dir, "dials.integrate.log")

    def _dials_integrate_range_from_ui(self) -> DialsImageRange:
        first = int(self.integrate_first.value())
        last = int(self.integrate_last.value())
        if last < first:
            first, last = last, first
        return DialsImageRange(first_image=first, last_image=last)

    def _compute_frame_indices(self) -> list[int]:
        """Return the frame index list from the Start/End/Step spin boxes."""
        start = int(self.frame_start.value())
        end = int(self.frame_end.value())
        step = int(self.frame_step.value())
        if end < start:
            start, end = end, start
        return list(range(start, end + 1, step))

    def _prepare_frames(
        self,
    ) -> tuple[list[np.ndarray], IndexingGeometry, list[int], list[tuple[float, float]]]:
        """Load frames and geometry for spot finding / indexing."""
        app = self.parent()
        if app is None or getattr(app, "dataset", None) is None:
            raise ValueError(
                "Open an HDF5 / NeXus master (or image stack) before indexing."
            )
        geometry = app.build_indexing_geometry()
        self._source_geometry = geometry
        indices = self._compute_frame_indices()
        if not indices:
            raise ValueError("No frames selected")
        frames, phis = app.load_frames_for_indexing(indices)
        return frames, geometry, indices, phis

    def _dials_range_from_ui(self) -> DialsImageRange:
        """Convert 0-based Start/End/Step frames to a contiguous 1-based DIALS range."""
        indices = self._compute_frame_indices()
        if not indices:
            raise ValueError("No frames selected")
        # DIALS scan_range is contiguous; use min..max of the selection.
        first = min(indices) + 1
        last = max(indices) + 1
        return DialsImageRange(first_image=first, last_image=last)

    def _on_find_spots(self) -> None:
        if self._use_dials:
            self._on_find_spots_dials()
            return
        try:
            frames, geometry, indices, phis = self._prepare_frames()
        except Exception as ex:
            QMessageBox.critical(self, "Cannot find spots", str(ex))
            return
        self._spotfinding = None
        self._dials_spots = None
        self._dials_index = None
        self._result = None
        self.table.setRowCount(0)
        self._table_solutions = []
        self._selected_frames = list(indices)
        self.log.clear()
        self.log.append(f"Spot finding on frames: {indices}")
        self.log.append(
            f"Beam {geometry.beam_x_mm:.3f}, {geometry.beam_y_mm:.3f} mm; "
            f"σ threshold {self.spot_sigma.value():.1f}"
        )
        self.solutionSelected.emit(None)
        self.find_spots_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.mosaicity_estimate_button.setEnabled(False)

        work_dir = Path.home() / ".qtmosflm" / "work"
        self._spot_thread = QThread(self)
        self._spot_worker = _SpotFindWorker(
            frames, geometry, self._settings(), phis, indices, work_dir
        )
        self._spot_worker.moveToThread(self._spot_thread)
        self._spot_thread.started.connect(self._spot_worker.run)
        self._spot_worker.progress.connect(self.status_label.setText)
        self._spot_worker.finished.connect(self._on_spots_found)
        self._spot_worker.failed.connect(self._on_spots_failed)
        self._spot_worker.finished.connect(self._spot_thread.quit)
        self._spot_worker.failed.connect(self._spot_thread.quit)
        self._spot_thread.finished.connect(self._cleanup_spot_thread)
        self._spot_thread.start()

    def _on_find_spots_dials(self) -> None:
        app = self.parent()
        if app is None or getattr(app, "current_file", None) is None:
            QMessageBox.warning(
                self,
                "No dataset",
                "Open an HDF5 / NeXus master before DIALS spot finding.",
            )
            return
        try:
            geometry = app.build_indexing_geometry()
            self._source_geometry = geometry
            image_range = self._dials_range_from_ui()
        except Exception as ex:
            QMessageBox.critical(self, "Cannot find spots", str(ex))
            return
        self._spotfinding = None
        self._dials_spots = None
        self._dials_index = None
        self._result = None
        self.table.setRowCount(0)
        self._table_solutions = []
        self._selected_frames = self._compute_frame_indices()
        self.log.clear()
        self.log.append(
            f"DIALS spot finding on images "
            f"{image_range.first_image}–{image_range.last_image}"
        )
        self.solutionSelected.emit(None)
        self.find_spots_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.full_pipeline_button.setEnabled(False)
        self._set_busy(True)
        self._reset_pipeline_checklist()

        work_dir = self._resolve_dials_work_dir()
        settings = self._dials_settings_from_ui()
        session_log = self._begin_dials_log()
        self._spot_thread = QThread(self)
        self._spot_worker = _DialsSpotFindWorker(
            Path(app.current_file),
            image_range,
            work_dir,
            settings,
            on_chunk=self._session_log_chunker(session_log),
        )
        self._spot_worker.moveToThread(self._spot_thread)
        self._spot_thread.started.connect(self._spot_worker.run)
        self._spot_worker.progress.connect(self.status_label.setText)
        self._spot_worker.finished.connect(self._on_dials_spots_found)
        self._spot_worker.failed.connect(self._on_spots_failed)
        self._spot_worker.finished.connect(self._spot_thread.quit)
        self._spot_worker.failed.connect(self._spot_thread.quit)
        self._spot_thread.finished.connect(self._cleanup_spot_thread)
        self._spot_thread.start()

    def _on_dials_spots_found(self, spots: DialsSpotFindingResult) -> None:
        self._dials_spots = spots
        min_spots = max(12, self._settings().min_spots_to_index)
        self.spot_summary_label.setText(
            f"DIALS: {spots.n_spots} strong spots on images "
            f"{spots.image_range.first_image}–{spots.image_range.last_image}."
        )
        self.status_label.setText(
            f"DIALS found {spots.n_spots} spots → {spots.strong_refl.name}"
        )
        self.log.append(f"strong.refl: {spots.strong_refl}")
        if spots.centroids_npz is not None:
            self.log.append(f"centroids cache: {spots.centroids_npz}")
        self._mark_pipeline_step(
            self.pipeline_import_label,
            True,
            f"Import — {spots.imported_expt.name}",
        )
        self._mark_pipeline_step(
            self.pipeline_spots_label,
            True,
            f"Find spots — {spots.strong_refl.name} ({spots.n_spots})",
        )
        self._remember_dials_log(spots.work_dir, "dials.find_spots.log")
        self._update_run_button_state()
        self.pipeline_tabs.setCurrentIndex(2)  # Index tab
        self._show_dials_found_spots_overlay()
        if spots.n_spots < min_spots:
            QMessageBox.warning(
                self,
                "Few spots",
                f"Only {spots.n_spots} spots were found. "
                "Widen the frame range or check the images.",
            )

    def _ensure_dials_centroids_cache(self) -> Path | None:
        """Return the centroids ``.npz`` path, exporting from ``strong.refl`` if needed."""
        spots = self._dials_spots
        if spots is None:
            return None
        cache = spots.centroids_npz
        if cache is not None and Path(cache).is_file():
            return Path(cache)
        strong = Path(spots.strong_refl)
        if not strong.is_file():
            return None
        try:
            cache = export_strong_centroids_npz(strong)
        except Exception as ex:  # noqa: BLE001
            logger.exception("Failed to export DIALS centroids")
            self.log.append(f"Could not export spot centroids: {ex}")
            return None
        spots.centroids_npz = cache
        return cache

    def _show_dials_found_spots_overlay(self) -> None:
        """Draw DIALS strong spots for the currently displayed HDF5 frame."""
        cache = self._ensure_dials_centroids_cache()
        if cache is None:
            return
        app = self._viewer()
        if app is None:
            return
        frame_index = 0
        slider = getattr(app, "frame_slider", None)
        if slider is not None:
            frame_index = int(slider.value())
        try:
            centroids = centroids_px_for_image(cache, frame_index)
        except Exception as ex:  # noqa: BLE001
            logger.exception("Failed to load DIALS centroids for frame %s", frame_index)
            self.log.append(f"Spot overlay error: {ex}")
            return
        # Mosflm: found/observed spots are red (predicted=False).
        overlay = [(x, y, False) for x, y in centroids]
        app.set_indexing_spot_overlay(overlay, solution=None)
        if overlay:
            self.status_label.setText(
                f"Showing {len(overlay)} found spots (red) on frame {frame_index}"
            )
        else:
            self.status_label.setText(
                f"No DIALS spots on frame {frame_index} "
                f"(spot-finding range "
                f"{self._dials_spots.image_range.first_image}–"
                f"{self._dials_spots.image_range.last_image})"
            )

    def _cleanup_spot_thread(self) -> None:
        self._spot_worker = None
        if self._spot_thread is not None:
            self._spot_thread.deleteLater()
            self._spot_thread = None
        self.find_spots_button.setEnabled(True)
        self.full_pipeline_button.setEnabled(self._use_dials)
        self._set_busy(False)
        self._update_run_button_state()

    def _on_spots_failed(self, message: str) -> None:
        self.status_label.setText("Spot finding failed.")
        self.log.append(f"ERROR: {message}")
        self.spot_summary_label.setText(message)
        QMessageBox.critical(self, "Spot finding failed", message)

    def _on_spots_found(self, spotfinding: SpotFindingResult) -> None:
        self._spotfinding = spotfinding
        min_spots = self._settings().min_spots_to_index
        self.spot_summary_label.setText(
            f"{spotfinding.n_spots} spots in {len(spotfinding.frame_indices)} "
            f"frame(s) — need ≥{min_spots} for indexing."
        )
        self.status_label.setText(
            f"Found {spotfinding.n_spots} spots (spotfile {spotfinding.spotfile.name})"
        )
        self.log.append(f"Spotfile: {spotfinding.spotfile}")
        self._update_run_button_state()
        self._show_found_spots_overlay()
        if spotfinding.n_spots < min_spots:
            QMessageBox.warning(
                self,
                "Few spots",
                f"Only {spotfinding.n_spots} spots were found. "
                f"iMosflm normally needs at least {min_spots} for indexing. "
                "Adjust the spot σ threshold or frame selection and try again.",
            )

    def _update_run_button_state(self) -> None:
        min_spots = self._settings().min_spots_to_index
        if self._use_dials:
            ok = (
                self._dials_spots is not None
                and self._dials_spots.n_spots >= min_spots
            )
        else:
            ok = (
                self._spotfinding is not None
                and self._spotfinding.n_spots >= min_spots
            )
        self.run_button.setEnabled(ok)

    def _show_found_spots_overlay(self) -> None:
        """Draw all found spots on the main image (before indexing)."""
        if self._spotfinding is None:
            return
        app = self._viewer()
        if app is None:
            return
        pseudo = IndexingResult(
            spotfile=self._spotfinding.spotfile,
            n_spots=self._spotfinding.n_spots,
            suggested_number=None,
            solutions=[],
            refined=[],
            spotlists=self._spotfinding.spotlists,
            frame_indices=self._spotfinding.frame_indices,
            geometry=self._spotfinding.source_geometry or self._spotfinding.geometry,
        )
        frame_index = 0
        slider = getattr(app, "frame_slider", None)
        if slider is not None:
            frame_index = int(slider.value())
        spots = spots_for_frame(pseudo, frame_index)
        if not spots and pseudo.frame_indices:
            frame_index = int(pseudo.frame_indices[0])
            spots = spots_for_frame(pseudo, frame_index)
        geometry = pseudo.geometry
        if geometry is None:
            return
        pixel_size = float(geometry.pixel_size_mm) or 1.0
        # Mosflm: found/observed spots are red (predicted=False).
        overlay = [
            (float(spot.x_mm) / pixel_size, float(spot.y_mm) / pixel_size, False)
            for spot in spots
        ]
        app.set_indexing_spot_overlay(overlay, solution=None)

    def _on_run(self) -> None:
        if self._use_dials:
            self._on_run_dials()
            return
        if self._spotfinding is None:
            QMessageBox.warning(
                self,
                "Find spots first",
                "Run spot finding (step 1) before DPS indexing, as in the iMosflm tutorial.",
            )
            return
        self.run_button.setEnabled(False)
        self.table.setRowCount(0)
        self._table_solutions = []
        self.status_label.setText("Running DPS autoindexing…")
        self.solutionSelected.emit(None)

        work_dir = Path.home() / ".qtmosflm" / "work"
        self._thread = QThread(self)
        self._worker = _IndexWorker(
            self._spotfinding, self._settings(), work_dir
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.status_label.setText)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.start()

    def _on_run_dials(self) -> None:
        if self._dials_spots is None:
            QMessageBox.warning(
                self,
                "Find spots first",
                "Run DIALS spot finding (step 1) before indexing.",
            )
            return
        self.run_button.setEnabled(False)
        self.table.setRowCount(0)
        self._table_solutions = []
        self.status_label.setText("Running DIALS indexing…")
        self.solutionSelected.emit(None)
        self._set_busy(True)
        self.full_pipeline_button.setEnabled(False)
        work_dir = self._resolve_dials_work_dir()
        session_log = self._begin_dials_log()
        self._thread = QThread(self)
        self._worker = _DialsIndexWorker(
            self._dials_spots,
            work_dir,
            self._dials_settings_from_ui(),
            on_chunk=self._session_log_chunker(session_log),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.status_label.setText)
        self._worker.finished.connect(self._on_dials_index_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.start()

    def _on_dials_index_finished(self, indexed: DialsIndexingResult) -> None:
        self._dials_index = indexed
        sol = _dials_to_index_solution(indexed)
        n_spots = self._dials_spots.n_spots if self._dials_spots else 0
        result = IndexingResult(
            spotfile=self._dials_spots.strong_refl if self._dials_spots else Path("."),
            n_spots=n_spots,
            suggested_number=1,
            solutions=[sol],
            refined=[sol] if indexed.refined else [],
            messages=[
                f"DIALS indexed cell {sol.a:.2f} {sol.b:.2f} {sol.c:.2f} "
                f"{sol.alpha:.1f} {sol.beta:.1f} {sol.gamma:.1f} {sol.spacegroup}"
            ],
            geometry=self._source_geometry,
            uses_placeholder_template=False,
        )
        self._result = result
        self.status_label.setText(
            f"DIALS indexing done — {sol.spacegroup}; "
            f"a={sol.a:.2f} b={sol.b:.2f} c={sol.c:.2f}"
            + (" (refined)" if indexed.refined else "")
        )
        self.log.append(f"experiments: {indexed.experiments}")
        self.log.append(
            f"Cell: {sol.a:.3f} {sol.b:.3f} {sol.c:.3f} "
            f"{sol.alpha:.2f} {sol.beta:.2f} {sol.gamma:.2f}  {sol.spacegroup}"
        )
        self._mark_pipeline_step(
            self.pipeline_index_label,
            True,
            f"Index / refine — {indexed.experiments.name}"
            + (" (refined)" if indexed.refined else ""),
        )
        self._remember_dials_log(indexed.work_dir, "dials.index.log")
        # Seed integrate range from spot-finding wedge if still default-ish.
        if self._dials_spots is not None:
            self.integrate_first.setValue(self._dials_spots.image_range.first_image)
            self.integrate_last.setValue(self._dials_spots.image_range.last_image)
        dials_sol = indexed.solution
        if dials_sol.n_indexed is not None:
            self.log.append(
                f"Indexed {dials_sol.n_indexed}"
                + (
                    f" / unindexed {dials_sol.n_unindexed}"
                    if dials_sol.n_unindexed is not None
                    else ""
                )
            )
        elif sol.reflections_used is not None:
            self.log.append(f"Indexed {sol.reflections_used}")
        if indexed.predictions_npz is not None:
            self.log.append(f"predictions cache: {indexed.predictions_npz}")
        self._fill_table(result)
        self.refine_button.setEnabled(False)
        self.mosaicity_estimate_button.setEnabled(False)
        self._update_integration_button_state()
        self.pipeline_tabs.setCurrentIndex(3)  # Integrate tab

    def _cleanup_thread(self) -> None:
        self._worker = None
        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None
        self.full_pipeline_button.setEnabled(self._use_dials)
        self._set_busy(False)
        self._update_run_button_state()
        self._update_integration_button_state()

    def _cleanup_refine_thread(self) -> None:
        self._refine_worker = None
        if self._refine_thread is not None:
            self._refine_thread.deleteLater()
            self._refine_thread = None
        self.refine_button.setEnabled(self._result is not None)
        self.run_button.setEnabled(True)

    def _on_failed(self, message: str) -> None:
        self.status_label.setText("Indexing failed.")
        self.log.append(f"ERROR: {message}")
        QMessageBox.critical(self, "Indexing failed", message)

    def _on_finished(self, result: IndexingResult) -> None:
        self._result = result
        self.status_label.setText(
            f"Done — {result.n_spots} spots; "
            f"{len(result.solutions)} solutions, {len(result.refined)} refined"
            + (
                f"; suggested #{result.suggested_number}"
                if result.suggested_number is not None
                else ""
            )
        )
        for msg in result.messages:
            if msg:
                self.log.append(msg)
        self.log.append(f"Spotfile: {result.spotfile}")
        self._fill_table(result)
        self.refine_button.setEnabled(not result.uses_placeholder_template)
        if result.uses_placeholder_template:
            self.refine_button.setToolTip(
                "Refinement needs a Mosflm-readable image template. "
                "HDF5 masters use an internal placeholder for DPS only."
            )
        else:
            self.refine_button.setToolTip("")
        self._update_mosaicity_button_state()
        self._update_integration_button_state()

    def _fill_table(self, result: IndexingResult) -> None:
        rows: list[IndexSolution] = []
        # One row per solution number: prefer refined, else regularised, else raw.
        by_number: dict[int, IndexSolution] = {}
        for sol in sorted(
            (s for s in result.solutions if s.kind == "raw"),
            key=lambda s: s.penalty,
        ):
            by_number[sol.number] = sol
        for sol in sorted(
            (s for s in result.solutions if s.kind == "reg"),
            key=lambda s: s.penalty,
        ):
            by_number[sol.number] = sol
        for sol in sorted(result.refined, key=lambda s: s.penalty):
            by_number[sol.number] = sol
        rows = sorted(by_number.values(), key=lambda s: (s.penalty, s.number))

        self._table_solutions = rows
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        suggested = result.suggested_number
        select_row = 0
        for row, sol in enumerate(rows):
            beta_gamma = f"{sol.beta:.2f} / {sol.gamma:.2f}"
            values = [
                sol.kind,
                str(sol.number),
                f"{sol.penalty:.2f}",
                sol.lattice,
                sol.spacegroup,
                f"{sol.a:.3f}",
                f"{sol.b:.3f}",
                f"{sol.c:.3f}",
                f"{sol.alpha:.2f}",
                beta_gamma,
            ]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, row)
                if suggested is not None and sol.number == suggested:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                    select_row = row
                if col in (1, 2):
                    item.setTextAlignment(
                        int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    )
                self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents()
        self.table.blockSignals(False)
        if rows:
            self.table.setCurrentCell(select_row, 0)
            self._apply_solution_row(select_row)

    def _on_current_cell_changed(
        self,
        current_row: int,
        _current_column: int,
        _previous_row: int,
        _previous_column: int,
    ) -> None:
        """Update the image overlay when the user selects another solution."""
        self._apply_solution_row(current_row)

    def _on_show_spots_toggled(self, checked: bool) -> None:
        """Sync the main-window indexing-spots checkbox."""
        app = self._viewer()
        if app is None:
            return
        checkbox = getattr(app, "indexing_spots_checkbox", None)
        if checkbox is not None and checkbox.isChecked() != bool(checked):
            checkbox.blockSignals(True)
            checkbox.setChecked(bool(checked))
            checkbox.blockSignals(False)
        if hasattr(app, "show_indexing_spots"):
            app.show_indexing_spots = bool(checked)
        if hasattr(app, "show_image") and getattr(app, "current_image", None) is not None:
            app.show_image(app.current_image)

    def _apply_solution_row(self, row: int) -> None:
        """Push the solution at *row* (and its matched spots) to the main window."""
        if row < 0 or row >= len(self._table_solutions):
            self.solutionSelected.emit(None)
            self._update_parent_overlay(None)
            return
        solution = self._table_solutions[row]
        self._update_spacegroup_combo(solution)
        self._update_mosaicity_button_state()
        self._update_integration_button_state()
        self.solutionSelected.emit(solution)
        n_matched = self._update_parent_overlay(solution)
        if n_matched is None:
            return
        if self._use_dials and self._dials_index is not None:
            self.status_label.setText(
                f"Solution #{solution.number} ({solution.spacegroup}) — "
                f"{n_matched} predicted spots on this image"
            )
            return
        n_found = 0
        if self._result is not None:
            app = self._viewer()
            frame_index = 0
            if app is not None and getattr(app, "frame_slider", None) is not None:
                frame_index = int(app.frame_slider.value())
            n_found = len(spots_for_frame(self._result, frame_index))
        self.status_label.setText(
            f"Solution #{solution.number} ({solution.kind} {solution.lattice} "
            f"{solution.spacegroup}) — {n_matched} matched / {n_found} found"
        )

    def _viewer(self) -> QWidget | None:
        """Return the DiffractionLab main window that owns this dialog."""
        widget = self.parentWidget()
        while widget is not None:
            if hasattr(widget, "set_indexing_spot_overlay"):
                return widget
            widget = widget.parentWidget()
        return None

    def _ensure_dials_predictions_cache(self) -> Path | None:
        """Return the predictions ``.npz`` path, exporting from indexed/refined.refl if needed."""
        indexed = self._dials_index
        if indexed is None:
            return None
        cache = indexed.predictions_npz
        if cache is not None and Path(cache).is_file():
            return Path(cache)
        reflections = Path(indexed.reflections)
        if not reflections.is_file():
            return None
        try:
            cache = export_indexed_predictions_npz(reflections)
        except Exception as ex:  # noqa: BLE001
            logger.exception("Failed to export DIALS predictions")
            self.log.append(f"Could not export index predictions: {ex}")
            return None
        indexed.predictions_npz = cache
        return cache

    def _show_dials_predictions_overlay(
        self, solution: IndexSolution | None = None
    ) -> int | None:
        """Draw DIALS ``xyzcal.px`` predictions for the currently displayed frame."""
        cache = self._ensure_dials_predictions_cache()
        if cache is None:
            return None
        app = self._viewer()
        if app is None:
            return None
        frame_index = 0
        slider = getattr(app, "frame_slider", None)
        if slider is not None:
            frame_index = int(slider.value())
        try:
            predictions = centroids_px_for_image(cache, frame_index)
        except Exception as ex:  # noqa: BLE001
            logger.exception("Failed to load DIALS predictions for frame %s", frame_index)
            self.log.append(f"Prediction overlay error: {ex}")
            return None
        # Mosflm: predicted/indexed fulls are blue (predicted=True).
        overlay = [(x, y, True) for x, y in predictions]
        app.set_indexing_spot_overlay(overlay, solution=solution)
        return len(overlay)

    def _update_parent_overlay(self, solution: IndexSolution | None) -> int | None:
        """Ask the parent viewer to overlay matched spots for *solution*."""
        app = self._viewer()
        if app is None:
            logger.warning("No viewer parent for indexing spot overlay")
            return None
        if solution is None or self._result is None:
            app.set_indexing_spot_overlay(None)
            return None
        if self._use_dials and self._dials_index is not None:
            return self._show_dials_predictions_overlay(solution)
        frame_index = 0
        slider = getattr(app, "frame_slider", None)
        if slider is not None:
            frame_index = int(slider.value())
        spots = spots_for_frame(self._result, frame_index)
        # If the displayed frame was not used for indexing, jump to the first
        # indexed frame so spots are actually visible.
        if not spots and self._result.frame_indices:
            previous_frame = frame_index
            target = int(self._result.frame_indices[0])
            spots = spots_for_frame(self._result, target)
            if spots and slider is not None and previous_frame != target:
                slider.blockSignals(True)
                slider.setValue(target)
                slider.blockSignals(False)
                if hasattr(app, "display_hdf5_image"):
                    app.display_hdf5_image(target)
                if hasattr(app, "frame_label"):
                    app.frame_label.setText(f"Frame: {target}")
                frame_index = target
                self.log.append(
                    f"Showing spots on indexed frame {target} "
                    f"(display was on frame {previous_frame})."
                )
        geometry = self._result.geometry
        if geometry is None:
            try:
                geometry = app.build_indexing_geometry()
            except Exception:
                app.set_indexing_spot_overlay(None)
                return None
        if not spots:
            app.set_indexing_spot_overlay([])
            indexed = ", ".join(str(i) for i in self._result.frame_indices[:12])
            self.status_label.setText(
                f"Solution #{solution.number}: no indexing spots on frame {frame_index} "
                f"(indexed frames: {indexed})"
            )
            return 0
        matched = match_spots_to_solution(
            spots,
            geometry,
            solution,
            hkldev=float(self._settings().hkldev_max),
        )
        # Mosflm colours: found (unmatched) red; predicted/matched blue.
        pixel_size = float(geometry.pixel_size_mm) or 1.0
        overlay = [
            (
                float(spot.x_mm) / pixel_size,
                float(spot.y_mm) / pixel_size,
                bool(is_matched),
            )
            for spot, is_matched in zip(spots, matched)
        ]
        logger.info(
            "Overlay solution #%s on frame %s: %d spots "
            "(%d predicted/blue, %d found/red)",
            solution.number,
            frame_index,
            len(overlay),
            sum(matched),
            len(matched) - sum(matched),
        )
        app.set_indexing_spot_overlay(overlay, solution=solution)
        return int(sum(matched))

    def _integration_geometry(self) -> IndexingGeometry | None:
        """Experiment geometry for integration (native template, not DPS placeholder)."""
        if self._spotfinding is not None and self._spotfinding.source_geometry:
            return self._spotfinding.source_geometry
        if self._source_geometry is not None:
            return self._source_geometry
        if self._result is not None and self._result.geometry is not None:
            geom = self._result.geometry
            if not self._result.uses_placeholder_template:
                return geom
        return None

    def _selected_solution(self) -> IndexSolution | None:
        row = int(self.table.currentRow())
        if row < 0 or row >= len(self._table_solutions):
            return None
        return self._table_solutions[row]

    def _dataset_num_images(self) -> int:
        app = self._viewer()
        if app is None:
            return 0
        return int(getattr(app, "num_frames", 0) or 0)

    def _format_integration_range(self, image_range: IntegrationImageRange) -> str:
        n = image_range.last_image - image_range.first_image + 1
        return (
            f"Images {image_range.first_image}–{image_range.last_image} "
            f"({n} frame(s), {image_range.selection_method})"
        )

    def _preview_heuristic_integration_range(self) -> None:
        solution = self._selected_solution()
        total = self._dataset_num_images()
        if solution is None or total <= 0:
            return
        geom = self._integration_geometry()
        if geom is None:
            return
        osc = max(abs(geom.phi_end - geom.phi_start), 0.01)
        preview = heuristic_integration_range(
            total_images=total,
            mosaicity_deg=float(self.mosaicity_value.value()),
            oscillation_deg=osc,
            first_image=1,
        )
        self.integrate_range_label.setText(
            "Suggested wedge: " + self._format_integration_range(preview)
        )

    def _update_integration_button_state(self) -> None:
        if self._use_dials:
            enabled = self._dials_index is not None
            self.auto_integrate_select_button.setEnabled(enabled)
            self.integrate_button.setEnabled(enabled)
            self.auto_integrate_select_button.setToolTip(
                "Pick a short image wedge from mosaicity / oscillation (DIALS)."
            )
            self.integrate_button.setToolTip("Run dials.integrate + dials.export.")
            if enabled:
                self._preview_dials_integration_range()
            else:
                self.integrate_range_label.setText(
                    "Run DIALS indexing first, then auto-select images and integrate."
                )
            return

        solution = self._selected_solution()
        has_matrix = (
            solution is not None
            and len(solution.matrix) == 9
            and any(solution.matrix)
        )
        geom = self._integration_geometry()
        hdf5 = bool(geom is not None and geom.is_hdf5_master)
        enabled = bool(
            self._result is not None and has_matrix and geom is not None and not hdf5
        )
        self.auto_integrate_select_button.setEnabled(enabled)
        self.integrate_button.setEnabled(enabled)
        if hdf5:
            self.integrate_range_label.setText(
                "Mosflm cannot integrate HDF5 masters; reopen as DIALS backend "
                "or use a native .img/.cbf template."
            )
            tip = "Not supported for HDF5 masters via Mosflm."
            self.auto_integrate_select_button.setToolTip(tip)
            self.integrate_button.setToolTip(tip)
        elif enabled:
            self.auto_integrate_select_button.setToolTip("")
            self.integrate_button.setToolTip("")
            self._preview_heuristic_integration_range()
        else:
            self.integrate_range_label.setText(
                "Select an indexed solution with an A-matrix to enable integration."
            )

    def _preview_dials_integration_range(self) -> None:
        total = self._dataset_num_images()
        geom = self._source_geometry
        if total <= 0 or geom is None:
            return
        osc = max(abs(geom.phi_end - geom.phi_start), 0.01)
        preview = auto_dials_image_range(
            total,
            oscillation_deg=osc,
            mosaicity_deg=float(self.mosaicity_value.value()),
            first_image=1,
        )
        self.integrate_range_label.setText(
            f"Suggested DIALS wedge: images {preview.first_image}–{preview.last_image} "
            f"({preview.n_images} frames)"
        )

    def _on_auto_select_integration(self) -> None:
        if self._use_dials:
            total = self._dataset_num_images()
            geom = self._source_geometry
            if total <= 0 or geom is None:
                return
            osc = max(abs(geom.phi_end - geom.phi_start), 0.01)
            dials_range = auto_dials_image_range(
                total,
                oscillation_deg=osc,
                mosaicity_deg=float(self.mosaicity_value.value()),
                first_image=1,
            )
            self._integration_range = IntegrationImageRange(
                first_image=dials_range.first_image,
                last_image=dials_range.last_image,
                selection_method="dials_heuristic",
            )
            self.integrate_range_label.setText(
                self._format_integration_range(self._integration_range)
            )
            self.status_label.setText("DIALS integration image range selected.")
            self.log.append(self._format_integration_range(self._integration_range))
            self._update_integration_button_state()
            return

        solution = self._selected_solution()
        geometry = self._integration_geometry()
        total = self._dataset_num_images()
        if solution is None or geometry is None or total <= 0:
            return
        spacegroup = normalize_spacegroup(
            self.spacegroup_combo.currentText().strip() or solution.spacegroup
        )
        self.auto_integrate_select_button.setEnabled(False)
        self.integrate_button.setEnabled(False)
        self.status_label.setText("Selecting integration images via Mosflm…")
        work_dir = Path.home() / ".qtmosflm" / "work"
        settings = self._settings()

        class _SelectWorker(QObject):
            finished = Signal(object)
            failed = Signal(str)

            def __init__(self) -> None:
                super().__init__()

            def run(self) -> None:
                try:
                    engine = IndexingEngine(work_dir=work_dir, settings=settings)
                    image_range = engine.auto_select_integration_range(
                        geometry,
                        solution,
                        spacegroup=spacegroup,
                        mosaicity_deg=float(self.mosaicity_value.value()),
                        total_images=total,
                    )
                    self.finished.emit(image_range)
                except Exception as ex:  # noqa: BLE001
                    self.failed.emit(str(ex))

        thread = QThread(self)
        worker = _SelectWorker()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        def _done(image_range: IntegrationImageRange) -> None:
            self._integration_range = image_range
            self.integrate_range_label.setText(
                self._format_integration_range(image_range)
            )
            self.status_label.setText("Integration image range selected.")
            self.log.append(self._format_integration_range(image_range))
            thread.quit()

        def _fail(msg: str) -> None:
            self.status_label.setText("Auto-select failed.")
            self.log.append(f"INTEGRATION SELECT ERROR: {msg}")
            QMessageBox.warning(self, "Auto-select images", msg)
            thread.quit()

        worker.finished.connect(_done)
        worker.failed.connect(_fail)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(lambda: self._update_integration_button_state())
        thread.start()

    def _on_integrate(self) -> None:
        if self._use_dials:
            self._on_integrate_dials()
            return
        solution = self._selected_solution()
        geometry = self._integration_geometry()
        total = self._dataset_num_images()
        if solution is None or geometry is None or total <= 0:
            QMessageBox.warning(
                self,
                "Integration",
                "Load a dataset, index, and select a solution first.",
            )
            return
        if geometry.is_hdf5_master:
            QMessageBox.warning(
                self,
                "Integration not available",
                "ipmosflm crashes when PROCESS opens an HDF5 master from DiffractionLab. "
                "This dialog should use the DIALS backend for HDF5 — reopen Index "
                "after loading the master, or use a native image template.",
            )
            return
        spacegroup = normalize_spacegroup(
            self.spacegroup_combo.currentText().strip() or solution.spacegroup
        )
        app = self._viewer()
        phi_start = geometry.phi_start
        osc = max(abs(geometry.phi_end - geometry.phi_start), 0.01)
        if app is not None and hasattr(app, "phi_range_for_frame"):
            idx = max(0, int(self._integration_range.first_image - 1)
                      if self._integration_range else 0)
            phi_start, phi_end = app.phi_range_for_frame(idx)
            osc = max(abs(phi_end - phi_start), 0.01)

        self.integrate_button.setEnabled(False)
        self.auto_integrate_select_button.setEnabled(False)
        self.find_spots_button.setEnabled(False)
        self.run_button.setEnabled(False)
        work_dir = Path.home() / ".qtmosflm" / "work"
        settings = self._settings()
        self._integrate_thread = QThread(self)
        self._integrate_worker = _IntegrationWorker(
            geometry,
            solution,
            spacegroup,
            settings,
            work_dir,
            total_images=total,
            mosaicity_deg=float(self.mosaicity_value.value()),
            phi_start=phi_start,
            oscillation_deg=osc,
            image_range=self._integration_range,
        )
        self._integrate_worker.moveToThread(self._integrate_thread)
        self._integrate_thread.started.connect(self._integrate_worker.run)
        self._integrate_worker.progress.connect(self.status_label.setText)
        self._integrate_worker.finished.connect(self._on_integrate_finished)
        self._integrate_worker.failed.connect(self._on_integrate_failed)
        self._integrate_worker.finished.connect(self._integrate_thread.quit)
        self._integrate_worker.failed.connect(self._integrate_thread.quit)
        self._integrate_thread.finished.connect(self._cleanup_integrate_thread)
        self._integrate_thread.start()

    def _on_integrate_dials(self) -> None:
        if self._dials_index is None:
            QMessageBox.warning(
                self,
                "Integration",
                "Run DIALS indexing before integration.",
            )
            return
        try:
            dials_range = self._dials_integrate_range_from_ui()
        except Exception as ex:
            QMessageBox.critical(self, "Integration", str(ex))
            return
        self._integration_range = IntegrationImageRange(
            first_image=dials_range.first_image,
            last_image=dials_range.last_image,
            selection_method="dials_ui",
        )

        self.integrate_button.setEnabled(False)
        self.auto_integrate_select_button.setEnabled(False)
        self.find_spots_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.full_pipeline_button.setEnabled(False)
        self._set_busy(True)
        work_dir = self._resolve_dials_work_dir()
        session_log = self._begin_dials_log()
        self._integrate_thread = QThread(self)
        self._integrate_worker = _DialsIntegrateWorker(
            self._dials_index,
            dials_range,
            work_dir,
            self._dials_settings_from_ui(),
            on_chunk=self._session_log_chunker(session_log),
        )
        self._integrate_worker.moveToThread(self._integrate_thread)
        self._integrate_thread.started.connect(self._integrate_worker.run)
        self._integrate_worker.progress.connect(self.status_label.setText)
        self._integrate_worker.finished.connect(self._on_dials_integrate_finished)
        self._integrate_worker.failed.connect(self._on_integrate_failed)
        self._integrate_worker.finished.connect(self._integrate_thread.quit)
        self._integrate_worker.failed.connect(self._integrate_thread.quit)
        self._integrate_thread.finished.connect(self._cleanup_integrate_thread)
        self._integrate_thread.start()

    def _on_dials_integrate_finished(self, result: DialsIntegrationResult) -> None:
        self._integration_range = IntegrationImageRange(
            first_image=result.image_range.first_image,
            last_image=result.image_range.last_image,
            selection_method="dials",
        )
        nref = result.n_reflections or "?"
        self.integrate_range_label.setText(
            self._format_integration_range(self._integration_range)
            + f" → {result.mtz_path.name} ({nref} reflections)"
        )
        self.status_label.setText(f"DIALS integration complete — {result.mtz_path}")
        self.log.append(f"DIALS MTZ: {result.mtz_path}")
        if result.n_reflections is not None:
            self.log.append(f"Integrated reflections: {result.n_reflections}")
        self._update_mtz_result_ui(result)
        self._maybe_show_summary_vs_resolution(result.work_dir)

    def _on_run_full_pipeline(self) -> None:
        """One-shot import → find spots → index → integrate → MTZ."""
        app = self.parent()
        if app is None or getattr(app, "current_file", None) is None:
            QMessageBox.warning(
                self,
                "No dataset",
                "Open an HDF5 / NeXus master before running the DIALS pipeline.",
            )
            return
        try:
            geometry = app.build_indexing_geometry()
            self._source_geometry = geometry
            image_range = self._dials_range_from_ui()
        except Exception as ex:
            QMessageBox.critical(self, "DIALS pipeline", str(ex))
            return

        self._spotfinding = None
        self._dials_spots = None
        self._dials_index = None
        self._dials_integration = None
        self._result = None
        self.table.setRowCount(0)
        self._table_solutions = []
        self._reset_pipeline_checklist()
        self.log.clear()
        self.log.append(
            f"Full DIALS pipeline on images "
            f"{image_range.first_image}–{image_range.last_image}"
        )
        self.solutionSelected.emit(None)

        self.find_spots_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.integrate_button.setEnabled(False)
        self.full_pipeline_button.setEnabled(False)
        self._set_busy(True)

        work_dir = self._resolve_dials_work_dir()
        session_log = self._begin_dials_log()
        self._pipeline_thread = QThread(self)
        self._pipeline_worker = _DialsFullPipelineWorker(
            Path(app.current_file),
            image_range,
            work_dir,
            self._dials_settings_from_ui(),
            on_chunk=self._session_log_chunker(session_log),
        )
        self._pipeline_worker.moveToThread(self._pipeline_thread)
        self._pipeline_thread.started.connect(self._pipeline_worker.run)
        self._pipeline_worker.progress.connect(self.status_label.setText)
        self._pipeline_worker.finished.connect(self._on_full_pipeline_finished)
        self._pipeline_worker.failed.connect(self._on_full_pipeline_failed)
        self._pipeline_worker.finished.connect(self._pipeline_thread.quit)
        self._pipeline_worker.failed.connect(self._pipeline_thread.quit)
        self._pipeline_thread.finished.connect(self._cleanup_pipeline_thread)
        self._pipeline_thread.start()

    def _on_full_pipeline_finished(self, payload: object) -> None:
        spots, indexed, integ = payload  # type: ignore[misc]
        self._on_dials_spots_found(spots)
        self._on_dials_index_finished(indexed)
        if integ is not None:
            self._on_dials_integrate_finished(integ)
        self.status_label.setText(
            f"Full DIALS pipeline complete — {integ.mtz_path if integ else 'no MTZ'}"
        )

    def _on_full_pipeline_failed(self, message: str) -> None:
        self.status_label.setText("DIALS full pipeline failed.")
        self.log.append(f"PIPELINE ERROR: {message}")
        QMessageBox.critical(self, "DIALS pipeline failed", message)

    def _cleanup_pipeline_thread(self) -> None:
        self._pipeline_worker = None
        if self._pipeline_thread is not None:
            self._pipeline_thread.deleteLater()
            self._pipeline_thread = None
        self.find_spots_button.setEnabled(True)
        self.full_pipeline_button.setEnabled(self._use_dials)
        self._set_busy(False)
        self._update_run_button_state()
        self._update_integration_button_state()

    def _cleanup_integrate_thread(self) -> None:
        self._integrate_worker = None
        if self._integrate_thread is not None:
            self._integrate_thread.deleteLater()
            self._integrate_thread = None
        self.find_spots_button.setEnabled(True)
        self.full_pipeline_button.setEnabled(self._use_dials)
        self._set_busy(False)
        self._update_run_button_state()
        self._update_integration_button_state()

    def _on_integrate_failed(self, message: str) -> None:
        self.status_label.setText("Integration failed.")
        self.log.append(f"INTEGRATION ERROR: {message}")
        QMessageBox.critical(self, "Integration failed", message)

    def _on_integrate_finished(self, result: IntegrationResult) -> None:
        self._integration_result = result
        self._integration_range = result.image_range
        self.integrate_range_label.setText(
            self._format_integration_range(result.image_range)
            + f" → {result.mtz_path.name}"
        )
        self.status_label.setText(
            f"Integration complete — {result.n_images} image(s), MTZ "
            f"{result.mtz_path}"
        )
        self.log.append(
            f"Integration: {self._format_integration_range(result.image_range)}"
        )
        self.log.append(f"MTZ: {result.mtz_path}")
        if result.message:
            self.log.append(result.message)
        self._set_mtz_available(result.mtz_path, switch_tab=True)

    def _update_mosaicity_button_state(self) -> None:
        """Enable mosaicity estimation when a solution with an A-matrix is selected."""
        if self._result is None or self._result.uses_placeholder_template:
            self.mosaicity_estimate_button.setEnabled(False)
            if self._result is not None and self._result.uses_placeholder_template:
                self.mosaicity_estimate_button.setToolTip(
                    "Not available with HDF5 placeholder images."
                )
            return
        row = int(self.table.currentRow())
        if row < 0 or row >= len(self._table_solutions):
            self.mosaicity_estimate_button.setEnabled(False)
            return
        solution = self._table_solutions[row]
        has_matrix = len(solution.matrix) == 9 and any(solution.matrix)
        self.mosaicity_estimate_button.setEnabled(has_matrix)
        self.mosaicity_estimate_button.setToolTip(
            "Estimate mosaic spread for the selected indexing solution."
            if has_matrix
            else "Select an indexed solution with an A-matrix."
        )

    def _on_estimate_mosaicity(self) -> None:
        if self._result is None:
            return
        row = int(self.table.currentRow())
        if row < 0 or row >= len(self._table_solutions):
            return
        solution = self._table_solutions[row]
        geometry = self._result.geometry or self._source_geometry
        if geometry is None or len(solution.matrix) != 9:
            QMessageBox.warning(
                self,
                "Mosaicity",
                "Select an indexing solution with a valid A-matrix.",
            )
            return
        spacegroup = normalize_spacegroup(
            self.spacegroup_combo.currentText().strip()
            or solution.spacegroup
        )
        phi_start = geometry.phi_start
        phi_end = geometry.phi_end
        phi_scan_end = phi_end + max(phi_end - phi_start, 0.01)
        if self._spotfinding and len(self._spotfinding.spotlists) >= 2:
            sl0 = self._spotfinding.spotlists[0]
            sl1 = self._spotfinding.spotlists[1]
            phi_start = float(sl0.phi_start)
            phi_end = float(sl0.phi_end)
            phi_scan_end = float(sl1.phi_end)

        self.mosaicity_estimate_button.setEnabled(False)
        self.find_spots_button.setEnabled(False)
        self.run_button.setEnabled(False)
        work_dir = Path.home() / ".qtmosflm" / "work"
        settings = self._settings()
        self._mosaic_thread = QThread(self)
        self._mosaic_worker = _MosaicityWorker(
            geometry,
            solution,
            spacegroup,
            settings,
            work_dir,
            phi_start=phi_start,
            phi_end=phi_end,
            phi_scan_end=phi_scan_end,
            image_number=geometry.image_number,
        )
        self._mosaic_worker.moveToThread(self._mosaic_thread)
        self._mosaic_thread.started.connect(self._mosaic_worker.run)
        self._mosaic_worker.progress.connect(self.status_label.setText)
        self._mosaic_worker.finished.connect(self._on_mosaicity_finished)
        self._mosaic_worker.failed.connect(self._on_mosaicity_failed)
        self._mosaic_worker.finished.connect(self._mosaic_thread.quit)
        self._mosaic_worker.failed.connect(self._mosaic_thread.quit)
        self._mosaic_thread.finished.connect(self._cleanup_mosaic_thread)
        self._mosaic_thread.start()

    def _cleanup_mosaic_thread(self) -> None:
        self._mosaic_worker = None
        if self._mosaic_thread is not None:
            self._mosaic_thread.deleteLater()
            self._mosaic_thread = None
        self.find_spots_button.setEnabled(True)
        self._update_run_button_state()
        self._update_mosaicity_button_state()

    def _on_mosaicity_failed(self, message: str) -> None:
        self.status_label.setText("Mosaicity estimation failed.")
        self.log.append(f"MOSAICITY ERROR: {message}")
        QMessageBox.critical(self, "Mosaicity estimation failed", message)

    def _on_mosaicity_finished(self, estimation: MosaicityEstimation) -> None:
        if self._result is None:
            return
        self._result.estimated_mosaicity = estimation.mosaicity_deg
        self._result.mosaicity_scan = list(estimation.scan_points)
        self.mosaicity_value.setValue(float(estimation.mosaicity_deg))
        self.status_label.setText(
            f"Mosaic spread estimated: {estimation.mosaicity_deg:.3f}°"
        )
        self.log.append(
            f"Mosaicity: {estimation.mosaicity_deg:.4f}° "
            f"({len(estimation.scan_points)} trial point(s))"
        )
        for mosaic, intensity in estimation.scan_points[-8:]:
            self.log.append(f"  trial {mosaic:.3f}° → intensity {intensity:.1g}")

    def _update_spacegroup_combo(self, solution: IndexSolution) -> None:
        """Populate the space-group combo for the solution's Bravais lattice."""
        groups = spacegroups_for_lattice(solution.lattice)
        self.spacegroup_combo.blockSignals(True)
        self.spacegroup_combo.clear()
        self.spacegroup_combo.addItems(list(groups))
        pick = solution.spacegroup
        if pick not in groups:
            pick = groups[0] if groups else "P1"
        index = self.spacegroup_combo.findText(pick)
        if index >= 0:
            self.spacegroup_combo.setCurrentIndex(index)
        else:
            self.spacegroup_combo.setEditText(pick)
        self.spacegroup_combo.blockSignals(False)

    def _cell_for_refinement(
        self,
        solution: IndexSolution,
    ) -> tuple[float, float, float, float, float, float]:
        """Starting cell for ``index2`` (regularised preselection when available)."""
        if self._result is not None:
            pre = preselection_solution(self._result, solution.number, prefer="reg")
            if pre is not None:
                return pre.cell_tuple
        return solution.cell_tuple

    def _on_refine(self) -> None:
        if self._result is None or not self._table_solutions:
            return
        row = int(self.table.currentRow())
        if row < 0 or row >= len(self._table_solutions):
            QMessageBox.warning(self, "Refine", "Select a solution to refine.")
            return
        solution = self._table_solutions[row]
        geometry = self._result.geometry or self._source_geometry
        if geometry is None:
            QMessageBox.warning(
                self,
                "Refine",
                "No experiment geometry is available for refinement.",
            )
            return
        if self._result.uses_placeholder_template:
            QMessageBox.warning(
                self,
                "Refine not available",
                "Unit-cell refinement cannot be run on HDF5 masters in DiffractionLab "
                "because Mosflm re-opens detector images during index2. DPS "
                "preselection still works from the spot file.\n\n"
                "Use a native Mosflm image template, or refine in iMosflm.",
            )
            return
        spacegroup = normalize_spacegroup(
            self.spacegroup_combo.currentText().strip()
        )
        if not spacegroup:
            QMessageBox.warning(self, "Refine", "Choose a space group.")
            return
        cell = self._cell_for_refinement(solution)
        self.run_button.setEnabled(False)
        self.refine_button.setEnabled(False)
        self.status_label.setText(
            f"Refining solution #{solution.number} with {spacegroup}…"
        )
        work_dir = Path.home() / ".qtmosflm" / "work"
        settings = self._settings()
        self._refine_thread = QThread(self)
        self._refine_worker = _RefineWorker(
            geometry,
            self._result.spotfile,
            solution.number,
            cell,
            spacegroup,
            settings,
            work_dir,
        )
        self._refine_worker.moveToThread(self._refine_thread)
        self._refine_thread.started.connect(self._refine_worker.run)
        self._refine_worker.progress.connect(self.status_label.setText)
        self._refine_worker.finished.connect(self._on_refine_finished)
        self._refine_worker.failed.connect(self._on_refine_failed)
        self._refine_worker.finished.connect(self._refine_thread.quit)
        self._refine_worker.failed.connect(self._refine_thread.quit)
        self._refine_thread.finished.connect(self._cleanup_refine_thread)
        self._refine_thread.start()

    def _on_refine_failed(self, message: str) -> None:
        self.status_label.setText("Refinement failed.")
        self.log.append(f"REFINE ERROR: {message}")
        QMessageBox.critical(self, "Refinement failed", message)

    def _on_refine_finished(self, refined: IndexSolution) -> None:
        if self._result is None:
            return
        merged = IndexSolution(
            number=refined.number,
            kind="ref",
            penalty=refined.penalty,
            lattice=refined.lattice or self._table_solutions[int(self.table.currentRow())].lattice,
            spacegroup=refined.spacegroup,
            a=refined.a,
            b=refined.b,
            c=refined.c,
            alpha=refined.alpha,
            beta=refined.beta,
            gamma=refined.gamma,
            matrix=refined.matrix,
            reflections_used=refined.reflections_used,
            spot_deviation_pos=refined.spot_deviation_pos,
            spot_deviation_phi=refined.spot_deviation_phi,
            beam_x=refined.beam_x,
            beam_y=refined.beam_y,
            volume=refined.volume,
        )
        replaced = False
        for idx, existing in enumerate(self._result.refined):
            if existing.number == merged.number:
                self._result.refined[idx] = merged
                replaced = True
                break
        if not replaced:
            self._result.refined.append(merged)
        self.log.append(
            f"Refined #{merged.number} {merged.spacegroup}: "
            f"a={merged.a:.3f} b={merged.b:.3f} c={merged.c:.3f}; "
            f"Nref={merged.reflections_used or '?'}"
        )
        self._fill_table(self._result)
        for row, sol in enumerate(self._table_solutions):
            if sol.number == merged.number:
                self.table.setCurrentCell(row, 0)
                self._apply_solution_row(row)
                break
        self.status_label.setText(
            f"Refined solution #{merged.number} ({merged.spacegroup})"
        )

    def refresh_overlay_for_current_frame(self) -> None:
        """Recompute the spot overlay after the main window changes frame."""
        if self._table_solutions:
            row = int(self.table.currentRow())
            if 0 <= row < len(self._table_solutions):
                self._apply_solution_row(row)
                return
        if self._dials_spots is not None and self._dials_spots.centroids_npz is not None:
            self._show_dials_found_spots_overlay()
            return
        if self._spotfinding is not None:
            self._show_found_spots_overlay()

    @property
    def indexing_result(self) -> IndexingResult | None:
        """Most recent indexing result, if any.

        Named to avoid shadowing :meth:`QDialog.result`.
        """
        return self._result

    def closeEvent(self, event) -> None:  # noqa: N802
        """Clear the spot overlay when the indexing window closes."""
        self.solutionSelected.emit(None)
        app = self.parent()
        if app is not None and hasattr(app, "set_indexing_spot_overlay"):
            app.set_indexing_spot_overlay(None)
        super().closeEvent(event)
