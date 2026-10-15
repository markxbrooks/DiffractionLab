"""Indexing dialog for qtMosflm (findspots → DPS → refine)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from imosflm.dials_engine import (
    DialsEngine,
    DialsImageRange,
    DialsIndexingResult,
    DialsIntegrationResult,
    DialsSettings,
    DialsSpotFindingResult,
    auto_dials_image_range,
    find_dials_prefix,
)
from imosflm.indexing import (
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
from imosflm.mosflm_client import find_ipmosflm

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
    ) -> None:
        super().__init__()
        self._master_path = master_path
        self._image_range = image_range
        self._work_dir = work_dir
        self._settings = settings or DialsSettings()

    def run(self) -> None:
        try:
            engine = DialsEngine(work_dir=self._work_dir, settings=self._settings)
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
    ) -> None:
        super().__init__()
        self._spots = spots
        self._work_dir = work_dir
        self._settings = settings or DialsSettings()

    def run(self) -> None:
        try:
            engine = DialsEngine(work_dir=self._work_dir, settings=self._settings)
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
    ) -> None:
        super().__init__()
        self._indexed = indexed
        self._image_range = image_range
        self._work_dir = work_dir
        self._settings = settings or DialsSettings()

    def run(self) -> None:
        try:
            engine = DialsEngine(work_dir=self._work_dir, settings=self._settings)
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
    """UI for Mosflm-style autoindexing of the currently loaded dataset."""

    solutionSelected = Signal(object)  # IndexSolution | None

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("Indexing — spots, DPS, mosaicity")
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
        self._use_dials: bool = False
        self._selected_frames: list[int] = []
        self._source_geometry: IndexingGeometry | None = None
        self._build_ui()
        self._populate_defaults()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        form_box = QGroupBox("Frames & settings")
        form = QFormLayout(form_box)

        self.frame_start = QSpinBox()
        self.frame_start.setRange(0, 999999)
        self.frame_end = QSpinBox()
        self.frame_end.setRange(0, 999999)
        self.frame_step = QSpinBox()
        self.frame_step.setRange(1, 9999)
        self.frame_step.setValue(90)

        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Start"))
        frame_row.addWidget(self.frame_start)
        frame_row.addWidget(QLabel("End"))
        frame_row.addWidget(self.frame_end)
        frame_row.addWidget(QLabel("Step"))
        frame_row.addWidget(self.frame_step)
        form.addRow("Frames", frame_row)

        self.spot_sigma = QDoubleSpinBox()
        self.spot_sigma.setRange(1.0, 50.0)
        self.spot_sigma.setDecimals(1)
        self.spot_sigma.setSingleStep(0.5)
        self.spot_sigma.setValue(5.0)
        form.addRow("Spot σ threshold", self.spot_sigma)

        self.i_sig_i = QDoubleSpinBox()
        self.i_sig_i.setRange(1.0, 200.0)
        self.i_sig_i.setDecimals(1)
        self.i_sig_i.setValue(5.0)
        form.addRow("DPS I/σI", self.i_sig_i)

        self.max_penalty = QDoubleSpinBox()
        self.max_penalty.setRange(0.0, 500.0)
        self.max_penalty.setDecimals(1)
        self.max_penalty.setValue(50.0)
        form.addRow("Max penalty to refine", self.max_penalty)

        layout.addWidget(form_box)

        self.backend_label = QLabel("")
        self.backend_label.setWordWrap(True)
        layout.addWidget(self.backend_label)

        spot_box = QGroupBox("1. Spot finding")
        spot_layout = QHBoxLayout(spot_box)
        self.find_spots_button = QPushButton("Find spots")
        self.find_spots_button.clicked.connect(self._on_find_spots)
        spot_layout.addWidget(self.find_spots_button)
        self.spot_summary_label = QLabel(
            "Search selected frames, then check spots on the image before indexing."
        )
        self.spot_summary_label.setWordWrap(True)
        spot_layout.addWidget(self.spot_summary_label, stretch=1)
        layout.addWidget(spot_box)

        self.run_button = QPushButton("2. Run indexing")
        self.run_button.setEnabled(False)
        self.run_button.setToolTip(
            "Requires spots from step 1 (Mosflm DPS or DIALS)."
        )
        self.run_button.clicked.connect(self._on_run)
        layout.addWidget(self.run_button)

        self.status_label = QLabel("Ready.")
        layout.addWidget(self.status_label)

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
        # currentCellChanged fires reliably when the user clicks another row.
        self.table.currentCellChanged.connect(self._on_current_cell_changed)
        layout.addWidget(self.table, stretch=1)

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
        layout.addWidget(refine_box)

        mosaic_box = QGroupBox("3. Mosaicity estimation (§5.6)")
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
        layout.addWidget(mosaic_box)

        integrate_box = QGroupBox("4. Trial integration")
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
        self.integrate_button = QPushButton("Integrate selected wedge")
        self.integrate_button.setEnabled(False)
        self.integrate_button.clicked.connect(self._on_integrate)
        integrate_layout.addWidget(self.integrate_button)
        layout.addWidget(integrate_box)

        spot_row = QHBoxLayout()
        self.show_spots_checkbox = QCheckBox("Show indexing spots on image")
        self.show_spots_checkbox.setChecked(True)
        self.show_spots_checkbox.toggled.connect(self._on_show_spots_toggled)
        spot_row.addWidget(self.show_spots_checkbox)
        spot_row.addStretch(1)
        layout.addLayout(spot_row)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(120)
        self.log.setFont(QFont("Menlo", 11) if QFont("Menlo").exactMatch() else QFont("Courier", 11))
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
            self.run_button.setText("2. Run DIALS indexing")
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
            if not dials_ok:
                self.status_label.setText(
                    "DIALS not found — install CCP4/DIALS for HDF5 processing."
                )
                self.find_spots_button.setEnabled(False)
            else:
                self.status_label.setText(f"Using DIALS in {find_dials_prefix()}")
                self.run_button.setEnabled(False)
        else:
            self.backend_label.setText("Backend: Mosflm (native image templates).")
            self.find_spots_button.setText("Find spots")
            self.run_button.setText("2. Run DPS indexing")
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

        work_dir = Path.home() / ".qtmosflm" / "dials"
        self._spot_thread = QThread(self)
        self._spot_worker = _DialsSpotFindWorker(
            Path(app.current_file),
            image_range,
            work_dir,
            DialsSettings(nproc=4),
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
        self._update_run_button_state()
        if spots.n_spots < min_spots:
            QMessageBox.warning(
                self,
                "Few spots",
                f"Only {spots.n_spots} spots were found. "
                "Widen the frame range or check the images.",
            )

    def _cleanup_spot_thread(self) -> None:
        self._spot_worker = None
        if self._spot_thread is not None:
            self._spot_thread.deleteLater()
            self._spot_thread = None
        self.find_spots_button.setEnabled(True)
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
        overlay = [
            (float(spot.x_mm) / pixel_size, float(spot.y_mm) / pixel_size, True)
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
        work_dir = Path.home() / ".qtmosflm" / "dials"
        self._thread = QThread(self)
        self._worker = _DialsIndexWorker(
            self._dials_spots, work_dir, DialsSettings(nproc=4)
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
        self._fill_table(result)
        self.refine_button.setEnabled(False)
        self.mosaicity_estimate_button.setEnabled(False)
        self._update_integration_button_state()

    def _cleanup_thread(self) -> None:
        self._worker = None
        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None
        self._update_run_button_state()

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
        n_found = 0
        if self._result is not None:
            app = self._viewer()
            frame_index = 0
            if app is not None and getattr(app, "frame_slider", None) is not None:
                frame_index = int(app.frame_slider.value())
            n_found = len(spots_for_frame(self._result, frame_index))
        if n_matched is not None:
            self.status_label.setText(
                f"Solution #{solution.number} ({solution.kind} {solution.lattice} "
                f"{solution.spacegroup}) — {n_matched} matched / {n_found} found"
            )

    def _viewer(self) -> QWidget | None:
        """Return the qtMosflm main window that owns this dialog."""
        widget = self.parentWidget()
        while widget is not None:
            if hasattr(widget, "set_indexing_spot_overlay"):
                return widget
            widget = widget.parentWidget()
        return None

    def _update_parent_overlay(self, solution: IndexSolution | None) -> int | None:
        """Ask the parent viewer to overlay matched spots for *solution*."""
        app = self._viewer()
        if app is None:
            logger.warning("No viewer parent for indexing spot overlay")
            return None
        if solution is None or self._result is None:
            app.set_indexing_spot_overlay(None)
            return None
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
        # Always draw observed spots; matched ones are coloured green.
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
            "Overlay solution #%s on frame %s: %d spots (%d matched)",
            solution.number,
            frame_index,
            len(overlay),
            sum(matched),
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
                "ipmosflm crashes when PROCESS opens an HDF5 master from qtMosflm. "
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
        total = self._dataset_num_images()
        geom = self._source_geometry
        if total <= 0 or geom is None:
            return
        if self._integration_range is not None:
            dials_range = DialsImageRange(
                self._integration_range.first_image,
                self._integration_range.last_image,
            )
        else:
            osc = max(abs(geom.phi_end - geom.phi_start), 0.01)
            dials_range = auto_dials_image_range(
                total,
                oscillation_deg=osc,
                mosaicity_deg=float(self.mosaicity_value.value()),
            )
            self._integration_range = IntegrationImageRange(
                first_image=dials_range.first_image,
                last_image=dials_range.last_image,
                selection_method="dials_heuristic",
            )

        self.integrate_button.setEnabled(False)
        self.auto_integrate_select_button.setEnabled(False)
        self.find_spots_button.setEnabled(False)
        self.run_button.setEnabled(False)
        work_dir = Path.home() / ".qtmosflm" / "dials"
        self._integrate_thread = QThread(self)
        self._integrate_worker = _DialsIntegrateWorker(
            self._dials_index,
            dials_range,
            work_dir,
            DialsSettings(nproc=4),
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
        self._dials_integration = result
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

    def _cleanup_integrate_thread(self) -> None:
        self._integrate_worker = None
        if self._integrate_thread is not None:
            self._integrate_thread.deleteLater()
            self._integrate_thread = None
        self.find_spots_button.setEnabled(True)
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
                "Unit-cell refinement cannot be run on HDF5 masters in qtMosflm "
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
        if not self._table_solutions:
            return
        row = int(self.table.currentRow())
        if 0 <= row < len(self._table_solutions):
            self._apply_solution_row(row)

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
