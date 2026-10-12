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

from imosflm.indexing import (
    IndexSolution,
    IndexingEngine,
    IndexingGeometry,
    IndexingResult,
    IndexingSettings,
    match_spots_to_solution,
    spots_for_frame,
)
from imosflm.mosflm_client import find_ipmosflm

logger = logging.getLogger(__name__)


class _IndexWorker(QObject):
    """Background worker that runs :meth:`IndexingEngine.index_frames`."""

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
            self.progress.emit("Running Mosflm DPS autoindexing…")
            result = engine.index_frames(
                self._frames,
                self._geometry,
                phis=self._phis,
                frame_indices=self._frame_indices,
                refine=True,
            )
            self.finished.emit(result)
        except Exception as ex:  # noqa: BLE001 - surface to UI
            logger.exception("Indexing failed")
            self.failed.emit(str(ex))


class IndexingDialog(QDialog):
    """UI for Mosflm-style autoindexing of the currently loaded dataset."""

    solutionSelected = Signal(object)  # IndexSolution | None

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("Index — Mosflm DPS")
        self.setMinimumSize(780, 560)
        self.setModal(False)
        self._result: IndexingResult | None = None
        self._table_solutions: list[IndexSolution] = []
        self._thread: QThread | None = None
        self._worker: _IndexWorker | None = None
        self._selected_frames: list[int] = []
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

        self.run_button = QPushButton("Run indexing")
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

        exe = find_ipmosflm()
        if exe is None:
            self.status_label.setText(
                "ipmosflm not found — install CCP4 or set MOSFLM_EXEC."
            )
            self.run_button.setEnabled(False)
        else:
            self.status_label.setText(f"Using {exe}")

        app = self.parentWidget()
        if app is not None and hasattr(app, "show_indexing_spots"):
            self.show_spots_checkbox.blockSignals(True)
            self.show_spots_checkbox.setChecked(bool(app.show_indexing_spots))
            self.show_spots_checkbox.blockSignals(False)

    def _settings(self) -> IndexingSettings:
        return IndexingSettings(
            spot_threshold_sigma=float(self.spot_sigma.value()),
            i_sig_i=float(self.i_sig_i.value()),
            max_penalty_to_refine=float(self.max_penalty.value()),
        )

    def _compute_frame_indices(self) -> list[int]:
        """Return the frame index list from the Start/End/Step spin boxes."""
        start = int(self.frame_start.value())
        end = int(self.frame_end.value())
        step = int(self.frame_step.value())
        if end < start:
            start, end = end, start
        return list(range(start, end + 1, step))

    def _on_run(self) -> None:
        app = self.parent()
        if app is None or getattr(app, "dataset", None) is None:
            QMessageBox.warning(
                self,
                "No dataset",
                "Open an HDF5 / NeXus master (or image stack) before indexing.",
            )
            return
        try:
            geometry = app.build_indexing_geometry()
            indices = self._compute_frame_indices()
            if not indices:
                raise ValueError("No frames selected")
            frames, phis = app.load_frames_for_indexing(indices)
        except Exception as ex:
            QMessageBox.critical(self, "Cannot prepare indexing", str(ex))
            return
        self.run_button.setEnabled(False)
        self.table.setRowCount(0)
        self._table_solutions = []
        self._selected_frames = list(indices)
        self.log.clear()
        self.status_label.setText(
            f"Indexing {len(frames)} frame(s): {indices[0]}…{indices[-1]}"
        )
        self.log.append(f"Frames: {indices}")
        self.log.append(
            f"Beam {geometry.beam_x_mm:.3f}, {geometry.beam_y_mm:.3f} mm; "
            f"dist {geometry.distance_mm:.2f} mm; λ {geometry.wavelength_a:.5f} Å"
        )
        # Clear any previous overlay while a new run starts.
        self.solutionSelected.emit(None)

        work_dir = Path.home() / ".qtmosflm" / "work"
        self._thread = QThread(self)
        self._worker = _IndexWorker(
            frames, geometry, self._settings(), phis, indices, work_dir
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

    def _cleanup_thread(self) -> None:
        self._worker = None
        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None
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
