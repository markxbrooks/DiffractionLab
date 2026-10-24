"""A "Dimple" tab for the processing windows: refine a known model against the scaled data.

Both processing windows (:class:`~diffraction_lab.processing_wizard.ProcessingWizard`
and :class:`~diffraction_lab.processinglab_window.ProcessingLabWindow`) add
:class:`DimpleTab` after their scaling tab. The window supplies the merged
MTZ it produced and a folder for the runs; the tab runs CCP4 Dimple
(:mod:`diffraction_lab.dimple_runner`) as a background job of the window
and shows the R factors, the unmodelled blobs and their pictures, with
buttons to open the result in Coot.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QFont, QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from diffraction_lab.dimple_runner import (
    MODEL_SUFFIXES,
    DimpleOptions,
    DimpleResult,
    find_dimple,
    new_output_dir,
    open_in_coot,
    run_dimple,
)

if TYPE_CHECKING:
    from diffraction_lab.processing_window_base import ProcessingWindowBase

logger = logging.getLogger(__name__)


def format_dimple(result: DimpleResult) -> str:
    """HTML summary of a Dimple run: outcome, R factors per Refmac step, blobs."""
    def r(v: float | None) -> str:
        return "—" if v is None else f"{v:.4f}"

    def change(start: float | None, end: float | None) -> str:
        return "—" if start is None and end is None else f"{r(start)} → {r(end)}"

    lines = []
    if result.error:
        lines.append(f"<b style='color:#b00'>Dimple stopped:</b> {result.error}")
    elif result.final is not None:
        f = result.final
        lines.append(f"<b>Final R / R<sub>free</sub>:</b> {r(f.r_end)} / {r(f.r_free_end)}")
    if result.molecular_replacement:
        lines.append(f"Molecular replacement with {result.molecular_replacement} (rigid-body R stayed high).")
    if result.refinement:
        rows = "".join(
            f"<tr><td>{s.name}&nbsp;&nbsp;</td><td>{change(s.r_start, s.r_end)}&nbsp;&nbsp;</td>"
            f"<td>{change(s.r_free_start, s.r_free_end)}</td></tr>" for s in result.refinement
        )
        lines.append("<table><tr><th align=left>Refmac&nbsp;</th><th align=left>R&nbsp;</th>"
                     f"<th align=left>R<sub>free</sub></th></tr>{rows}</table>")
    if result.final_pdb is not None:
        n = len(result.blobs)
        lines.append("No unmodelled blobs." if n == 0 else
                     f"{n} unmodelled blob{'s' if n > 1 else ''} in the difference map (below).")
    lines.append(f"Folder: {result.output_dir}")
    return "<br>".join(lines)


class DimpleTab(QWidget):
    """Run CCP4 Dimple on a window's merged data and a model, and show the result.

    :param window: The processing window (runs the job, shows the log).
    :param mtz_provider: The merged MTZ the window produced, or ``None``.
    :param folder_provider: Where runs go (each in a new ``dimple_*``
        folder), or ``None`` to ask.
    :param job_service_provider: A SchedArray
        :class:`~diffraction_lab.jobs.service.JobService` to run Dimple as a
        listed job, or ``None`` to run it directly.
    :param session_id_provider: Recorded on the job (groups a session's steps).
    :param view_mtz: Opens an MTZ in the Reflection Viewer, if the window can.
    :param show_log: Show Dimple's output in the tab (off when the window
        shows job output in its own log).
    """

    def __init__(
        self,
        window: ProcessingWindowBase,
        *,
        mtz_provider: Callable[[], Path | None],
        folder_provider: Callable[[], Path | None],
        job_service_provider: Callable[[], Any] | None = None,
        session_id_provider: Callable[[], str | None] | None = None,
        view_mtz: Callable[[Path], None] | None = None,
        show_log: bool = True,
    ) -> None:
        super().__init__(window)
        self._window = window
        self._mtz_provider = mtz_provider
        self._folder_provider = folder_provider
        self._job_service_provider = job_service_provider
        self._session_id_provider = session_id_provider
        self._view_mtz = view_mtz
        self._running = False
        self.result: DimpleResult | None = None
        self._build_ui(show_log)
        self.update_enabled()

    # ================================================================ UI

    def _build_ui(self, show_log: bool) -> None:
        controls = QWidget()
        v = QVBoxLayout(controls)

        inputs = QGroupBox("Refine a known model against the merged data (CCP4 Dimple)")
        form = QFormLayout(inputs)
        self.mtz_edit = QLineEdit()
        self.mtz_edit.setPlaceholderText("merged MTZ (from the Scale tab)")
        self.use_scaled_btn = QPushButton("Use scaled data")
        self.use_scaled_btn.setToolTip("The merged MTZ from this window's Scale tab")
        self.use_scaled_btn.clicked.connect(self.fill_from_window)
        form.addRow("Merged MTZ", _row(self.mtz_edit, self.use_scaled_btn,
                                       self._browse_button(self.mtz_edit, "Merged MTZ", "MTZ (*.mtz)"),
                                       stretch_first=True))
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("PDB or mmCIF of the same or a similar crystal")
        self.model_edit.textChanged.connect(self.update_enabled)
        self.mtz_edit.textChanged.connect(self.update_enabled)
        model_filter = f"Models ({' '.join('*' + s for s in MODEL_SUFFIXES)})"
        form.addRow("Model", _row(self.model_edit, self._browse_button(self.model_edit, "Model", model_filter),
                                  stretch_first=True))
        self.folder_label = QLabel()
        self.folder_label.setWordWrap(True)
        form.addRow("Output", self.folder_label)
        v.addWidget(inputs)

        opts = QGroupBox("Options")
        oform = QFormLayout(opts)
        self.slow_spin = QSpinBox()
        self.slow_spin.setRange(0, 2)
        self.slow_spin.setToolTip("-s: more refinement cycles (1 or 2)")
        oform.addRow("Extra refinement (-s)", self.slow_spin)
        self.mr_spin = QDoubleSpinBox()
        self.mr_spin.setRange(0.0, 1.0)
        self.mr_spin.setSingleStep(0.05)
        self.mr_spin.setDecimals(2)
        self.mr_spin.setValue(0.40)
        self.mr_spin.setToolTip("-M: molecular replacement (Phaser) when R after rigid-body refinement is above this")
        oform.addRow("Molecular replacement when R >", self.mr_spin)
        self.reso_spin = QDoubleSpinBox()
        self.reso_spin.setRange(0.0, 20.0)
        self.reso_spin.setDecimals(2)
        self.reso_spin.setSpecialValueText("all data")
        self.reso_spin.setSuffix(" Å")
        oform.addRow("High-resolution limit", self.reso_spin)
        self.free_edit = QLineEdit()
        self.free_edit.setPlaceholderText("new flags; '-' = the data MTZ's own; or an MTZ with flags")
        oform.addRow("Free-R flags (-R)", _row(self.free_edit, self._browse_button(self.free_edit, "Free-R flags",
                                                                                  "MTZ (*.mtz)"),
                                               stretch_first=True))
        self.libin_edit = QLineEdit()
        self.libin_edit.setPlaceholderText("optional ligand descriptions (CIF) for Refmac")
        oform.addRow("Ligand library", _row(self.libin_edit, self._browse_button(self.libin_edit, "Ligand CIF",
                                                                                "CIF (*.cif)"),
                                            stretch_first=True))
        self.images_check = QCheckBox("Pictures of the blobs (-f png)")
        self.images_check.setChecked(True)
        self.anode_check = QCheckBox("Anomalous peaks (--anode)")
        self.hetatm_check = QCheckBox("Remove HETATM from the model")
        oform.addRow(_row(self.images_check, self.anode_check, self.hetatm_check))
        self.run_btn = QPushButton("Run Dimple")
        self.run_btn.clicked.connect(self._on_run)
        oform.addRow(self.run_btn)
        v.addWidget(opts)

        result = QGroupBox("Result")
        rv = QVBoxLayout(result)
        self.result_label = QLabel("Not run yet.")
        self.result_label.setWordWrap(True)
        self.result_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        rv.addWidget(self.result_label)
        self.blob_list = QListWidget()
        self.blob_list.setMaximumHeight(90)
        self.blob_list.currentRowChanged.connect(self._show_blob)
        self.blob_list.itemDoubleClicked.connect(lambda *_: self._open_blob())
        rv.addWidget(self.blob_list)
        self.blob_images = QHBoxLayout()
        rv.addLayout(self.blob_images)
        self.coot_btn = QPushButton("Open in Coot")
        self.coot_btn.setToolTip("coot.sh: the refined model, 2mFo-DFc and mFo-DFc maps")
        self.coot_btn.clicked.connect(lambda: self._coot(self.result.coot_script if self.result else None))
        self.coot_blob_btn = QPushButton("Open blob in Coot")
        self.coot_blob_btn.clicked.connect(self._open_blob)
        self.mtz_btn = QPushButton("Open final MTZ")
        self.mtz_btn.clicked.connect(self._on_view_mtz)
        self.folder_btn = QPushButton("Open folder")
        self.folder_btn.clicked.connect(lambda: self._open(self.result.output_dir if self.result else None))
        self.log_btn = QPushButton("Dimple log")
        self.log_btn.clicked.connect(lambda: self._open(self.result.output_dir / "screen.log" if self.result else None))
        rv.addLayout(_row(self.coot_btn, self.coot_blob_btn, self.mtz_btn, self.folder_btn, self.log_btn))
        v.addWidget(result)
        v.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Menlo", 10))
        self.log_view.setPlaceholderText("Dimple's output appears here as it runs.")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if show_log:
            split = QSplitter(Qt.Orientation.Vertical)
            split.addWidget(scroll)
            split.addWidget(self.log_view)
            split.setSizes([480, 160])
            layout.addWidget(split)
        else:
            self.log_view.hide()
            layout.addWidget(scroll)

    def _browse_button(self, edit: QLineEdit, title: str, pattern: str) -> QPushButton:
        button = QPushButton("Browse…")

        def browse() -> None:
            start = edit.text().strip() or str(self._folder_provider() or Path.home())
            path, _ = QFileDialog.getOpenFileName(self, title, start, f"{pattern};;All files (*)")
            if path:
                edit.setText(path)

        button.clicked.connect(browse)
        return button

    # ================================================================ state

    def fill_from_window(self) -> None:
        """Put the window's merged MTZ in the MTZ field."""
        mtz = self._mtz_provider()
        if mtz is not None:
            self.mtz_edit.setText(str(mtz))
        self.update_enabled()

    def showEvent(self, event) -> None:  # noqa: N802
        """Fill the MTZ from the window when the field is still empty."""
        if not self.mtz_edit.text().strip():
            self.fill_from_window()
        self.update_enabled()
        super().showEvent(event)

    def update_enabled(self, *_args: Any) -> None:
        """Enable the controls that apply now."""
        folder = self._folder_provider()
        self.folder_label.setText(f"a new dimple_… folder in {folder}" if folder else
                                  "a new dimple_… folder beside the MTZ")
        self.use_scaled_btn.setEnabled(self._mtz_provider() is not None)
        ready = bool(self.mtz_edit.text().strip() and self.model_edit.text().strip())
        self.run_btn.setEnabled(ready and not self._running and find_dimple() is not None)
        self.run_btn.setToolTip("" if find_dimple() else "CCP4 Dimple not found")
        r = self.result
        self.coot_btn.setEnabled(r is not None and r.coot_script is not None)
        self.coot_blob_btn.setEnabled(r is not None and self._selected_blob() is not None
                                      and self._selected_blob().coot_script is not None)
        self.mtz_btn.setEnabled(self._view_mtz is not None and r is not None and r.final_mtz is not None)
        self.folder_btn.setEnabled(r is not None)
        self.log_btn.setEnabled(r is not None)

    def options(self) -> DimpleOptions:
        """The options set in the tab."""
        libin = self.libin_edit.text().strip()
        return DimpleOptions(
            slow=self.slow_spin.value(),
            mr_when_r=None if abs(self.mr_spin.value() - 0.40) < 1e-9 else self.mr_spin.value(),
            free_r_flags=self.free_edit.text().strip() or None,
            resolution=self.reso_spin.value() or None,
            anode=self.anode_check.isChecked(),
            images=self.images_check.isChecked(),
            no_hetatm=self.hetatm_check.isChecked(),
            libin=Path(libin).expanduser() if libin else None,
        )

    # ================================================================ run

    def _on_run(self) -> None:
        hklin = Path(self.mtz_edit.text().strip()).expanduser()
        xyzin = Path(self.model_edit.text().strip()).expanduser()
        missing = [str(p) for p in (hklin, xyzin) if not p.is_file()]
        if missing:
            QMessageBox.warning(self, "Dimple", "Not found:\n" + "\n".join(missing))
            return
        out = new_output_dir(self._folder_provider() or hklin.parent)
        options = self.options()
        service = self._job_service_provider() if self._job_service_provider else None
        session_id = self._session_id_provider() if self._session_id_provider else None

        def task(emit: Callable[[str], None]) -> DimpleResult | Exception:
            # Failures come back as the result, so the tab can report them.
            try:
                return run_dimple(hklin, xyzin, out, options=options, on_line=emit, job_service=service,
                                  session_id=session_id)
            except Exception as ex:  # noqa: BLE001
                logger.exception("Dimple failed")
                return ex

        self.log_view.clear()
        if not self._window.start_job(f"Dimple: {xyzin.name} against {hklin.name} → {out}", task, self._done,
                                      on_line=self.log_view.appendPlainText):
            return
        self._running = True
        self.result_label.setText(f"Running in {out}…")
        self.update_enabled()

    def _done(self, outcome: DimpleResult | Exception) -> None:
        self._running = False
        if isinstance(outcome, Exception):
            self.result_label.setText(f"<b style='color:#b00'>Dimple failed:</b> {outcome}")
            QMessageBox.warning(self, "Dimple", str(outcome))
            self.update_enabled()
            return
        self.show_result(outcome)
        self._window.report_status("Dimple: " + (outcome.error or (
            f"R/Rfree {outcome.final.r_end:.4f}/{outcome.final.r_free_end:.4f}, {len(outcome.blobs)} blob(s)"
            if outcome.final and outcome.final.r_end is not None and outcome.final.r_free_end is not None
            else "finished")))

    def show_result(self, result: DimpleResult) -> None:
        """Show a run's summary and blobs."""
        self.result = result
        self.result_label.setText(format_dimple(result))
        self.blob_list.clear()
        for k, blob in enumerate(result.blobs, start=1):
            score = "" if blob.score is None else f", score {blob.score:.0f}"
            item = QListWidgetItem(f"Blob {k} at ({blob.x:.1f}, {blob.y:.1f}, {blob.z:.1f}) Å{score}")
            if blob.images:
                item.setIcon(QIcon(str(blob.images[0])))
            self.blob_list.addItem(item)
        if result.blobs:
            self.blob_list.setCurrentRow(0)
        else:
            self._show_blob(-1)
        self.update_enabled()

    def _selected_blob(self):
        row = self.blob_list.currentRow()
        if self.result is None or not 0 <= row < len(self.result.blobs):
            return None
        return self.result.blobs[row]

    def _show_blob(self, _row: int) -> None:
        while self.blob_images.count():
            widget = self.blob_images.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()
        blob = self._selected_blob()
        for image in blob.images if blob else ():
            label = QLabel()
            label.setPixmap(QPixmap(str(image)).scaledToHeight(180, Qt.TransformationMode.SmoothTransformation))
            label.setToolTip(str(image))
            self.blob_images.addWidget(label)
        self.blob_images.addStretch(1)
        self.update_enabled()

    def _open_blob(self) -> None:
        blob = self._selected_blob()
        self._coot(blob.coot_script if blob else None)

    def _coot(self, script: Path | None) -> None:
        if script is None:
            return
        try:
            open_in_coot(script)
        except (OSError, FileNotFoundError) as ex:
            QMessageBox.warning(self, "Coot", str(ex))

    def _on_view_mtz(self) -> None:
        if self.result is not None and self.result.final_mtz is not None and self._view_mtz is not None:
            try:
                self._view_mtz(self.result.final_mtz)
            except Exception as ex:  # noqa: BLE001 - the viewer reports its own failures
                QMessageBox.warning(self, "Reflection File Viewer", str(ex))

    @staticmethod
    def _open(path: Path | None) -> None:
        if path is not None and path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def _row(*widgets: QWidget, stretch_first: bool = False) -> QHBoxLayout:
    row = QHBoxLayout()
    for i, w in enumerate(widgets):
        row.addWidget(w, 1 if stretch_first and i == 0 else 0)
    if not stretch_first:
        row.addStretch(1)
    return row
