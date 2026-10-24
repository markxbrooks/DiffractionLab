"""ProcessingLab tab: run, check symmetry and scale, re-run from refined values.

Workflow on one sweep:

1. **Run** ``ProcessingLab run`` on the open image (find spots, autoindex,
   integrate) into a new run directory.
2. **Symmetry & scaling**: Pointless then Aimless on the run's unmerged
   ``process.mtz``, into the run's ``ccp4/`` folder.
3. **Re-run** with the refined unit cell and detector parameters
   (``ProcessingLab run --from``), choosing what is refined again and,
   optionally, the space group Pointless found.

Each step runs in a background thread; ProcessingLab's output streams into
the log.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from diffraction_lab.aimless_runner import ScaleMergeResult, find_aimless, scale_and_merge
from diffraction_lab.pointless_runner import find_pointless, is_identity_reindex
from diffraction_lab.processinglab_client import find_integrationlab
from diffraction_lab.processinglab_run import (
    REFINABLE_PARAMETERS,
    RunOptions,
    RunRecord,
    latest_run_dir,
    load_run_record,
    rerun_options,
    run_processinglab,
)
from diffraction_lab.processing_sweep import Frames, ProcessingSweep

logger = logging.getLogger(__name__)

#: ``--refine`` default for a re-run: refine the orientation, re-estimate the
#: mosaicity, hold the refined cell and detector parameters.
DEFAULT_RERUN_REFINE: tuple[str, ...] = ("orientation", "mosaicity")

_REFINE_LABELS = {
    "orientation": "Orientation",
    "cell": "Unit cell",
    "distance": "Detector distance",
    "beam": "Beam centre",
    "tilt_twist": "Detector tilt / twist",
    "mosaicity": "Mosaicity (re-estimate)",
}


def parse_cell_text(text: str) -> tuple[float, float, float, float, float, float] | None:
    """Parse an optional ``a b c α β γ`` (spaces or commas) cell entry.

    :param text: The entry; blank means none.
    :returns: Six values, or ``None`` when blank.
    :raises ValueError: If it is not six positive numbers.
    """
    tokens = text.replace(",", " ").split()
    if not tokens:
        return None
    try:
        values = tuple(float(t) for t in tokens)
    except ValueError as ex:
        raise ValueError(f"cell must be six numbers, got {text!r}") from ex
    if len(values) != 6 or any(v <= 0 for v in values):
        raise ValueError(f"cell must be six positive numbers, got {text!r}")
    return values  # type: ignore[return-value]


def default_work_parent(image: Path) -> Path:
    """Where runs for *image* go: ``<image_dir>/<stem>-processinglab/``, or
    ``~/.qtmosflm/processinglab/<stem>/`` when the image folder is read-only.

    :param image: The open image (any frame, or an HDF5 master).
    :returns: The folder (created).
    """
    stem = image.stem.removesuffix("_master")
    beside = image.parent / f"{stem}-processinglab"
    try:
        beside.mkdir(parents=True, exist_ok=True)
        probe = beside / ".write-test"
        probe.touch()
        probe.unlink()
        return beside
    except OSError:
        home = Path.home() / ".qtmosflm" / "processinglab" / stem
        home.mkdir(parents=True, exist_ok=True)
        return home


def format_record(record: RunRecord) -> str:
    """HTML summary of a run for the Result box."""

    def num(v: float | None, fmt: str) -> str:
        return "—" if v is None else format(v, fmt)

    rows = [("Status", record.status + (f" — {record.error}" if record.error else ""))]
    if record.space_group_number is not None:
        symbol = f" ({record.space_group_symbol})" if record.space_group_symbol else ""
        rows.append(("Space group", f"{record.space_group_number}{symbol}"))
    if record.cell is not None:
        a, b, c, al, be, ga = record.cell
        rows.append(("Cell", f"{a:.3f} {b:.3f} {c:.3f} Å, {al:.2f} {be:.2f} {ga:.2f}°"))
    if record.distance_mm is not None:
        beam = "—" if record.beam_mm is None else f"{record.beam_mm[0]:.3f} {record.beam_mm[1]:.3f}"
        rows.append(("Detector", f"distance {record.distance_mm:.3f} mm, beam {beam} mm"))
    if record.tilt_twist_mrad is not None:
        rows.append(("Tilt / twist", f"{record.tilt_twist_mrad[0]:.2f} / {record.tilt_twist_mrad[1]:.2f} mrad"))
    if record.mosaicity_deg is not None:
        rows.append(("Mosaicity", f"{record.mosaicity_deg:.3f}°"))
    if record.refinement:
        refined = [k for k, v in record.refinement.items() if v != "fixed"]
        fixed = [k for k, v in record.refinement.items() if v == "fixed"]
        rows.append(("Refined", ", ".join(refined) or "nothing"))
        rows.append(("Held fixed", ", ".join(fixed) or "nothing"))
    if record.integrated_reflections is not None:
        rows.append(("Integrated", f"{record.integrated_reflections} reflections"))
    q = record.quality_total
    if q:
        rows.append((
            "Unscaled",
            f"R<sub>meas</sub> {num(q.get('r_meas_pct'), '.1f')}%, I/σ {num(q.get('mean_i_over_sigma'), '.2f')}, "
            f"CC½ {num(q.get('cc_half_pct'), '.1f')}%, completeness {num(q.get('completeness_pct'), '.1f')}%",
        ))
    rows.append(("Folder", str(record.directory)))
    return "<table>" + "".join(f"<tr><td><b>{k}</b>&nbsp;</td><td>{v}</td></tr>" for k, v in rows) + "</table>"


def format_scaling(result: ScaleMergeResult) -> str:
    """HTML summary of Pointless + Aimless for the scaling box."""
    p, s = result.pointless, result.aimless.summary
    kind = p.solution_kind or "space group"
    lines = [
        f"<b>Pointless:</b> {kind} {p.space_group or '—'}, reindex {p.reindex_operator or '—'}, "
        f"space-group confidence {'—' if p.confidence is None else f'{p.confidence:.2f}'}"
        + ("" if p.laue_confidence is None else f", Laue-group confidence {p.laue_confidence:.2f}"),
    ]
    if kind == "point group":
        lines.append("<i>Only the point group was determined (too few axial reflections for screw axes).</i>")

    def cell(label: str, shell: str, fmt: str) -> str:
        v = s.value(label, shell)
        return "—" if v is None else format(v, fmt)

    table = [("Resolution (Å)", "High resolution limit", ".2f"),
             ("R<sub>meas</sub>", "Rmeas (all I+ & I-)", ".3f"),
             ("CC½", "Mn(I) half-set correlation CC(1/2)", ".3f"),
             ("⟨I/σ⟩", "Mean((I)/sd(I))", ".1f"),
             ("Completeness (%)", "Completeness", ".1f"),
             ("Multiplicity", "Multiplicity", ".1f")]
    rows = "".join(
        f"<tr><td>{name}</td><td align=right>{cell(label, 'overall', f)}</td>"
        f"<td align=right>{cell(label, 'inner', f)}</td><td align=right>{cell(label, 'outer', f)}</td></tr>"
        for name, label, f in table
    )
    lines.append(f"<b>Aimless</b> ({s.space_group or '—'}):<table><tr><th></th><th>Overall</th>"
                 f"<th>Inner</th><th>Outer</th></tr>{rows}</table>")
    if (cc := s.resolution_estimates.get("half-dataset correlation CC(1/2) >  0.30")) is not None:
        lines.append(f"Resolution estimate (CC½ &gt; 0.3): {cc:.2f} Å")
    return "<br>".join(lines)


class CallWorker(QObject):
    """Run one callable in a thread; ``line`` relays its log output."""

    finished = Signal(object)
    failed = Signal(str)
    line = Signal(str)

    def __init__(self, fn: Callable[[Callable[[str], None]], Any]) -> None:
        super().__init__()
        self._fn = fn

    def run(self) -> None:
        try:
            self.finished.emit(self._fn(self.line.emit))
        except Exception as ex:  # noqa: BLE001 - surfaced in the panel
            logger.exception("ProcessingLab panel task failed")
            self.failed.emit(str(ex))


class ProcessingLabPanel(QWidget):
    """The ProcessingLab tab of the Processing window."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image: Path | None = None
        self._record: RunRecord | None = None
        self._scaling: ScaleMergeResult | None = None
        self._thread: QThread | None = None
        self._worker: CallWorker | None = None
        self._build_ui()
        self._update_enabled()

    # ----- UI -----

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        run_box = QGroupBox("Run ProcessingLab (find spots → index → integrate)")
        form = QFormLayout(run_box)
        self.image_label = QLabel("(no image loaded)")
        self.image_label.setWordWrap(True)
        form.addRow("Image", self.image_label)
        self.first_frame = QSpinBox()
        self.last_frame = QSpinBox()
        for spin in (self.first_frame, self.last_frame):
            spin.setRange(1, 999999)
        frames = QHBoxLayout()
        frames.addWidget(self.first_frame)
        frames.addWidget(QLabel("to"))
        frames.addWidget(self.last_frame)
        frames.addStretch(1)
        form.addRow("Frames", frames)
        self.work_edit = QLineEdit()
        self.work_edit.setToolTip("Each run creates a new processing_YYYYMMDD_HHMMSS folder here")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._on_browse_work)
        work_row = QHBoxLayout()
        work_row.addWidget(self.work_edit, stretch=1)
        work_row.addWidget(browse)
        form.addRow("Work folder", work_row)
        self.space_group_edit = QLineEdit()
        self.space_group_edit.setPlaceholderText("optional, e.g. 92 or P41212 (default: from indexing)")
        form.addRow("Space group", self.space_group_edit)
        self.cell_edit = QLineEdit()
        self.cell_edit.setPlaceholderText("optional a b c α β γ")
        form.addRow("Unit cell", self.cell_edit)
        buttons = QHBoxLayout()
        self.run_btn = QPushButton("Run ProcessingLab")
        self.run_btn.clicked.connect(self._on_run)
        self.open_btn = QPushButton("Open run…")
        self.open_btn.setToolTip("Load an existing ProcessingLab run folder")
        self.open_btn.clicked.connect(self._on_open_run)
        buttons.addWidget(self.run_btn)
        buttons.addWidget(self.open_btn)
        buttons.addStretch(1)
        form.addRow(buttons)
        layout.addWidget(run_box)

        result_box = QGroupBox("Result")
        result_layout = QVBoxLayout(result_box)
        self.result_label = QLabel("No run yet.")
        self.result_label.setWordWrap(True)
        self.result_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        result_layout.addWidget(self.result_label)
        result_buttons = QHBoxLayout()
        self.reveal_btn = QPushButton("Reveal folder")
        self.reveal_btn.clicked.connect(lambda: self._open(self._record.directory if self._record else None))
        self.report_btn = QPushButton("Open report")
        self.report_btn.clicked.connect(
            lambda: self._open(self._record.directory / "process.md" if self._record else None))
        result_buttons.addWidget(self.reveal_btn)
        result_buttons.addWidget(self.report_btn)
        result_buttons.addStretch(1)
        result_layout.addLayout(result_buttons)
        layout.addWidget(result_box)

        scale_box = QGroupBox("Symmetry and scaling (CCP4 Pointless → Aimless)")
        scale_layout = QVBoxLayout(scale_box)
        self.scale_btn = QPushButton("Run Pointless → Aimless")
        self.scale_btn.clicked.connect(self._on_scale)
        self.scale_label = QLabel("")
        self.scale_label.setWordWrap(True)
        scale_layout.addWidget(self.scale_btn)
        scale_layout.addWidget(self.scale_label)
        layout.addWidget(scale_box)

        rerun_box = QGroupBox("Re-run with refined unit cell and detector parameters")
        rerun_layout = QVBoxLayout(rerun_box)
        rerun_layout.addWidget(QLabel(
            "Starts from this run's refined cell, space group, distance, beam, tilt/twist and "
            "orientation (no autoindexing). Ticked parameters are refined again; the rest stay fixed."))
        checks = QHBoxLayout()
        self.refine_checks: dict[str, QCheckBox] = {}
        for name in REFINABLE_PARAMETERS:
            box = QCheckBox(_REFINE_LABELS[name])
            box.setChecked(name in DEFAULT_RERUN_REFINE)
            self.refine_checks[name] = box
            checks.addWidget(box)
        checks.addStretch(1)
        rerun_layout.addLayout(checks)
        rerun_form = QFormLayout()
        self.rerun_space_group_edit = QLineEdit()
        self.rerun_space_group_edit.setPlaceholderText("optional; filled from Pointless when it is safe")
        rerun_form.addRow("Space group", self.rerun_space_group_edit)
        self.rerun_cell_edit = QLineEdit()
        self.rerun_cell_edit.setPlaceholderText("optional a b c α β γ (default: the refined cell)")
        rerun_form.addRow("Unit cell", self.rerun_cell_edit)
        rerun_layout.addLayout(rerun_form)
        self.rerun_btn = QPushButton("Re-run with refined parameters")
        self.rerun_btn.clicked.connect(self._on_rerun)
        rerun_layout.addWidget(self.rerun_btn)
        layout.addWidget(rerun_box)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(20000)
        layout.addWidget(self.log_view, stretch=1)

    # ----- dataset -----

    def set_dataset(self, image: Path | None, n_frames: int) -> None:
        """Point the panel at the open image (any frame, or an HDF5 master).

        :param image: The image, or ``None``.
        :param n_frames: Frames in the sweep (0 if unknown).
        """
        self._image = Path(image).expanduser().resolve() if image else None
        self.image_label.setText(str(self._image) if self._image else "(no image loaded)")
        # The whole sweep as ProcessingLab sees it (the viewer's frame count
        # is per HDF5 data file); n_frames is the fallback.
        sweep = ProcessingSweep.from_image(self._image) if self._image else None
        frames = sweep.frames if sweep else Frames(1, max(int(n_frames), 1))
        for spin in (self.first_frame, self.last_frame):
            spin.setRange(frames.start, frames.end)
        self.first_frame.setValue(frames.start)
        self.last_frame.setValue(frames.end)
        if self._image is not None and not self.work_edit.text().strip():
            parent = default_work_parent(self._image)
            self.work_edit.setText(str(parent))
            if self._record is None and (latest := latest_run_dir(parent)) is not None:
                self._load(latest)
        self._update_enabled()

    # ----- actions -----

    def _on_browse_work(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Work folder", self.work_edit.text())
        if folder:
            self.work_edit.setText(folder)

    def _on_open_run(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "ProcessingLab run folder", self.work_edit.text())
        if folder:
            self._load(Path(folder))

    def _load(self, run_dir: Path) -> None:
        try:
            self._show_record(load_run_record(run_dir))
        except (FileNotFoundError, ValueError) as ex:
            QMessageBox.warning(self, "ProcessingLab", f"Not a ProcessingLab run:\n{ex}")

    def _on_run(self) -> None:
        if self._image is None:
            QMessageBox.warning(self, "ProcessingLab", "Open an image first.")
            return
        try:
            options = RunOptions(
                image=self._image,
                frames=self._frames(),
                parent_dir=Path(self.work_edit.text().strip() or default_work_parent(self._image)),
                space_group=self.space_group_edit.text().strip() or None,
                cell=parse_cell_text(self.cell_edit.text()),
            )
        except ValueError as ex:
            QMessageBox.warning(self, "ProcessingLab", str(ex))
            return
        self._start("Running ProcessingLab…",
                    lambda emit: run_processinglab(options, on_line=emit), self._on_run_finished)

    def _frames(self) -> Frames:
        """The ``--range``: always explicit, so what is shown is what is processed."""
        return Frames.between(self.first_frame.value(), self.last_frame.value())

    def _on_rerun(self) -> None:
        if self._record is None:
            return
        refine = tuple(name for name, box in self.refine_checks.items() if box.isChecked())
        try:
            options = rerun_options(
                self._record,
                refine=refine,
                space_group=self.rerun_space_group_edit.text().strip() or None,
                cell=parse_cell_text(self.rerun_cell_edit.text()),
            )
        except ValueError as ex:
            QMessageBox.warning(self, "ProcessingLab", str(ex))
            return
        self._start(f"Re-running from {self._record.directory.name} (refining {', '.join(refine) or 'nothing'})…",
                    lambda emit: run_processinglab(options, on_line=emit), self._on_run_finished)

    def _on_scale(self) -> None:
        record = self._record
        if record is None or record.mtz is None:
            return
        work = record.directory / "ccp4"

        def task(emit: Callable[[str], None]) -> ScaleMergeResult:
            emit(f"Pointless → Aimless on {record.mtz} into {work}")
            return scale_and_merge(record.mtz, work)

        self._start("Running Pointless and Aimless…", task, self._on_scale_finished)

    # ----- results -----

    def _on_run_finished(self, record: RunRecord) -> None:
        self._show_record(record)
        self.status_label.setText(f"Run completed: {record.directory}")

    def _show_record(self, record: RunRecord) -> None:
        self._record = record
        self._scaling = None
        self.result_label.setText(format_record(record))
        self.scale_label.setText("")
        self.rerun_space_group_edit.clear()
        self.rerun_cell_edit.clear()
        self._update_enabled()

    def _on_scale_finished(self, result: ScaleMergeResult) -> None:
        self._scaling = result
        self.scale_label.setText(format_scaling(result))
        p = result.pointless
        # ProcessingLab's --spacegroup does not reindex: offer Pointless's
        # space group only when it keeps the axes (and is a space group).
        if p.solution_kind != "point group" and p.space_group and is_identity_reindex(p.reindex_operator):
            self.rerun_space_group_edit.setText(p.space_group)
            self.status_label.setText(f"Pointless: {p.space_group}; filled in for the re-run.")
        elif not is_identity_reindex(p.reindex_operator):
            self.status_label.setText(
                f"Pointless reindexes ({p.reindex_operator}); its space group is not applied to the re-run.")
        else:
            self.status_label.setText("Pointless found only the point group; set the space group yourself.")
        self._update_enabled()

    # ----- plumbing -----

    def _start(self, message: str, fn: Callable[[Callable[[str], None]], Any],
               on_finished: Callable[[Any], None]) -> None:
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(self, "ProcessingLab", "A job is already running.")
            return
        self.status_label.setText(message)
        self.log_view.appendPlainText(f"\n=== {datetime.now():%H:%M:%S} {message}")
        self._thread = QThread(self)
        self._worker = CallWorker(fn)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.line.connect(self.log_view.appendPlainText)
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

    def _update_enabled(self) -> None:
        busy = self._busy()
        have_pl = find_integrationlab() is not None
        record = self._record
        self.run_btn.setEnabled(not busy and have_pl and self._image is not None)
        self.run_btn.setToolTip("" if have_pl else "ProcessingLab not found (make install, or set $PROCESSINGLAB)")
        self.open_btn.setEnabled(not busy)
        self.reveal_btn.setEnabled(record is not None)
        self.report_btn.setEnabled(record is not None and (record.directory / "process.md").is_file())
        have_ccp4 = find_pointless() is not None and find_aimless() is not None
        self.scale_btn.setEnabled(not busy and have_ccp4 and record is not None and record.mtz is not None)
        self.scale_btn.setToolTip("" if have_ccp4 else "CCP4 Pointless / Aimless not found")
        self.rerun_btn.setEnabled(not busy and have_pl and record is not None and record.can_rerun)

    def _open(self, path: Path | None) -> None:
        if path is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
