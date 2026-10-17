"""MxPanDDA-style Jobs history tab backed by SchedArray."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QBrush
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from diffraction_lab.jobs.metadata import metadata_log_file, metadata_tool
from diffraction_lab.jobs.service import JobService, get_job_service, use_schedarray

logger = logging.getLogger(__name__)

_COLUMNS = (
    "Job ID",
    "Name",
    "Tool",
    "State",
    "Submitted",
    "Started",
    "Completed",
    "RC",
    "Working Dir",
    "Command",
    "Log",
)

# Bootstrap-inspired status colors (mxflask queue_viewer getStatusColor)
_STATE_COLORS = {
    "pending": ("#6c757d", "#f8f9fa"),
    "running": ("#ffffff", "#0d6efd"),
    "completed": ("#ffffff", "#198754"),
    "failed": ("#ffffff", "#dc3545"),
    "cancelled": ("#ffffff", "#6c757d"),
    "timeout": ("#000000", "#ffc107"),
    "error": ("#ffffff", "#dc3545"),
}


class JobsTab(QWidget):
    """Table of SchedArray jobs with refresh, filter, cancel, and open-log."""

    open_log_requested = Signal(object)  # Path
    jobs_changed = Signal()

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        service: JobService | None = None,
        work_dir_provider: Callable[[], Path | None] | None = None,
        prepare_service: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._work_dir_provider = work_dir_provider
        self._prepare_service = prepare_service
        self._tool_filter: str | None = None
        self._jobs: list[dict] = []

        self._auto_timer = QTimer(self)
        self._auto_timer.setInterval(2_000)
        self._auto_timer.timeout.connect(self.refresh)

        self._build_ui()
        # Default on so in-flight SchedArray jobs (e.g. dials.integrate) appear
        # without requiring a manual Refresh.
        self.auto_refresh.setChecked(True)
        self.refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        bar = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh)
        bar.addWidget(self.refresh_btn)

        self.auto_refresh = QCheckBox("Auto-refresh")
        self.auto_refresh.setToolTip(
            "Poll the per-master SchedArray DB (2s while jobs run, else 10s)."
        )
        self.auto_refresh.toggled.connect(self._toggle_auto_refresh)
        bar.addWidget(self.auto_refresh)

        self.work_dir_only = QCheckBox("This work dir only")
        self.work_dir_only.setChecked(False)
        self.work_dir_only.setToolTip(
            "When checked, only show jobs for the current Processing work directory."
        )
        self.work_dir_only.toggled.connect(lambda _checked: self.refresh())
        bar.addWidget(self.work_dir_only)

        bar.addWidget(QLabel("Filter:"))
        self._filter_buttons: dict[str | None, QPushButton] = {}
        for label, key in (
            ("All", None),
            ("DIALS", "dials"),
            ("Pointless", "pointless"),
            ("MOSFLM", "mosflm"),
        ):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(key is None)
            btn.clicked.connect(lambda checked=False, k=key: self._set_tool_filter(k))
            bar.addWidget(btn)
            self._filter_buttons[key] = btn

        bar.addStretch(1)

        self.view_log_btn = QPushButton("View Log")
        self.view_log_btn.setEnabled(False)
        self.view_log_btn.clicked.connect(self._on_view_log)
        bar.addWidget(self.view_log_btn)

        self.cancel_btn = QPushButton("Cancel Selected Job")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel)
        bar.addWidget(self.cancel_btn)

        layout.addLayout(bar)

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(list(_COLUMNS))
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.doubleClicked.connect(self._on_view_log)
        layout.addWidget(self.table)

    def _service_or_none(self) -> JobService | None:
        if self._service is not None:
            return self._service
        if not use_schedarray():
            return None
        try:
            return get_job_service()
        except Exception as ex:  # noqa: BLE001
            logger.warning("JobService unavailable: %s", ex)
            return None

    def _toggle_auto_refresh(self, enabled: bool) -> None:
        if enabled:
            self._auto_timer.start()
        else:
            self._auto_timer.stop()

    def _set_tool_filter(self, tool: str | None) -> None:
        self._tool_filter = tool
        for key, btn in self._filter_buttons.items():
            btn.setChecked(key == tool)
        self.refresh()

    def _adjust_refresh_interval(self, running: int) -> None:
        """Poll faster while any job is running."""
        self._auto_timer.setInterval(2_000 if running else 10_000)

    def refresh(self) -> None:
        """Reload the jobs table from SchedArray."""
        if self._prepare_service is not None:
            try:
                self._prepare_service()
            except Exception:  # noqa: BLE001
                logger.debug("prepare_service failed", exc_info=True)
        service = self._service_or_none()
        if service is None:
            self._jobs = []
            self.table.setRowCount(0)
            if not use_schedarray():
                self.status_label.setText(
                    "SchedArray not active — DIALS/Pointless steps run locally "
                    "and are not recorded. Install schedarray into this Python "
                    "(pip install -e ../SchedArray) and restart DiffractionLab."
                )
            else:
                self.status_label.setText("Job service unavailable.")
            return

        work_dir = None
        if self.work_dir_only.isChecked() and self._work_dir_provider is not None:
            try:
                work_dir = self._work_dir_provider()
            except Exception:  # noqa: BLE001
                work_dir = None

        try:
            jobs = service.list_jobs(
                tool=self._tool_filter,
                work_dir=work_dir,
                limit=300,
                app_only=True,
            )
        except Exception as ex:  # noqa: BLE001
            logger.exception("Failed to list jobs")
            self.status_label.setText(f"Error: {ex}")
            return

        # Newest first for the UI
        jobs = list(reversed(jobs))
        self._jobs = jobs
        self.table.setRowCount(len(jobs))
        running = 0
        for row, job in enumerate(jobs):
            state = str(job.get("state", "")).lower()
            if state == "running":
                running += 1
            meta_tool = metadata_tool(job) or ""
            log_path = metadata_log_file(job)
            values = [
                str(job.get("job_id", ""))[:8],
                str(job.get("job_name", "")),
                meta_tool,
                state,
                _fmt_ts(job.get("submitted_at")),
                _fmt_ts(job.get("started_at")),
                _fmt_ts(job.get("completed_at")),
                "" if job.get("return_code") is None else str(job.get("return_code")),
                str(job.get("working_dir") or ""),
                _short(str(job.get("command") or ""), 80),
                str(log_path) if log_path else "",
            ]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, job.get("job_id"))
                if col == 3:
                    fg, bg = _STATE_COLORS.get(state, ("#000000", "#ffffff"))
                    item.setForeground(QBrush(QColor(fg)))
                    item.setBackground(QBrush(QColor(bg)))
                self.table.setItem(row, col, item)

        self.table.resizeColumnsToContents()
        self._adjust_refresh_interval(running)
        db_hint = ""
        try:
            db = getattr(service, "db_path", None)
            if db is not None:
                db_hint = f" | DB: {db}"
        except Exception:  # noqa: BLE001
            pass
        self.status_label.setText(
            f"Total jobs: {len(jobs)} | Running: {running}{db_hint}"
        )
        self.jobs_changed.emit()

    def _selected_job(self) -> Optional[dict]:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        row = rows[0].row()
        if 0 <= row < len(self._jobs):
            return self._jobs[row]
        return None

    def _on_selection_changed(self) -> None:
        job = self._selected_job()
        has = job is not None
        self.cancel_btn.setEnabled(has)
        log_ok = bool(job and metadata_log_file(job))
        self.view_log_btn.setEnabled(log_ok)

    def _on_cancel(self) -> None:
        job = self._selected_job()
        if not job:
            return
        job_id = str(job.get("job_id", ""))
        reply = QMessageBox.question(
            self,
            "Cancel Job",
            f"Cancel job {job_id[:8]} ({job.get('job_name')})?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        service = self._service_or_none()
        if service is None:
            return
        ok = service.cancel(job_id)
        if not ok:
            QMessageBox.warning(self, "Cancel", f"Could not cancel job {job_id[:8]}")
        self.refresh()

    def _on_view_log(self) -> None:
        job = self._selected_job()
        if not job:
            return
        path = metadata_log_file(job)
        if path is None:
            QMessageBox.information(self, "Log", "No log file recorded for this job.")
            return
        self.open_log_requested.emit(path)


def _fmt_ts(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    # ISO timestamps → show date/time without microseconds
    if "T" in text:
        return text.replace("T", " ")[:19]
    return text[:19]


def _short(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    return text[: n - 1] + "…"
