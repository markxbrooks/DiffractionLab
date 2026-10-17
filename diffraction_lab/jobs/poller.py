"""QObject that polls a SchedArray job until it finishes."""

from __future__ import annotations

import time
from typing import Any, Optional

from PySide6.QtCore import QObject, Signal

from diffraction_lab.jobs.service import JobService, get_job_service


class JobPollWorker(QObject):
    """Poll one SchedArray job id on a worker thread.

    Emits :attr:`finished` with the final status dict, or :attr:`failed` with
    an error message.
    """

    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        job_id: str,
        *,
        service: JobService | None = None,
        poll_interval: float = 0.5,
        timeout: float | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._job_id = job_id
        self._service = service
        self._poll_interval = poll_interval
        self._timeout = timeout

    def run(self) -> None:
        """Block until the job reaches a terminal state."""
        try:
            service = self._service or get_job_service()
            self.progress.emit(f"Waiting for job {self._job_id[:8]}…")
            status = service.wait_for_job(
                self._job_id,
                poll_interval=self._poll_interval,
                timeout=self._timeout,
            )
            state = str(status.get("state", "")).lower()
            if state == "completed":
                self.finished.emit(status)
                return
            rc = status.get("return_code")
            self.failed.emit(
                f"Job {self._job_id} ended with state={state!r} "
                f"(return_code={rc})"
            )
        except Exception as ex:  # noqa: BLE001
            self.failed.emit(str(ex))


def wait_status_ok(status: dict[str, Any]) -> None:
    """Raise :class:`RuntimeError` unless *status* is ``completed``."""
    state = str(status.get("state", "")).lower()
    if state != "completed":
        rc = status.get("return_code")
        raise RuntimeError(
            f"SchedArray job failed (state={state!r}, return_code={rc})"
        )
