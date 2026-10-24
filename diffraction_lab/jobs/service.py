"""SchedArray-backed job service for DiffractionLab.

Owns a :class:`~schedarray.SqliteJobScheduler` instance (not the package
import-time singleton) and exposes submit / list / cancel / wait helpers.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from diffraction_lab.jobs.metadata import build_job_metadata

logger = logging.getLogger(__name__)

_TERMINAL_STATES = frozenset(
    {
        "completed",
        "failed",
        "cancelled",
        "timeout",
        "error",
        "skipped",
        "disabled",
    }
)

#: States after which a job will not change again.
TERMINAL_STATES = _TERMINAL_STATES

_service: Optional["JobService"] = None
_service_db_path: Optional[Path] = None

# Per-master SchedArray DB filename under ``<master_stem>/``
SCHEDARRAY_DB_NAME = "schedarray_scheduler.db"

# Index names created by SchedArray SqliteJobScheduler._create_indexes
_SCHEDARRAY_ADHOC_INDEX_NAMES = frozenset(
    {
        "idx_job_queue_state",
        "idx_job_queue_priority_submitted",
        "idx_job_queue_user",
        "idx_worker_nodes_state",
        "idx_resource_usage_job_id",
    }
)


def schedarray_db_path_for_master(master_path: Path | str) -> Path:
    """Return ``<master_dir>/<master_stem>/schedarray_scheduler.db``.

    Example: ``…/thaumatin_9_1_master.h5`` →
    ``…/thaumatin_9_1_master/schedarray_scheduler.db``.
    """
    from diffraction_lab.dials_engine import dials_run_root_for_master

    root = dials_run_root_for_master(master_path)
    root.mkdir(parents=True, exist_ok=True)
    return root / SCHEDARRAY_DB_NAME


def schedarray_db_path_for_work_dir(work_dir: Path | str) -> Path:
    """Return the SchedArray DB for a DIALS run directory.

    Run dirs live at ``<stem_root>/<stamp>-dials/``, so the DB is
    ``<stem_root>/schedarray_scheduler.db``.
    """
    work = Path(work_dir).expanduser().resolve()
    stem_root = work.parent
    stem_root.mkdir(parents=True, exist_ok=True)
    return stem_root / SCHEDARRAY_DB_NAME


def _stop_service_workers(service: "JobService") -> None:
    """Best-effort stop of a JobService worker pool before switching DBs."""
    try:
        pool = getattr(service.scheduler, "_worker_pool", None)
        if pool is not None and getattr(pool, "running", False):
            pool.stop_workers()
    except Exception:  # noqa: BLE001
        logger.debug("Could not stop SchedArray workers", exc_info=True)


def _clear_schedarray_adhoc_indexes(base) -> None:
    """Remove SchedArray ad-hoc Index objects from SQLAlchemy metadata."""
    try:
        for table in list(base.metadata.tables.values()):
            for index in list(table.indexes):
                if index.name in _SCHEDARRAY_ADHOC_INDEX_NAMES:
                    table.indexes.discard(index)
    except Exception:  # noqa: BLE001
        logger.debug("Could not clear SchedArray ad-hoc indexes", exc_info=True)


def schedarray_available() -> bool:
    """Return ``True`` if the ``schedarray`` package can be imported.

    Uses :func:`importlib.util.find_spec` so importing DiffractionLab does not
    construct SchedArray's import-time global scheduler singleton.
    """
    import importlib.util

    return importlib.util.find_spec("schedarray") is not None


def use_schedarray() -> bool:
    """Whether DiffractionLab should submit jobs via SchedArray.

    Controlled by ``USE_SCHEDARRAY`` (default ``true`` when SchedArray is
    installed). Set to ``0`` / ``false`` / ``no`` to force the in-process
    ``Popen`` path in :class:`~diffraction_lab.dials_engine.DialsEngine`.
    """
    env = os.environ.get("USE_SCHEDARRAY", "true").strip().lower()
    if env in ("0", "false", "no", "off"):
        return False
    return schedarray_available()


class JobService:
    """Thin wrapper around :class:`~schedarray.SqliteJobScheduler`."""

    def __init__(
        self,
        db_path: Path | str | None = None,
        *,
        auto_start_workers: bool = True,
        max_workers: int | None = None,
    ) -> None:
        """Create a scheduler backed by *db_path* (or SchedArray default).

        :param db_path: SQLite database path. ``None`` uses SchedArray defaults.
        :param auto_start_workers: Start the local worker pool on first submit.
        :param max_workers: Optional worker pool size override.
        """
        if not schedarray_available():
            raise ImportError(
                "schedarray is not installed. "
                "Install with: pip install -e ../SchedArray"
            )
        from schedarray.core.job.scheduler import SqliteJobScheduler
        from schedarray.core.sql.base import Base

        path = Path(db_path).expanduser().resolve() if db_path is not None else None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = path
        # SchedArray's SqliteJobScheduler._create_indexes() attaches Index()
        # objects onto shared SQLAlchemy Table metadata. A later create_all on
        # any engine then emits duplicate CREATE INDEX statements. Drop those
        # ad-hoc indexes from metadata before constructing the scheduler.
        _clear_schedarray_adhoc_indexes(Base)
        self._scheduler = SqliteJobScheduler(
            db_path=path,
            auto_start_workers=auto_start_workers,
        )
        if max_workers is not None and auto_start_workers:
            try:
                pool = getattr(self._scheduler, "_worker_pool", None)
                if pool is not None and hasattr(pool, "max_workers"):
                    pool.max_workers = int(max_workers)
            except Exception:  # noqa: BLE001
                logger.debug("Could not set max_workers=%s", max_workers, exc_info=True)

    @property
    def scheduler(self):
        """Underlying :class:`~schedarray.SqliteJobScheduler` instance."""
        return self._scheduler

    def close(self) -> None:
        """Stop this service's local worker pool (jobs already finished are kept)."""
        _stop_service_workers(self)

    def submit(
        self,
        command: str | Sequence[str],
        *,
        working_dir: Path | str | None = None,
        job_name: str | None = None,
        output_file: Path | str | None = None,
        error_file: Path | str | None = None,
        tool: str = "dials",
        step: str = "job",
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        cpus: int = 1,
        timeout: int | None = None,
        priority: int = 2,
        max_retries: int = 0,
    ) -> str:
        """Submit a shell command and return the SchedArray job id.

        :param command: Shell string or argv sequence (argv is shell-quoted).
        :param working_dir: Job working directory.
        :param job_name: Display name (defaults to ``tool.step``).
        :param output_file: Live stdout log path (tailed by TerminalLogViewer).
        :param error_file: Stderr path.
        :param tool: Metadata tool filter key.
        :param step: Metadata step name.
        :param session_id: Optional session id for filtering.
        :param metadata: Extra metadata merged into the row.
        :param cpus: CPU request.
        :param timeout: Optional timeout seconds.
        :param priority: SchedArray priority (higher runs first).
        :param max_retries: Retry count on failure (default 0 for crystallography).
        :returns: Job UUID string.
        """
        if isinstance(command, (list, tuple)):
            cmd_str = " ".join(shlex.quote(str(c)) for c in command)
        else:
            cmd_str = str(command)

        work = Path(working_dir) if working_dir else None
        out = Path(output_file) if output_file else None
        err = Path(error_file) if error_file else None
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
        if err is not None:
            err.parent.mkdir(parents=True, exist_ok=True)
        if work is not None:
            work.mkdir(parents=True, exist_ok=True)

        name = job_name or f"{tool}.{step}"
        meta = build_job_metadata(
            tool=tool,
            step=step,
            work_dir=work,
            log_file=out,
            session_id=session_id,
            extra=metadata,
        )
        job_id = self._scheduler.submit_job(
            command=cmd_str,
            working_dir=str(work) if work else None,
            job_name=name,
            cpus=cpus,
            timeout=timeout,
            priority=priority,
            max_retries=max_retries,
            output_file=str(out) if out else None,
            error_file=str(err) if err else None,
            metadata=meta,
        )
        logger.info("Submitted SchedArray job %s (%s)", job_id, name)
        return job_id

    def get_status(self, job_id: str) -> Optional[dict[str, Any]]:
        """Return job status dict or ``None`` if missing."""
        return self._scheduler.get_job_status(job_id)

    def list_jobs(
        self,
        *,
        state: str | None = None,
        tool: str | None = None,
        session_id: str | None = None,
        work_dir: Path | str | None = None,
        limit: int | None = 200,
        app_only: bool = False,
    ) -> list[dict[str, Any]]:
        """List jobs, optionally filtered by metadata fields.

        :param state: SchedArray state filter (exact).
        :param tool: Filter ``metadata.tool`` (case-insensitive).
        :param session_id: Filter ``metadata.session_id``.
        :param work_dir: Keep jobs whose ``working_dir`` (or ``metadata.work_dir``)
            is this directory or inside it.
        :param limit: Max rows from SchedArray before client-side filters.
        :param app_only: If True, keep only jobs with ``metadata.app`` of
            ``diffraction_lab`` (or missing app for older rows).
        :returns: Matching job dictionaries (newest-friendly order preserved).
        """
        jobs = self._scheduler.list_jobs(state=state, limit=limit)
        tool_l = tool.lower() if tool else None
        work_s = str(Path(work_dir)) if work_dir else None
        out: list[dict[str, Any]] = []
        for job in jobs:
            meta = job.get("metadata") or {}
            if not isinstance(meta, dict):
                meta = {}
            if app_only:
                app = str(meta.get("app", "diffraction_lab")).lower()
                if app != "diffraction_lab":
                    continue
            if tool_l and str(meta.get("tool", "")).lower() != tool_l:
                continue
            if session_id and str(meta.get("session_id", "")) != str(session_id):
                continue
            if work_s:
                jd = job.get("working_dir") or meta.get("work_dir") or ""
                try:
                    job_dir, wanted = Path(jd).resolve(), Path(work_s).resolve()
                    same = job_dir == wanted or wanted in job_dir.parents
                except OSError:
                    same = str(Path(jd)) == work_s or work_s in str(jd)
                if not same:
                    continue
            out.append(job)
        return out

    def cancel(self, job_id: str) -> bool:
        """Cancel a pending/running job, stopping its command if it is running.

        SchedArray's ``cancel_job`` only marks the job; the command a local
        worker started (a shell and the program it runs) is stopped here.
        """
        ok = bool(self._scheduler.cancel_job(job_id))
        self._stop_process(job_id)
        return ok

    def _stop_process(self, job_id: str) -> None:
        """Terminate the local worker process running *job_id* and its children."""
        pool = getattr(self._scheduler, "_worker_pool", None)
        for worker in list((getattr(pool, "workers", None) or {}).values()):
            process = getattr(worker, "process", None)
            if getattr(worker, "current_job_id", None) != job_id or process is None:
                continue
            if process.poll() is None:
                # The job runs through a shell: stop what it started, then it.
                subprocess.run(["pkill", "-TERM", "-P", str(process.pid)], check=False)
                process.terminate()

    def wait_for_job(
        self,
        job_id: str,
        *,
        poll_interval: float = 0.5,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Block until *job_id* reaches a terminal state.

        :param job_id: SchedArray job id.
        :param poll_interval: Seconds between status polls.
        :param timeout: Optional wall-clock timeout for waiting (not the job).
        :returns: Final status dictionary.
        :raises TimeoutError: If *timeout* elapses before a terminal state.
        :raises RuntimeError: If the job disappears from the database.
        """
        started = time.monotonic()
        while True:
            status = self.get_status(job_id)
            if status is None:
                raise RuntimeError(f"SchedArray job {job_id} not found")
            state = str(status.get("state", "")).lower()
            if state in _TERMINAL_STATES:
                return status
            if timeout is not None and (time.monotonic() - started) > timeout:
                raise TimeoutError(
                    f"Timed out waiting for SchedArray job {job_id} "
                    f"(last state={state!r})"
                )
            time.sleep(poll_interval)

    def submit_and_wait(
        self,
        command: str | Sequence[str],
        *,
        poll_interval: float = 0.5,
        wait_timeout: float | None = None,
        **submit_kwargs: Any,
    ) -> dict[str, Any]:
        """Submit then :meth:`wait_for_job`.

        :returns: Final status dict (includes ``job_id``).
        """
        job_id = self.submit(command, **submit_kwargs)
        status = self.wait_for_job(
            job_id, poll_interval=poll_interval, timeout=wait_timeout
        )
        status = dict(status)
        status["job_id"] = job_id
        return status


def get_job_service(
    db_path: Path | str | None = None,
    *,
    force_new: bool = False,
) -> JobService:
    """Return the process-wide :class:`JobService` singleton.

    When *db_path* differs from the active singleton's database, the previous
    worker pool is stopped and a new service is created for that file. Each
    HDF5 master uses ``<stem>/schedarray_scheduler.db`` via
    :func:`configure_job_service_for_master` /
    :func:`schedarray_db_path_for_work_dir`.

    :param db_path: SQLite path. ``None`` keeps the current singleton (or
        creates one with SchedArray's default path if none exists).
    :param force_new: Always replace the singleton.
    """
    global _service, _service_db_path
    path = Path(db_path).expanduser().resolve() if db_path is not None else None
    need_new = force_new or _service is None
    if path is not None and _service_db_path is not None and path != _service_db_path:
        need_new = True
    if path is not None and _service is None:
        need_new = True
    if need_new:
        if _service is not None:
            _stop_service_workers(_service)
        _service = JobService(db_path=path, auto_start_workers=True)
        _service_db_path = _service.db_path
        logger.info("SchedArray JobService using DB %s", _service_db_path)
    elif path is not None and _service_db_path is None:
        # Singleton existed without an explicit path; recreate on requested path.
        _stop_service_workers(_service)
        _service = JobService(db_path=path, auto_start_workers=True)
        _service_db_path = _service.db_path
        logger.info("SchedArray JobService using DB %s", _service_db_path)
    return _service


def configure_job_service_for_master(master_path: Path | str) -> JobService:
    """Bind the singleton JobService to ``<master_stem>/schedarray_scheduler.db``.

    :param master_path: Path to the HDF5 / NeXus master file.
    :returns: Active :class:`JobService` for that dataset.
    """
    db = schedarray_db_path_for_master(master_path)
    return get_job_service(db_path=db)


def configure_job_service_for_work_dir(work_dir: Path | str) -> JobService:
    """Bind the singleton JobService to the SchedArray DB for *work_dir*'s stem."""
    db = schedarray_db_path_for_work_dir(work_dir)
    return get_job_service(db_path=db)


def reset_job_service() -> None:
    """Clear the singleton (for tests)."""
    global _service, _service_db_path
    if _service is not None:
        _stop_service_workers(_service)
    _service = None
    _service_db_path = None
