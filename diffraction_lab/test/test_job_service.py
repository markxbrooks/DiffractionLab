"""Unit tests for DiffractionLab JobService (SchedArray temp DB)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from diffraction_lab.jobs.metadata import build_job_metadata, metadata_log_file, metadata_tool
from diffraction_lab.jobs.service import (
    SCHEDARRAY_DB_NAME,
    JobService,
    configure_job_service_for_master,
    get_job_service,
    reset_job_service,
    schedarray_available,
    schedarray_db_path_for_master,
    schedarray_db_path_for_work_dir,
    use_schedarray,
)

pytestmark = pytest.mark.skipif(
    not schedarray_available(),
    reason="schedarray not installed",
)


@pytest.fixture(scope="module")
def _jobs_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One SQLite DB per test module (avoids SchedArray Index metadata dupes)."""
    return tmp_path_factory.mktemp("schedarray") / "test_scheduler.db"


@pytest.fixture()
def job_service(_jobs_db: Path):
    """Shared JobService for this module (singleton reset around each test)."""
    reset_job_service()
    service = JobService(db_path=_jobs_db, auto_start_workers=True, max_workers=1)
    yield service
    reset_job_service()


def test_build_job_metadata_round_trip(tmp_path: Path):
    meta = build_job_metadata(
        tool="dials",
        step="find_spots",
        work_dir=tmp_path,
        log_file=tmp_path / "dials.find_spots.log",
        session_id="sess-1",
        extra={"nproc": 4},
    )
    assert meta["app"] == "diffraction_lab"
    assert meta["tool"] == "dials"
    assert meta["step"] == "find_spots"
    assert meta["session_id"] == "sess-1"
    assert meta["nproc"] == 4
    assert meta["log_file"].endswith("dials.find_spots.log")


def test_submit_list_cancel_metadata(job_service: JobService, tmp_path: Path):
    work = tmp_path / "work"
    work.mkdir()
    out = work / "echo.log"
    err = work / "echo.err"
    job_id = job_service.submit(
        ["echo", "hello-schedarray"],
        working_dir=work,
        job_name="echo.test",
        output_file=out,
        error_file=err,
        tool="dials",
        step="echo",
        session_id="s1",
        max_retries=0,
    )
    assert job_id

    # Wait briefly for completion
    status = job_service.wait_for_job(job_id, poll_interval=0.2, timeout=30.0)
    assert status["state"] == "completed"
    assert out.is_file()
    assert "hello-schedarray" in out.read_text(encoding="utf-8")

    jobs = job_service.list_jobs(tool="dials", session_id="s1")
    assert any(j["job_id"] == job_id for j in jobs)
    listed = next(j for j in jobs if j["job_id"] == job_id)
    assert metadata_tool(listed) == "dials"
    assert metadata_log_file(listed) == out


def test_cancel_pending_job(job_service: JobService, tmp_path: Path):
    work = tmp_path / "work2"
    work.mkdir()
    # Long sleep so we can cancel while pending/running
    job_id = job_service.submit(
        "sleep 60",
        working_dir=work,
        job_name="sleep.cancel",
        output_file=work / "sleep.log",
        error_file=work / "sleep.err",
        tool="mosflm",
        step="sleep",
        max_retries=0,
    )
    # Give scheduler a moment to register
    time.sleep(0.3)
    ok = job_service.cancel(job_id)
    assert ok is True
    status = job_service.get_status(job_id)
    assert status is not None
    assert status["state"] == "cancelled"


def test_use_schedarray_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("USE_SCHEDARRAY", "false")
    assert use_schedarray() is False
    monkeypatch.setenv("USE_SCHEDARRAY", "true")
    assert use_schedarray() is True


def test_schedarray_db_path_for_master(tmp_path: Path):
    master = tmp_path / "thaumatin_9_1_master.h5"
    master.write_bytes(b"")
    db = schedarray_db_path_for_master(master)
    assert db.name == SCHEDARRAY_DB_NAME
    assert db.parent == tmp_path / "thaumatin_9_1_master"
    assert db.parent.is_dir()


def test_schedarray_db_path_for_work_dir(tmp_path: Path):
    work = tmp_path / "thaumatin_9_1_master" / "2026-10-15-20-29-46-dials"
    work.mkdir(parents=True)
    db = schedarray_db_path_for_work_dir(work)
    assert db == tmp_path / "thaumatin_9_1_master" / SCHEDARRAY_DB_NAME


def test_configure_job_service_switches_db(tmp_path: Path):
    reset_job_service()
    master_a = tmp_path / "dataset_a_master.h5"
    master_b = tmp_path / "dataset_b_master.h5"
    master_a.write_bytes(b"")
    master_b.write_bytes(b"")
    svc_a = configure_job_service_for_master(master_a)
    assert svc_a.db_path == schedarray_db_path_for_master(master_a)
    svc_b = configure_job_service_for_master(master_b)
    assert svc_b.db_path == schedarray_db_path_for_master(master_b)
    assert svc_a.db_path != svc_b.db_path
    # Singleton follows the latest master
    assert get_job_service().db_path == svc_b.db_path
    reset_job_service()
