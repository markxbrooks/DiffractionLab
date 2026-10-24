"""Tests for driving ``ProcessingLab run`` and reading its run records."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from diffraction_lab.processing_sweep import Frames
from diffraction_lab.processinglab_client import find_integrationlab
from diffraction_lab.processinglab_run import (
    ProcessingLabError,
    RunOptions,
    build_run_argv,
    latest_run_dir,
    load_run_record,
    rerun_options,
    run_processinglab,
)

EXE = Path("/opt/ProcessingLab")
INSULIN = Path(os.environ.get("CCP4", "/Applications/ccp4-9")) / "examples" / "data" / "insulin_1_001.img"


def _write_run(directory: Path, *, status: str = "completed", version: int = 2) -> Path:
    """A minimal run directory with schema 2 (or 1) records."""
    directory.mkdir(parents=True)
    (directory / "input.json").write_text(json.dumps({
        "inputs": {"images": [{"path": "/data/t_master.h5"}]},
        "experiment": {"frame_range": {"value": {"first": 1, "last": 100}}},
    }))
    process: dict = {"schema_version": version, "status": status, "error": None,
                     "outputs": [{"kind": "mtz", "path": "process.mtz"},
                                 {"kind": "xds_inp", "path": "xds/XDS.INP"}]}
    cell = {"a": 57.938, "b": 57.938, "c": 150.244, "alpha": 90, "beta": 90, "gamma": 90}
    if version == 2:
        process["experiment"] = {
            "crystal": {"unit_cell": cell, "space_group": {"number": 75, "symbol": "P 4"}},
            "wavelength_angstrom": 0.9795,
            "detector": {"distance_mm": 199.98, "beam_centre_mm": {"x": 166.12, "y": 172.93},
                         "tilt_twist_mrad": {"tilt": 9.39, "twist": -8.43}},
            "mosaicity_deg": 0.101,
        }
        process["refinement"] = {"stage": "process",
                                 "parameters": {"orientation": "refined", "unit_cell": "fixed"}}
        process["integration"] = {"counts": {"integrated_reflections": 60881}}
        process["quality"] = {"total": {"r_meas_pct": 17.9, "cc_half_pct": 97.6}}
    else:
        process["refinement"] = {"unit_cell": cell, "space_group_number": 75, "detector_distance_mm": 200.0,
                                 "beam_centre_mm": [166.1, 172.9], "tilt_twist_mrad": [9.6, -8.3],
                                 "mosaicity_deg": 0.096}
        process["integration"] = {"reflections": 547829}
    if status != "completed":
        process["error"] = "indexing failed: too few spots"
    (directory / "process.json").write_text(json.dumps(process))
    return directory


def test_build_run_argv_first_run() -> None:
    argv = build_run_argv(EXE, RunOptions(image=Path("/d/x_master.h5"), frames=Frames(1, 100),
                                          space_group="P41212", cell=(57.9, 57.9, 150.2, 90, 90, 90)))
    assert argv == [str(EXE), "run", "--image", "/d/x_master.h5", "--range", "1:100",
                    "--spacegroup", "P41212", "--cell", "57.9", "57.9", "150.2", "90", "90", "90"]


def test_build_run_argv_rerun_and_refine() -> None:
    argv = build_run_argv(EXE, RunOptions(from_run=Path("/r/run1"), refine=("orientation", "cell"),
                                          output_dir=Path("again")))
    assert argv == [str(EXE), "run", "--from", "/r/run1", "--refine", "orientation,cell",
                    "--output-dir", "again"]
    # An empty refine list holds everything fixed.
    assert build_run_argv(EXE, RunOptions(from_run=Path("/r"), refine=()))[-2:] == ["--refine", "none"]


def test_build_run_argv_refusals() -> None:
    with pytest.raises(ValueError, match="needs an image"):
        build_run_argv(EXE, RunOptions())
    with pytest.raises(ValueError, match="unknown refine parameter"):
        build_run_argv(EXE, RunOptions(image=Path("x.cbf"), refine=("shape",)))


def test_load_run_record_schema2(tmp_path: Path) -> None:
    rec = load_run_record(_write_run(tmp_path / "run"))
    assert rec.completed and rec.can_rerun
    assert rec.schema_version == 2
    assert rec.space_group_number == 75 and rec.space_group_symbol == "P 4"
    assert rec.cell == (57.938, 57.938, 150.244, 90.0, 90.0, 90.0)
    assert rec.distance_mm == 199.98 and rec.beam_mm == (166.12, 172.93)
    assert rec.tilt_twist_mrad == (9.39, -8.43) and rec.mosaicity_deg == 0.101
    assert rec.refinement == {"orientation": "refined", "unit_cell": "fixed"}
    assert rec.integrated_reflections == 60881
    assert rec.quality_total["r_meas_pct"] == 17.9
    assert rec.image == Path("/data/t_master.h5") and rec.frames == Frames(1, 100)
    # Output paths are relative to the run directory in the record.
    assert rec.mtz == tmp_path / "run" / "process.mtz"
    assert rec.outputs["xds_inp"] == tmp_path / "run" / "xds" / "XDS.INP"


def test_load_run_record_schema1(tmp_path: Path) -> None:
    rec = load_run_record(_write_run(tmp_path / "old", version=1) / "process.json")
    assert rec.schema_version == 1
    assert rec.space_group_number == 75
    assert rec.beam_mm == (166.1, 172.9) and rec.tilt_twist_mrad == (9.6, -8.3)
    assert rec.integrated_reflections == 547829


def test_rerun_options(tmp_path: Path) -> None:
    rec = load_run_record(_write_run(tmp_path / "run"))
    opts = rerun_options(rec, refine=("orientation", "cell"), space_group="P 41 21 2")
    assert opts.from_run == rec.directory and opts.parent_dir == tmp_path
    assert opts.refine == ("orientation", "cell") and opts.space_group == "P 41 21 2"
    assert opts.image is None  # the images come from the earlier run
    failed = load_run_record(_write_run(tmp_path / "failed", status="failed"))
    with pytest.raises(ValueError, match="only a completed run"):
        rerun_options(failed)


def test_latest_run_dir(tmp_path: Path) -> None:
    assert latest_run_dir(tmp_path) is None
    first = _write_run(tmp_path / "processing_1")
    second = _write_run(tmp_path / "processing_2")
    os.utime(first / "process.json", (1, 1))
    assert latest_run_dir(tmp_path) == second
    assert latest_run_dir(tmp_path / "missing") is None


def _fake_exe(tmp_path: Path, body: str) -> Path:
    exe = tmp_path / "ProcessingLab"
    exe.write_text("#!/bin/sh\n" + body)
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return exe


def test_run_processinglab_refused(tmp_path: Path) -> None:
    exe = _fake_exe(tmp_path, 'echo "ProcessingLab run: RunMode.XDS needs --xds-inp XDS.INP"\nexit 2\n')
    with pytest.raises(ProcessingLabError, match="needs --xds-inp") as info:
        run_processinglab(RunOptions(image=Path("x.cbf"), parent_dir=tmp_path), executable=exe)
    assert info.value.run_dir is None


def test_run_processinglab_failed_run(tmp_path: Path) -> None:
    run = _write_run(tmp_path / "processing_x", status="failed")
    exe = _fake_exe(tmp_path, f'echo "line one"\necho " Output directory: {run}"\nexit 1\n')
    lines: list[str] = []
    with pytest.raises(ProcessingLabError, match="too few spots") as info:
        run_processinglab(RunOptions(image=Path("x.cbf"), parent_dir=tmp_path), executable=exe,
                          on_line=lines.append)
    assert info.value.run_dir == run
    assert lines[0] == "line one"


@pytest.mark.skipif(find_integrationlab() is None or not INSULIN.is_file(),
                    reason="needs ProcessingLab and the CCP4 insulin example")
def test_run_and_rerun_insulin(tmp_path: Path) -> None:
    rec = run_processinglab(RunOptions(image=INSULIN, parent_dir=tmp_path))
    assert rec.completed and rec.mtz is not None and rec.mtz.is_file()
    assert rec.space_group_number is not None and rec.cell is not None
    again = run_processinglab(rerun_options(rec, refine=("orientation", "cell")))
    assert again.completed
    assert again.refinement["unit_cell"] == "refined"
    assert again.refinement["detector_distance"] == "fixed"
    assert again.directory.parent == rec.directory.parent
