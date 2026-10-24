"""Tests for running ProcessingLab stage by stage (spots → index → integrate)."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pytest

from diffraction_lab.processinglab_client import find_integrationlab
from diffraction_lab.processinglab_run import RunOptions, build_run_argv, load_run_record
from diffraction_lab.processinglab_stages import (
    STAGE_INDEX,
    SptSpot,
    autoindex,
    autoindex_commands,
    find_spots,
    findspots_commands,
    integrate,
    load_index_result,
    load_strategy,
    new_session_dir,
    read_spt,
    scan_of,
    spot_overlay,
    stage_dir,
    strategy,
    strategy_commands,
)
from diffraction_lab.run_step import RunStep
from diffraction_lab.processing_sweep import Frames, ProcessingSweep

INSULIN = Path(os.environ.get("CCP4", "/Applications/ccp4-9")) / "examples" / "data" / "insulin_1_001.img"

SPT = """        4148        4362 0.07500000    1.000000  270.000000
           0           1
  166.07662  172.53719
     161.40    181.35    0.150    0.075       355.0       0.5
     100.35     95.85    0.150    0.225       252.0       0.5
     -99.00    -99.00  -99.000  -99.000       -99.0     -99.0
     129.00    212.70    0.300    7.425        41.0       0.5
    -999.00   -999.00 -999.000 -999.000      -999.0    -999.0
   4148  4362    0.0750    1
"""


def test_commands() -> None:
    assert findspots_commands(Frames(1, 50), threshold=4, max_spots=2000) == [
        "FINDSPOTS THRESHOLD 4 MAXSPOTS 2000 FIND 1 FILE spots.spt MULTIFRAME RANGE 1 50", "GO", "EXIT"]
    lines = autoindex_commands(Path("/s/spots.spt"), threshold=10, dmax=320, character=32)
    assert lines[0] == "DMAX 320"
    assert lines[1].startswith("AUTOINDEX DPS THRESHOLD 10 CHARACTER 32 FILE ")
    assert lines[-2:] == ["GO", "EXIT"]
    assert "CHARACTER" not in autoindex_commands(Path("s.spt"))[0]


def test_strategy_commands_and_plan(tmp_path: Path) -> None:
    from diffraction_lab.processinglab_run import load_run_record

    cmds = strategy_commands(Path("/r/XPARM.XDS"), "P 41 21 2", completeness_pct=99.5, anomalous=True,
                             within=Frames(1, 900), resolution=1.8, rotation=60)
    assert cmds == ["XPARM /r/XPARM.XDS", "SYMMETRY P41212",
                    "STRATEGY COMPLETENESS 99.5 ANOMALOUS FRAMES 1 900 RESOLUTION 1.8 ROTATE 60", "GO", "EXIT"]
    assert strategy_commands(Path("/r/X"), "92")[2] == "STRATEGY COMPLETENESS 99 FULL"
    run = tmp_path / "02-strategy"
    run.mkdir()
    (run / "input.json").write_text("{}")
    shell = {"d_low_angstrom": 40.0, "d_high_angstrom": 2.0, "possible": 100, "observed": 97,
             "completeness_pct": 97.0, "multiplicity": 3.1, "anomalous_completeness_pct": 80.0}
    (run / "process.json").write_text(json.dumps({
        "schema_version": 3, "status": "completed",
        "stages": [{"name": "strategy", "status": "completed"}],
        "strategy": {"target_completeness_pct": 95.0, "anomalous": False, "oscillation_deg": 0.1,
                     "frames_collected": {"first": 1, "last": 1800}, "resolution_angstrom": {"low": 40.0, "high": 2.0},
                     "unique_possible": 100, "max_completeness_pct": 99.0, "max_anomalous_completeness_pct": 90.0,
                     "target": {"start_deg": 12.0, "end_deg": 102.0, "rotation_deg": 90.0, "images": 900,
                                "frames": {"first": 121, "last": 1020}, "completeness_pct": 97.0,
                                "anomalous_completeness_pct": 80.0, "multiplicity": 3.1, "shells": [shell]},
                     "rotation": None,
                     "curve": [{"rotation_deg": 1.0, "start_deg": 5.0, "completeness_pct": 2.0,
                                "anomalous_completeness_pct": 0.0}]}}))
    plan = load_strategy(load_run_record(run))
    assert plan.target.frames == Frames(121, 1020) and plan.target.images == 900
    assert plan.frames_collected == Frames(1, 1800) and plan.resolution == (40.0, 2.0)
    assert plan.curve == ((1.0, 5.0, 2.0, 0.0),) and plan.target.shells[0]["observed"] == 97


def test_xparm_option() -> None:
    argv = build_run_argv(Path("PL"), RunOptions(image=Path("x.h5"), xparm=Path("/i/XPARM.XDS"),
                                                 space_group="P4", refine=("orientation",)))
    assert argv[argv.index("--xparm") + 1] == "/i/XPARM.XDS"


def test_session_and_stage_dirs(tmp_path: Path) -> None:
    now = datetime(2026, 10, 23, 9, 30, 0)
    first = new_session_dir(tmp_path, now)
    second = new_session_dir(tmp_path, now)
    assert first.name == "session_20261023_093000" and second.name == "session_20261023_093000_2"
    assert stage_dir(first, STAGE_INDEX) == first / "02-index"
    (first / "02-index").mkdir()
    assert stage_dir(first, STAGE_INDEX) == first / "02-index-2"


def test_read_spt_and_overlay(tmp_path: Path) -> None:
    path = tmp_path / "spots.spt"
    path.write_text(SPT)
    pixel_mm, spots = read_spt(path)
    assert pixel_mm == 0.075
    assert spots == [SptSpot(161.40, 181.35, 0.150, 0.075, 355.0, 0.5),
                     SptSpot(100.35, 95.85, 0.150, 0.225, 252.0, 0.5),
                     SptSpot(129.00, 212.70, 0.300, 7.425, 41.0, 0.5)]
    # Frame 1 covers φ 0–0.15, frame 2 0.15–0.30; the third spot spans
    # frames 50–51 (φ 7.275–7.575).
    on1 = spot_overlay(spots, pixel_mm, frame=1, first_frame=1, phi_start_deg=0.0, osc_deg=0.15)
    assert on1 == [(161.40 / 0.075, 181.35 / 0.075, False)]
    assert len(spot_overlay(spots, pixel_mm, frame=2, first_frame=1, phi_start_deg=0.0, osc_deg=0.15)) == 1
    for frame in (50, 51):
        assert len(spot_overlay(spots, pixel_mm, frame=frame, first_frame=1, phi_start_deg=0.0,
                                osc_deg=0.15)) == 1
    assert spot_overlay(spots, pixel_mm, frame=3, first_frame=1, phi_start_deg=0.0, osc_deg=0.15) == []
    with pytest.raises(ValueError, match="not a spot file"):
        (tmp_path / "bad.spt").write_text("nonsense\n")
        read_spt(tmp_path / "bad.spt")


def test_load_index_result_and_scan(tmp_path: Path) -> None:
    run = tmp_path / "02-index"
    run.mkdir()
    cell = {"a": 57.94, "b": 58.24, "c": 150.33, "alpha": 90.4, "beta": 90.2, "gamma": 90.0}
    (run / "process.json").write_text(json.dumps({
        "schema_version": 2, "status": "completed",
        "indexing": {"selected_solution": {"indexed_fraction": 0.97, "rmsd_hkl": 0.093},
                     "lastgood_rank": 2,
                     "solutions": [
                         {"rank": 1, "character": 44, "lattice": "aP", "penalty": 0.0, "unit_cell": cell,
                          "metric_deviation_pct": 0.0, "space_groups": ["P1"], "selected": False},
                         {"rank": 2, "character": 11, "lattice": "tP", "penalty": 132.0, "unit_cell": cell,
                          "metric_deviation_pct": 0.464, "space_groups": ["P4", "P41"], "selected": True}]},
        "outputs": [{"kind": "xparm", "path": "XPARM.XDS"}]}))
    (run / "input.json").write_text(json.dumps({"experiment": {
        "frame_range": {"value": {"first": 3, "last": 50}}, "first_frame": 1,
        "oscillation_width": {"value": 0.15}, "phi_start_deg": 10.0}}))
    result = load_index_result(load_run_record(run))
    assert [s.lattice for s in result.solutions] == ["aP", "tP"]
    assert result.selected is not None and result.selected.character == 11
    assert result.selected.space_groups == ("P4", "P41")
    assert result.lastgood_rank == 2 and result.indexed_fraction == 0.97
    assert result.xparm == run / "XPARM.XDS"
    # φ is mapped from the first frame on disk, not the run's first frame.
    assert scan_of(result.record) == (1, 10.0, 0.15)


@pytest.mark.skipif(find_integrationlab() is None or not INSULIN.is_file(),
                    reason="needs ProcessingLab and the CCP4 insulin example")
def test_stages_insulin(tmp_path: Path) -> None:
    session = new_session_dir(tmp_path)
    sweep = ProcessingSweep.from_image(INSULIN)
    assert sweep is not None and sweep.frames == Frames(1, 1)
    spots = find_spots(sweep, session)  # default wedge: the sweep's first 50 frames
    assert spots.outputs["spot_list"].is_file()
    index = autoindex(sweep, session, spots.outputs["spot_list"])
    assert index.selected is not None and len(index.solutions) == 44
    # Choosing another lattice by character re-indexes in it.
    other = next(s for s in index.solutions if s.lattice == "aP")
    again = autoindex(sweep, session, spots.outputs["spot_list"], character=other.character)
    assert again.selected is not None and again.selected.lattice == "aP"
    assert again.record.directory.name == "02-index-2"
    # RunStep.STRATEGY: planning on a full turn in the lattice's highest symmetry.
    plan = strategy(sweep, session, index, index.selected.space_groups[-1], completeness_pct=95.0)
    assert plan.frames_collected is None and plan.unique_possible > 0
    assert plan.target is not None and plan.target.completeness_pct >= 95.0 and plan.target.images > 0
    assert plan.target.frames is None and plan.record.stage(RunStep.STRATEGY).status == "completed"
    assert [c for c in plan.curve if c[0] == plan.target.rotation_deg][0][2] >= 95.0
    # Within the one collected frame: far from complete, and said so.
    one = strategy(sweep, session, index, index.selected.space_groups[-1], completeness_pct=95.0,
                   within=sweep.frames)
    assert one.frames_collected == Frames(1, 1) and one.target is None and one.max_completeness_pct < 95.0
    assert one.record.directory.name == "02-strategy-2"
    run = integrate(sweep, session, index, index.selected.space_groups[0])
    assert run.frames == Frames(1, 1)  # the whole sweep, passed explicitly
    assert run.completed and run.mtz is not None and run.mtz.is_file()
    assert run.refinement["unit_cell"] == "refined"  # a first integration refines everything
