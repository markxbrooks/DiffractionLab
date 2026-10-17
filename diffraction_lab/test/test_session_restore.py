"""Tests for Session / work-dir Processing pipeline rehydration."""

from __future__ import annotations

import json
from pathlib import Path

from diffraction_lab.dials_engine import (
    DialsImageRange,
    find_pointless_artifacts,
    load_index_result_from_work_dir,
    load_integration_result_from_work_dir,
    load_scale_result_from_work_dir,
    load_spot_result_from_work_dir,
    work_dir_has_report_inputs,
    work_dir_has_scale_inputs,
)
from diffraction_lab.pointless_runner import parse_pointless_log


def _minimal_expt(path: Path, *, hall: str = "P 1") -> None:
    """Write a tiny DIALS-like experiments JSON with one crystal."""
    data = {
        "crystal": [
            {
                "real_space_a": [50.0, 0.0, 0.0],
                "real_space_b": [0.0, 60.0, 0.0],
                "real_space_c": [0.0, 0.0, 70.0],
                "space_group_hall_symbol": hall,
            }
        ]
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_load_spot_result_from_work_dir(tmp_path: Path) -> None:
    work = tmp_path / "run-dials"
    work.mkdir()
    (work / "imported.expt").write_text("{}", encoding="utf-8")
    (work / "strong.refl").write_bytes(b"not-a-real-refl")
    (work / "dials.find_spots.log").write_text(
        "Saved 42 reflections to strong.refl\n", encoding="utf-8"
    )
    spots = load_spot_result_from_work_dir(
        work, image_range=DialsImageRange(1, 40)
    )
    assert spots is not None
    assert spots.n_spots == 42
    assert spots.imported_expt.name == "imported.expt"
    assert spots.strong_refl.name == "strong.refl"
    assert spots.image_range.first_image == 1
    assert spots.image_range.last_image == 40


def test_load_spot_result_missing(tmp_path: Path) -> None:
    assert load_spot_result_from_work_dir(tmp_path) is None


def test_load_index_and_bravais_from_work_dir(tmp_path: Path) -> None:
    work = tmp_path / "run-dials"
    work.mkdir()
    _minimal_expt(work / "refined.expt", hall="P 41 21 2")
    (work / "refined.refl").write_bytes(b"x")
    _minimal_expt(work / "indexed.expt")
    (work / "indexed.refl").write_bytes(b"x")
    summary = {
        "1": {
            "bravais": "aP",
            "unit_cell": [50, 60, 70, 90, 90, 90],
            "cb_op": "a,b,c",
            "max_angular_difference": 0.1,
            "rmsd": 0.05,
            "nspots": 100,
            "recommended": False,
        },
        "2": {
            "bravais": "tP",
            "unit_cell": [58, 58, 150, 90, 90, 90],
            "cb_op": "a,b,c",
            "max_angular_difference": 0.2,
            "rmsd": 0.04,
            "nspots": 200,
            "recommended": True,
        },
    }
    (work / "bravais_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    payload = load_index_result_from_work_dir(work)
    assert payload is not None
    indexed, bravais = payload
    assert indexed.experiments.name == "refined.expt"
    assert indexed.refined is True
    assert "41212" in indexed.solution.spacegroup.replace(" ", "")
    assert bravais is not None
    assert len(bravais.solutions) == 2
    assert any(s.recommended for s in bravais.solutions)


def test_load_integration_result_from_work_dir(tmp_path: Path) -> None:
    work = tmp_path / "run-dials"
    work.mkdir()
    _minimal_expt(work / "integrated.expt")
    (work / "integrated.refl").write_bytes(b"x")
    mtz = work / "integrated.mtz"
    mtz.write_bytes(b"MTZ")
    (work / "dials.integrate.log").write_text(
        "Saved 99 reflections to integrated.refl\n", encoding="utf-8"
    )
    integ = load_integration_result_from_work_dir(
        work, image_range=DialsImageRange(1, 751)
    )
    assert integ is not None
    assert integ.mtz_path == mtz
    assert integ.experiments.name == "integrated.expt"


def test_load_scale_result_from_work_dir(tmp_path: Path) -> None:
    work = tmp_path / "run-dials"
    work.mkdir()
    _minimal_expt(work / "scaled.expt")
    (work / "scaled.refl").write_bytes(b"x")
    _minimal_expt(work / "symmetrized.expt")
    (work / "symmetrized.refl").write_bytes(b"x")
    (work / "scaled.mtz").write_bytes(b"MTZ")
    (work / "dials.scale.html").write_text("<html></html>", encoding="utf-8")
    scale = load_scale_result_from_work_dir(work)
    assert scale is not None
    assert scale.merged_mtz is not None
    assert scale.scale_html is not None
    assert scale.scaled_expt.name == "scaled.expt"


def test_work_dir_has_report_and_scale_inputs(tmp_path: Path) -> None:
    work = tmp_path / "run-dials"
    work.mkdir()
    assert not work_dir_has_report_inputs(work)
    assert not work_dir_has_scale_inputs(work)
    (work / "dials.report.html").write_text("<html></html>", encoding="utf-8")
    assert work_dir_has_report_inputs(work)
    assert not work_dir_has_scale_inputs(work)
    _minimal_expt(work / "integrated.expt")
    (work / "integrated.refl").write_bytes(b"x")
    assert work_dir_has_scale_inputs(work)
    assert work_dir_has_report_inputs(work)


def test_find_pointless_artifacts_and_parse(tmp_path: Path) -> None:
    work = tmp_path / "run-dials"
    work.mkdir()
    hklin = work / "integrated.mtz"
    hklout = work / "pointless.mtz"
    hklin.write_bytes(b"IN")
    hklout.write_bytes(b"OUT")
    log = work / "250101-120000-pointless.log"
    log.write_text(
        "Best Solution:    space group P 41 21 2\n"
        "   Reindex operator:                   [h,k,l]\n"
        "   Space group confidence:             0.931\n",
        encoding="utf-8",
    )
    found_in, found_out, found_log = find_pointless_artifacts(work)
    assert found_in == hklin
    assert found_out == hklout
    assert found_log == log
    parsed = parse_pointless_log(log)
    assert parsed["space_group"] == "P 41 21 2"
    assert parsed["reindex_operator"] == "[h,k,l]"
    assert parsed["confidence"] == 0.931
