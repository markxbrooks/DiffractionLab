"""Tests for :mod:`diffraction_lab.run_step` and the stages of a ProcessingLab run record."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from diffraction_lab.processinglab_run import load_run_record
from diffraction_lab.run_step import RunStep, StageRecord

#: ProcessingLab's RunStep, which this Enum mirrors.
RUN_STEP_HPP = Path(os.environ.get("PROCESSINGLAB_SOURCE", Path.home() / "projects" / "ProcessingLab" / "processing-lab")) \
    / "include" / "mosflm" / "io" / "RunStep.hpp"


def test_run_step() -> None:
    assert [s.value for s in RunStep] == ["find_spots", "indexing", "defpix", "integrate", "strategy"]
    assert RunStep.DEFPIX.label == "RunStep.DEFPIX"
    assert RunStep.DEFPIX.keyword == RunStep.INTEGRATE.keyword == "PROCESS"
    for name, step in [("find_spots", RunStep.FIND_SPOTS), ("RunStep.INDEXING", RunStep.INDEXING),
                       ("Integrate", RunStep.INTEGRATE), ("DEFPIX", RunStep.DEFPIX),
                       ("findspots", RunStep.FIND_SPOTS), ("autoindex", RunStep.INDEXING),
                       ("process", RunStep.INTEGRATE)]:
        assert RunStep.parse(name) is step, name
    assert RunStep.parse("scale") is None


@pytest.mark.skipif(not RUN_STEP_HPP.is_file(), reason="needs ProcessingLab's source (set PROCESSINGLAB_SOURCE)")
def test_mirrors_processinglab() -> None:
    """Same steps, order, labels and record names as ProcessingLab's ``RunStep.hpp``."""
    text = RUN_STEP_HPP.read_text()
    members = re.search(r"enum class RunStep \{(.*?)\};", text, re.S).group(1)
    assert [m.strip() for m in members.split(",") if m.strip()] == [s.name for s in RunStep]
    ids = text[text.index("run_step_id("):text.index("run_step_keyword(")]
    for step in RunStep:
        assert f'"{step.label}"' in text
        assert f'return "{step.value}"' in ids


def test_record_stages(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "input.json").write_text("{}")
    stages = [
        {"name": "find_spots", "status": "completed", "duration_s": 3.2, "results": {"spots": 1500}},
        {"name": "indexing", "status": "completed", "duration_s": 1.0},
        {"name": "defpix", "status": "completed", "duration_s": 0.6, "message": "42 pixels ignored, 1000 trusted",
         "results": {"trusted_pixels": 1000, "ignored_pixels": 42}},
        {"name": "integrate", "status": "failed", "duration_s": 12.0, "error": "no reflections"},
        {"name": "polish", "status": "pending"},
    ]
    (run / "process.json").write_text(json.dumps({"schema_version": 3, "status": "failed", "stages": stages}))
    rec = load_run_record(run)
    assert [s.step for s in rec.stages] == [RunStep.FIND_SPOTS, RunStep.INDEXING, RunStep.DEFPIX,
                                            RunStep.INTEGRATE, None]
    assert rec.stage(RunStep.DEFPIX).results["ignored_pixels"] == 42
    assert rec.stage(RunStep.INTEGRATE).error == "no reflections" and rec.stage(RunStep.STRATEGY) is None
    assert rec.stages[-1].label == "polish"  # a name this version doesn't know is kept

    from diffraction_lab.processinglab_format import format_record

    html = format_record(rec)
    assert "RunStep.DEFPIX completed 0.6 s" in html and "RunStep.INTEGRATE failed 12.0 s" in html

    # Schema 2 names.
    old = StageRecord.from_json({"name": "autoindex", "status": "completed"})
    assert old.step is RunStep.INDEXING and old.label == "RunStep.INDEXING"
