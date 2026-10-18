"""Unit tests for ProcessingLab IntegrationLab discovery and dump parsing."""

from __future__ import annotations

from pathlib import Path

from diffraction_lab.processinglab_client import (
    build_autoindex_commands,
    parse_autoindex_dump,
    prefer_processinglab,
    spacegroup_for_system,
)


def test_parse_autoindex_dump_stdout() -> None:
    text = """
  AUTOINDEX: AUTOINDEX complete
INTEGRATIONLAB_CELL 78.19 78.19 78.19 90 90 90
INTEGRATIONLAB_MATRIX -0.01 0 0 0 -0.01 0 0 0 -0.01
INTEGRATIONLAB_SYSTEM cI
INTEGRATIONLAB_CHARACTER 12
INTEGRATIONLAB_NSPOTS 438
"""
    dump = parse_autoindex_dump(text)
    assert dump is not None
    assert dump.cell[0] == 78.19
    assert len(dump.matrix) == 9
    assert dump.system == "cI"
    assert dump.character == 12
    assert dump.n_spots == 438


def test_parse_autoindex_dump_sidecar() -> None:
    text = "CELL 50 60 70 90 90 90\nMATRIX 1 0 0 0 1 0 0 0 1\nSYSTEM aP\n"
    dump = parse_autoindex_dump(text)
    assert dump is not None
    assert dump.cell == (50.0, 60.0, 70.0, 90.0, 90.0, 90.0)
    assert dump.matrix[0] == 1.0
    assert dump.system == "aP"


def test_parse_autoindex_dump_incomplete() -> None:
    assert parse_autoindex_dump("INTEGRATIONLAB_CELL 1 2 3 4 5 6\n") is None


def test_spacegroup_for_system() -> None:
    assert spacegroup_for_system("cI") == "I23"
    assert spacegroup_for_system("aP") == "P1"
    assert spacegroup_for_system("unknown") == "P1"


def test_build_autoindex_commands(tmp_path: Path) -> None:
    cmds = build_autoindex_commands(
        directory=tmp_path,
        spotfile=tmp_path / "spots.spt",
        distance_mm=160.0,
        wavelength_a=0.979,
        beam_x_mm=94.0,
        beam_y_mm=94.7,
        pixel_mm=0.0816,
        prefer_system="cI",
    )
    joined = "\n".join(cmds)
    assert "AUTOINDEX DPS" in joined
    assert "FILE" in joined
    assert "SYSTEM cI" in joined
    assert cmds[-1] == "GO"


def test_prefer_processinglab_env(monkeypatch) -> None:
    monkeypatch.setenv("DIFFRACTIONLAB_BACKEND", "mosflm")
    assert prefer_processinglab() is False
    monkeypatch.setenv("DIFFRACTIONLAB_BACKEND", "processinglab")
    assert prefer_processinglab() is True
