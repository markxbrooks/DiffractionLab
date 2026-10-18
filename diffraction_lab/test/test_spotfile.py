"""Unit tests for Mosflm spotfile helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from diffraction_lab.spotfile import copy_spotfile, write_merged_spotfile
from diffraction_lab.spotfile import Spot, SpotFileGeometry, SpotList


def test_copy_spotfile_appends_spt_suffix(tmp_path: Path) -> None:
    src = tmp_path / "msf00001.tmp"
    geom = SpotFileGeometry(
        image_width=100,
        image_height=100,
        pixel_size_mm=0.075,
        beam_x_mm=50.0,
        beam_y_mm=50.0,
    )
    spots = SpotList(
        spots=[Spot(x_mm=10.0, y_mm=20.0, intensity=100.0, sigma=5.0)],
        phi_start=0.0,
        phi_end=0.15,
    )
    write_merged_spotfile(src, geom, [spots])
    dest = copy_spotfile(src, tmp_path / "thaumatin")
    assert dest == tmp_path / "thaumatin.spt"
    assert dest.is_file()
    assert dest.read_bytes() == src.read_bytes()


def test_copy_spotfile_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        copy_spotfile(tmp_path / "missing.spt", tmp_path / "out.spt")
