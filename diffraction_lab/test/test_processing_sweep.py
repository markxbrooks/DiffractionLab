"""Tests for the processing sweep and its frames."""

from __future__ import annotations

from pathlib import Path

import pytest

from diffraction_lab.processing_sweep import DataFile, Frames, ProcessingSweep


def test_frames() -> None:
    frames = Frames(1, 100)
    assert (frames.start, frames.end, frames.count) == (1, 100, 100)
    assert 1 in frames and 100 in frames and 101 not in frames and "1" not in frames
    assert list(Frames(3, 5)) == [3, 4, 5]
    assert frames.range_argument == "1:100" and str(frames) == "1–100"
    assert Frames.between(50, 10) == Frames(10, 50)
    assert frames.first(50) == Frames(1, 50) and Frames(1, 20).first(50) == Frames(1, 20)
    assert Frames(1, 2400).clamp(Frames(1001, 3000)) == Frames(1001, 2400)
    with pytest.raises(ValueError, match="start <= end"):
        Frames(10, 5)
    with pytest.raises(ValueError):
        Frames(1, 10).clamp(Frames(20, 30))
    with pytest.raises(AttributeError):
        frames.start = 5  # type: ignore[misc]  # frozen


def test_hdf5_sweep(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    import numpy as np

    master = tmp_path / "t_master.h5"
    with h5py.File(master, "w") as f:
        for name, n in (("data_000001", 1000), ("data_000002", 1000), ("data_000003", 400)):
            f.create_dataset(f"/entry/data/{name}", data=np.zeros((n, 1, 1), dtype=np.uint8))
        f.create_dataset("/entry/data/omega", data=np.zeros(2400))
    sweep = ProcessingSweep.from_image(master)
    assert sweep is not None and sweep.is_hdf5
    # The viewer shows one data file (1000 frames); the sweep is all 2400.
    assert sweep.frames == Frames(1, 2400)
    assert sweep.data_files == (DataFile("data_000001", Frames(1, 1000)),
                                DataFile("data_000002", Frames(1001, 2000)),
                                DataFile("data_000003", Frames(2001, 2400)))
    assert sweep.spot_wedge() == Frames(1, 50)
    # Viewer slider position 4 of data_000002 is sweep frame 1005.
    assert sweep.sweep_frame(4, "data_000002") == 1005
    assert sweep.sweep_frame(4) == 5
    assert ProcessingSweep.from_image(tmp_path / "missing_master.h5") is None


def test_frame_file_sweep(tmp_path: Path) -> None:
    for n in (5, 6, 7, 9):
        (tmp_path / f"x_1_{n:04d}.cbf").touch()
    sweep = ProcessingSweep.from_image(tmp_path / "x_1_0005.cbf")
    assert sweep is not None and not sweep.is_hdf5 and sweep.frames == Frames(5, 7)
    assert ProcessingSweep.from_image(tmp_path / "x_1_0009.cbf").frames == Frames(9, 9)
    assert ProcessingSweep.from_image(tmp_path / "image.tif").frames == Frames(1, 1)
