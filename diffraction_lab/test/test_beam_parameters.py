"""Unit tests for :class:`~diffraction_lab.beam.BeamParameters`."""

from __future__ import annotations

from diffraction_lab.beam import BeamParameters


def test_defaults() -> None:
    beam = BeamParameters()
    assert beam.x is None
    assert beam.y is None
    assert beam.detector_distance == 200.0
    assert beam.incident_wavelength is None
    assert beam.center is None


def test_center_none_until_both_set() -> None:
    beam = BeamParameters(x=100.0)
    assert beam.center is None
    beam.y = 200.0
    assert beam.center == (100.0, 200.0)


def test_center_setter_updates_xy() -> None:
    beam = BeamParameters()
    beam.center = (512.5, 480.0)
    assert beam.x == 512.5
    assert beam.y == 480.0
    assert beam.center == (512.5, 480.0)


def test_center_setter_clear() -> None:
    beam = BeamParameters(x=10.0, y=20.0)
    beam.center = None
    assert beam.x is None
    assert beam.y is None
    assert beam.center is None
