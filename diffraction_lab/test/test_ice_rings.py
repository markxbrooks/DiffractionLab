"""Tests for the hexagonal-ice ring overlay."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from diffraction_lab.image_canvas import (  # noqa: E402
    DiffractionImageCanvas,
    build_ice_rings,
    ice_ring_resolutions,
)
from diffraction_lab.session.model import ViewerState  # noqa: E402
from diffraction_lab.session.serialize import viewer_from_dict, viewer_to_dict  # noqa: E402


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_ice_ring_resolutions_match_ice_ih() -> None:
    """The familiar Ih rings, with (001) systematically absent."""
    d = ice_ring_resolutions(1.5)
    assert [round(x, 2) for x in d[:6]] == [3.90, 3.67, 3.44, 2.67, 2.25, 2.07]
    assert all(x >= 1.5 for x in d)
    assert not any(abs(x - 7.338) < 0.01 for x in d)


def test_build_ice_rings_band_brackets_ring() -> None:
    """Each band spans the filtered 1/d² window around its ring radius."""
    rings = build_ice_rings(
        beam_x=1200.0, beam_y=1250.0, wavelength_a=0.9795, distance_mm=385.0, pixel_size_mm=0.172
    )
    assert len(rings) == len(ice_ring_resolutions())
    for ring in rings:
        assert ring.inner_px < ring.radius_px < ring.outer_px
        assert (ring.center_x, ring.center_y) == (1200.0, 1250.0)


def test_canvas_ice_ring_toggle() -> None:
    """Ice rings are hidden by default and toggle without rebuilding."""
    _app()
    canvas = DiffractionImageCanvas()
    rings = build_ice_rings(
        beam_x=100.0, beam_y=100.0, wavelength_a=1.0, distance_mm=200.0, pixel_size_mm=0.1
    )
    canvas.set_ice_rings(rings)
    assert canvas._show_ice_rings is False
    canvas.set_ice_rings_visible(True)
    assert canvas._show_ice_rings is True
    assert len(canvas._ice_rings) == len(rings)
    canvas.set_ice_rings(None)
    assert canvas._ice_rings == []


def test_session_round_trips_ice_ring_toggle() -> None:
    """``show_ice_rings`` is saved, and old sessions default to hidden."""
    assert viewer_from_dict(viewer_to_dict(ViewerState(show_ice_rings=True))).show_ice_rings is True
    assert viewer_from_dict({"show_rings": True}).show_ice_rings is False
