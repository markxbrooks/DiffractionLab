"""Tests for miniCBF header geometry (beam centre for rings and the beam cross)."""

from __future__ import annotations

from pathlib import Path

import pytest

from diffraction_lab.cbf_header import CbfGeometry, parse_minicbf_header, read_cbf_geometry

# From 76c1b_0002212_1_0001.cbf (Pilatus 6M, Diamond).
PILATUS_HEADER = """# Detector: PILATUS 6M Prosport+, S/N 60-0100 Diamond\r
# Pixel_size 172e-6 m x 172e-6 m\r
# Silicon sensor, thickness 0.000320 m\r
# Wavelength 0.97949 A\r
# Energy_range (0, 0) eV\r
# Detector_distance 0.38519 m\r
# Detector_Voffset 0.00000 m\r
# Beam_xy (1245.82, 1209.07) pixels\r
# Start_angle 0.0000 deg.\r
"""

CBF_76C1B = Path("/Users/brooks/projects/test_xray_data_cbf/76c1b_0002212_1_0001.cbf")


def test_parse_pilatus_header() -> None:
    g = parse_minicbf_header(PILATUS_HEADER)
    assert (g.beam_x_px, g.beam_y_px) == (1245.82, 1209.07)
    assert g.pixel_x_mm == pytest.approx(0.172) and g.pixel_y_mm == pytest.approx(0.172)
    assert g.distance_mm == pytest.approx(385.19)
    assert g.wavelength_a == 0.97949


def test_parse_beam_in_mm() -> None:
    g = parse_minicbf_header("# Pixel_size 172e-6 m x 172e-6 m\n# Beam_xy (214.28, 207.96) mm\n")
    assert g.beam_x_px == pytest.approx(214.28 / 0.172)
    assert g.beam_y_px == pytest.approx(207.96 / 0.172)
    # Without a pixel size an mm centre cannot be placed.
    assert parse_minicbf_header("# Beam_xy (214.28, 207.96) mm\n").beam_x_px is None


def test_parse_missing_fields() -> None:
    assert parse_minicbf_header("") == CbfGeometry()
    assert read_cbf_geometry({"Content-Type": "application/octet-stream"}) == CbfGeometry()
    g = read_cbf_geometry({"_array_data.header_contents": "# Wavelength 1.0 A\n"})
    assert g.wavelength_a == 1.0 and g.beam_x_px is None


@pytest.mark.skipif(not CBF_76C1B.is_file(), reason="needs the 76c1b CBF sweep")
def test_read_real_cbf() -> None:
    fabio = pytest.importorskip("fabio")
    image = fabio.open(str(CBF_76C1B))
    g = read_cbf_geometry(image.header)
    rows, cols = image.data.shape
    # Beam_xy is (column, row): inside the 2463 × 2527 frame, near its middle.
    assert (cols, rows) == (2463, 2527)
    assert 0.4 * cols < g.beam_x_px < 0.6 * cols and 0.4 * rows < g.beam_y_px < 0.6 * rows
