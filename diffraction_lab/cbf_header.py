"""Experiment geometry from a miniCBF (Pilatus / Eiger CBF) header.

The ``_array_data.header_contents`` text of a miniCBF carries lines such as::

    # Pixel_size 172e-6 m x 172e-6 m
    # Wavelength 0.97949 A
    # Detector_distance 0.38519 m
    # Beam_xy (1245.82, 1209.07) pixels

``Beam_xy`` is (fast, slow) in pixels, i.e. (image column, image row): the
same convention as HDF5 ``beam_centre_x`` / ``beam_centre_y`` and as the
viewer's image-pixel coordinates.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

_NUM = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
_PIXEL = re.compile(rf"Pixel_size\s+{_NUM}\s*m\s*x\s*{_NUM}\s*m", re.IGNORECASE)
_WAVELENGTH = re.compile(rf"Wavelength\s+{_NUM}\s*A", re.IGNORECASE)
_DISTANCE = re.compile(rf"Detector_distance\s+{_NUM}\s*m\b", re.IGNORECASE)
_CUTOFF = re.compile(rf"Count_cutoff\s+{_NUM}", re.IGNORECASE)
_BEAM = re.compile(rf"Beam_xy\s*\(\s*{_NUM}\s*,\s*{_NUM}\s*\)\s*(pixels|mm)?", re.IGNORECASE)


@dataclass(frozen=True)
class CbfGeometry:
    """Geometry from a miniCBF header; ``None`` where the header lacks it.

    :param beam_x_px: Beam centre along the fast axis (image column), pixels.
    :param beam_y_px: Beam centre along the slow axis (image row), pixels.
    :param pixel_x_mm: Pixel size along fast, mm.
    :param pixel_y_mm: Pixel size along slow, mm.
    :param distance_mm: Detector distance, mm.
    :param wavelength_a: Wavelength, Å.
    :param overload: Count at which a pixel is saturated (``Count_cutoff``).
    """

    beam_x_px: float | None = None
    beam_y_px: float | None = None
    pixel_x_mm: float | None = None
    pixel_y_mm: float | None = None
    distance_mm: float | None = None
    wavelength_a: float | None = None
    overload: float | None = None


def parse_minicbf_header(text: str) -> CbfGeometry:
    """Parse the miniCBF ``header_contents`` text.

    :param text: The header text (``# Key value`` lines).
    :returns: The geometry found (missing fields are ``None``).
    """
    pixel_x = pixel_y = None
    if (m := _PIXEL.search(text)) is not None:
        pixel_x, pixel_y = float(m.group(1)) * 1000.0, float(m.group(2)) * 1000.0
    beam_x = beam_y = None
    if (m := _BEAM.search(text)) is not None:
        beam_x, beam_y = float(m.group(1)), float(m.group(2))
        if (m.group(3) or "pixels").lower() == "mm":
            # Rare mm form: convert with the pixel size, or leave unset.
            if pixel_x and pixel_y:
                beam_x, beam_y = beam_x / pixel_x, beam_y / pixel_y
            else:
                beam_x = beam_y = None
    wavelength = float(m.group(1)) if (m := _WAVELENGTH.search(text)) is not None else None
    distance = float(m.group(1)) * 1000.0 if (m := _DISTANCE.search(text)) is not None else None
    overload = float(m.group(1)) if (m := _CUTOFF.search(text)) is not None else None
    return CbfGeometry(beam_x_px=beam_x, beam_y_px=beam_y, pixel_x_mm=pixel_x, pixel_y_mm=pixel_y,
                       distance_mm=distance, wavelength_a=wavelength, overload=overload)


def read_cbf_geometry(header: Mapping[str, str]) -> CbfGeometry:
    """Geometry from a fabio CBF header dictionary.

    :param header: ``fabio.open(path).header``.
    :returns: The geometry (all ``None`` for a non-miniCBF header).
    """
    return parse_minicbf_header(str(header.get("_array_data.header_contents", "")))
