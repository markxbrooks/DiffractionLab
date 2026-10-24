"""Incident-beam geometry for DiffractionLab.

Beam-center coordinates are expressed in detector pixels.
Detector distance is in mm; wavelength is in Å.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DetectorParameters:
    """Detector pixel geometry and overload.

    ``overload`` is the count at which a pixel is saturated (HDF5
    ``saturation_value``, CBF ``Count_cutoff``), or ``None`` if unknown.
    """

    x_pixel_size: float = 0.075
    y_pixel_size: float = 0.075
    overload: float | None = None
