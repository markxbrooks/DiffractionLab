"""Incident-beam geometry for DiffractionLab.

Beam-center coordinates are expressed in detector pixels.
Detector distance is in mm; wavelength is in Å.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DetectorParameters:
    """Detector pixel geometry."""

    x_pixel_size: float = 0.075
    y_pixel_size: float = 0.075
