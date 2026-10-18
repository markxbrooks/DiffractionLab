"""Incident-beam geometry for DiffractionLab.

Beam-center coordinates are expressed in detector pixels.
Detector distance is in mm; wavelength is in Å.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BeamParameters:
    """Incident-beam geometry and related experimental parameters.

    ``x`` / ``y`` are the single source of truth for the beam centre in
    detector-pixel coordinates. :attr:`center` is a derived view.
    """

    x: float | None = None
    y: float | None = None
    detector_distance: float = 200.0
    incident_wavelength: float | None = None

    @property
    def center(self) -> tuple[float, float] | None:
        """Beam center in detector-pixel coordinates."""
        if self.x is None or self.y is None:
            return None
        return self.x, self.y

    @center.setter
    def center(self, value: tuple[float, float] | None) -> None:
        if value is None:
            self.x = None
            self.y = None
        else:
            self.x, self.y = float(value[0]), float(value[1])
