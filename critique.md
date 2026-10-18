I think introducing a `BeamParameters` dataclass is a good refactoring for DiffractionLab. Your current `DiffractionLabApp` stores beam geometry in several independent attributes, and `beam_center` overlaps conceptually with `beam_x` and `beam_y`.

I'd make `BeamParameters` the canonical owner of the incident-beam geometry and related experimental parameters, rather than simply moving the existing attributes into a dataclass.

## 1. Recommended `BeamParameters` design

There are two issues with the proposed definition:

- `center`, `detector_distance`, and `incident_wavelength` are currently class attributes, not dataclass instance fields, because they lack type annotations.
- `center` duplicates information already represented by `x` and `y`, unless it has a clearly defined coordinate system or purpose.

I'd start with this:

```
from dataclasses import dataclass@dataclassclass BeamParameters:    """Incident-beam geometry and experimental parameters.    Beam-center coordinates are expressed in detector pixels.    Detector distance is in mm; wavelength is in Å.    """    x: float | None = None    y: float | None = None    detector_distance: float = 200.0    incident_wavelength: float | None = None    @property    def center(self) -> tuple[float, float] | None:        """Beam center in detector-pixel coordinates."""        if self.x is None or self.y is None:            return None        return self.x, self.y    @center.setter    def center(self, value: tuple[float, float] | None) -> None:        if value is None:            self.x = None            self.y = None        else:            self.x, self.y = value
```

This design gives you:

- Typed instance fields with sensible defaults.
- A single source of truth for the beam center.
- Backward-compatible conceptual access to `center`, without storing duplicate state.
- Explicit units in the documentation, which matter when calculating resolution rings and scattering geometry.

The `center` property is optional. If you don't need tuple-based access, I'd remove it and use `x` and `y` consistently.

## 2. Integrate it into `DiffractionLabApp`

Replace the individual beam-related attributes with one object:

```
class DiffractionLabApp(QMainWindow):    def __init__(self):        super().__init__()        self.beam_parameters = BeamParameters()        self.x_pixel_size = 0.075        self.y_pixel_size = 0.075        # Remaining application initialization...
```

Then migrate existing references:

| Existing attribute         | Replacement                                |
| -------------------------- | ------------------------------------------ |
| `self.beam_x`              | `self.beam_parameters.x`                   |
| `self.beam_y`              | `self.beam_parameters.y`                   |
| `self.beam_center`         | `self.beam_parameters.center`              |
| `self.detector_distance`   | `self.beam_parameters.detector_distance`   |
| `self.incident_wavelength` | `self.beam_parameters.incident_wavelength` |

For example, setting the beam center becomes:

```
self.beam_parameters.center = (beam_x, beam_y)
```

And accessing its coordinates:

```
center = self.beam_parameters.centerif center is not None:    x, y = center
```

Important: Don't retain both `beam_center` and `beam_parameters.center` as independently mutable attributes. That would preserve the synchronization problem the refactoring is intended to eliminate.

## 3. Should pixel size also belong in `BeamParameters`?

I'd keep pixel size separate initially, unless you're treating this class as the complete detector-and-beam geometry model.

There are two reasonable domain models:

### Option A — Beam parameters

Best for a focused refactor

- Beam center
- Detector distance
- Incident wavelength

Pixel size stays in the detector or image configuration.

### Option B — Experimental geometry

Better if geometry calculations are becoming widespread

- Beam center
- Detector distance
- Incident wavelength
- Pixel size in X and Y
- Potentially detector orientation and beam direction

This could evolve into an `ExperimentGeometry` or `DetectorGeometry` model, with clear ownership of each parameter.

For diffraction processing, detector pixel size is a detector property, whereas wavelength and incident-beam direction are properties of the incident beam. Detector distance describes their experimental geometry. Keeping these concepts distinct becomes valuable when you support multiple detectors or experiments.

## 4. One additional design decision: how should parameters be updated?

If your beam center is initially unknown and later populated from image headers, HDF5 metadata, or indexing results, the mutable dataclass above is appropriate.

If you eventually need to validate updates, distinguish measured values from defaults, or track metadata provenance, consider a more explicit model. For example, `200.0 mm` may be a fallback rather than a measured detector distance, and `None` for wavelength means unknown—not zero.

I'd also avoid adding calculations for resolution rings directly to `BeamParameters`. Keep it as a data model and put the calculations in a dedicated geometry or resolution utility.

My recommendation: implement Option A now, migrate all five attributes to the single `self.beam_parameters` object, and keep pixel dimensions separate. Once the code handling detector geometry and resolution rings is consolidated, decide whether a broader `ExperimentGeometry` model is justified.