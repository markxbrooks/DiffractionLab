"""ProcessingLab stage by stage: find spots → index → integrate.

Each stage takes the :class:`~diffraction_lab.processing_sweep.ProcessingSweep`
and :class:`~diffraction_lab.processing_sweep.Frames` to work on. Each is its
own ProcessingLab run in a session folder, so each has its own records (``input.json``, ``process.json``) and log::

    <session>/
      01-findspots/        FINDSPOTS → spots.spt
      02-index/            AUTOINDEX → XPARM.XDS, index.log, lattice solutions
      03-integrate/        ProcessingLab run --xparm … --spacegroup N → process.mtz
      04-ccp4/             Pointless → pointless.mtz, Aimless → aimless.mtz

Repeating a stage makes a new numbered folder (``02-index-2``, …): a run
folder is never reused. Symmetry and scaling (Pointless, Aimless) are in
:mod:`diffraction_lab.pointless_runner` and :mod:`diffraction_lab.aimless_runner`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

from diffraction_lab.processing_sweep import Frames, ProcessingSweep
from diffraction_lab.processinglab_run import (
    JobQueue,
    PixelExclusions,
    RunOptions,
    RunRecord,
    run_control_stage,
    run_processinglab,
)

STAGE_FINDSPOTS = "01-findspots"
STAGE_INDEX = "02-index"
STAGE_INTEGRATE = "03-integrate"
STAGE_CCP4 = "04-ccp4"
STAGE_DIMPLE = "05-dimple"


@dataclass(frozen=True)
class LatticeSolution:
    """One AUTOINDEX lattice solution (``process.json`` ``indexing.solutions``).

    :param rank: 1 = lowest penalty.
    :param character: Lattice character 1–44 (selects it: ``CHARACTER n``).
    :param lattice: Bravais lattice, e.g. ``"tP"``.
    :param penalty: DPS penalty (lower is better).
    :param cell: Conventional cell (Å, °).
    :param metric_deviation_pct: Departure from the lattice's constrained
        metric (%), or ``None``.
    :param space_groups: Possible space groups, lowest symmetry first.
    :param selected: Whether AUTOINDEX chose it.
    """

    rank: int
    character: int
    lattice: str
    penalty: float
    cell: tuple[float, float, float, float, float, float]
    metric_deviation_pct: float | None
    space_groups: tuple[str, ...]
    selected: bool


@dataclass(frozen=True)
class IndexResult:
    """An AUTOINDEX stage: its record and every lattice solution."""

    record: RunRecord
    solutions: tuple[LatticeSolution, ...]
    lastgood_rank: int
    indexed_fraction: float | None
    rmsd_hkl: float | None

    @property
    def selected(self) -> LatticeSolution | None:
        """The solution AUTOINDEX chose."""
        return next((s for s in self.solutions if s.selected), None)

    @property
    def xparm(self) -> Path:
        """Orientation, cell and geometry for integration."""
        return self.record.outputs.get("xparm", self.record.directory / "XPARM.XDS")


@dataclass(frozen=True)
class SptSpot:
    """One ``.spt`` spot: position (mm, fast / slow), φ range, intensity."""

    x_mm: float
    y_mm: float
    osc_range_deg: float
    mid_phi_deg: float
    intensity: float
    sigma: float


def new_session_dir(parent: Path, now: datetime | None = None) -> Path:
    """Create ``<parent>/session_YYYYMMDD_HHMMSS`` (``_2`` … if taken)."""
    stamp = (now or datetime.now()).strftime("session_%Y%m%d_%H%M%S")
    parent = Path(parent).expanduser()
    parent.mkdir(parents=True, exist_ok=True)
    for n in range(1, 1000):
        candidate = parent / (stamp if n == 1 else f"{stamp}_{n}")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"no free session folder under {parent}")


def stage_dir(session: Path, stage: str) -> Path:
    """The next unused folder for *stage* (``02-index``, ``02-index-2``, …)."""
    candidate = session / stage
    n = 1
    while candidate.exists():
        n += 1
        candidate = session / f"{stage}-{n}"
    return candidate


def findspots_commands(frames: Frames, *, threshold: float = 5.0, max_spots: int = 1500) -> list[str]:
    """CONTROL for a multi-frame spot search over *frames*."""
    return [
        f"FINDSPOTS THRESHOLD {threshold:g} MAXSPOTS {int(max_spots)} FIND {frames.start} "
        f"FILE spots.spt MULTIFRAME RANGE {frames.start} {frames.end}",
        "GO",
        "EXIT",
    ]


def autoindex_commands(spot_file: Path, *, threshold: float = 15.0, dmax: float | None = None,
                       character: int | None = None) -> list[str]:
    """CONTROL for DPS autoindexing from *spot_file*.

    :param spot_file: The ``.spt`` from the spot search.
    :param threshold: Minimum I/σ of spots used.
    :param dmax: Longest cell edge searched (Å), or ``None`` (200 Å).
    :param character: Force the lattice solution with this character (1–44).
    """
    line = f"AUTOINDEX DPS THRESHOLD {threshold:g}"
    if character is not None:
        line += f" CHARACTER {int(character)}"
    line += f" FILE {Path(spot_file).resolve()}"
    lines = [f"DMAX {dmax:g}"] if dmax else []
    return [*lines, line, "GO", "EXIT"]


def find_spots(sweep: ProcessingSweep, session: Path, frames: Frames | None = None, *, threshold: float = 5.0,
               max_spots: int = 1500, on_line: Callable[[str], None] | None = None,
               jobs: JobQueue | None = None) -> RunRecord:
    """Stage 1: spot search over *frames* (default: the sweep's first 50) into a new stage folder.

    With *jobs*, every stage runs as a SchedArray job (listed in a Jobs tab).
    """
    out = stage_dir(session, STAGE_FINDSPOTS)
    wedge = frames or sweep.spot_wedge()
    return run_control_stage(sweep.image, findspots_commands(wedge, threshold=threshold, max_spots=max_spots),
                             out, on_line=on_line, jobs=jobs, step="findspots")


def autoindex(sweep: ProcessingSweep, session: Path, spot_file: Path, *, threshold: float = 15.0,
              dmax: float | None = None, character: int | None = None,
              on_line: Callable[[str], None] | None = None, jobs: JobQueue | None = None) -> IndexResult:
    """Stage 2: autoindex *spot_file* (optionally forcing one lattice solution)."""
    out = stage_dir(session, STAGE_INDEX)
    record = run_control_stage(
        sweep.image, autoindex_commands(spot_file, threshold=threshold, dmax=dmax, character=character), out,
        on_line=on_line, jobs=jobs, step="index")
    return load_index_result(record)


def integrate(sweep: ProcessingSweep, session: Path, index: IndexResult, space_group: str, *,
              frames: Frames | None = None, refine: Sequence[str] | None = None,
              on_line: Callable[[str], None] | None = None, jobs: JobQueue | None = None,
              exclusions: PixelExclusions | None = None) -> RunRecord:
    """Stage 3: integrate *frames* (default: the whole sweep) in *space_group*.

    Runs ``ProcessingLab run --xparm <index>/XPARM.XDS --spacegroup …
    --range start:end``: the indexing cell is constrained to the space group
    (CELLFIX) and everything is refined unless *refine* says otherwise. The
    frames are always passed explicitly. The run also writes the ``xds/``,
    ``xia2/`` and ``mosflm/`` repeat inputs. DEFPIX ignores *exclusions* as
    well as the shaded, defective and hot pixels it finds.
    """
    out = stage_dir(session, STAGE_INTEGRATE)
    options = RunOptions(image=sweep.image, frames=frames or sweep.frames, parent_dir=session, output_dir=out,
                         space_group=space_group, xparm=index.xparm,
                         refine=tuple(refine) if refine is not None else
                         ("orientation", "cell", "distance", "beam", "tilt_twist", "mosaicity"),
                         exclusions=exclusions)
    return run_processinglab(options, on_line=on_line, jobs=jobs, step="integrate")


def load_index_result(record: RunRecord) -> IndexResult:
    """Read the lattice solutions of an AUTOINDEX stage from its ``process.json``."""
    process = json.loads((record.directory / "process.json").read_text(encoding="utf-8"))
    indexing = process.get("indexing") or {}
    selected = indexing.get("selected_solution") or {}
    solutions = []
    for row in indexing.get("solutions") or []:
        c = row.get("unit_cell") or {}
        solutions.append(LatticeSolution(
            rank=int(row["rank"]),
            character=int(row["character"]),
            lattice=str(row["lattice"]),
            penalty=float(row["penalty"]),
            cell=tuple(float(c[k]) for k in ("a", "b", "c", "alpha", "beta", "gamma")),  # type: ignore[arg-type]
            metric_deviation_pct=(None if row.get("metric_deviation_pct") is None
                                  else float(row["metric_deviation_pct"])),
            space_groups=tuple(row.get("space_groups") or ()),
            selected=bool(row.get("selected")),
        ))
    return IndexResult(
        record=record,
        solutions=tuple(solutions),
        lastgood_rank=int(indexing.get("lastgood_rank") or 0),
        indexed_fraction=selected.get("indexed_fraction"),
        rmsd_hkl=selected.get("rmsd_hkl"),
    )


def read_spt(path: Path) -> tuple[float, list[SptSpot]]:
    """Read a MOSFLM ``.spt`` spot file.

    :param path: The file.
    :returns: ``(pixel_size_mm, spots)``.
    :raises ValueError: If the header is malformed.
    """
    lines = Path(path).read_text(encoding="ascii", errors="replace").splitlines()
    try:
        pixel_mm = float(lines[0].split()[2])
    except (IndexError, ValueError) as ex:
        raise ValueError(f"{path}: not a spot file") from ex
    spots: list[SptSpot] = []
    for line in lines[3:]:
        fields = line.split()
        if len(fields) < 6:
            continue
        try:
            x, y, osc, phi, i, s = (float(f) for f in fields[:6])
        except ValueError:
            continue
        if x <= -999.0:
            break  # end sentinel
        if x <= -99.0:
            continue  # wedge separator
        spots.append(SptSpot(x, y, osc, phi, i, s))
    return pixel_mm, spots


def spot_overlay(spots: Sequence[SptSpot], pixel_mm: float, *, frame: int, first_frame: int,
                 phi_start_deg: float, osc_deg: float) -> list[tuple[float, float, bool]]:
    """Spots on *frame*, as the main window's overlay markers.

    A spot is on a frame when the frame's φ range overlaps the spot's
    (``mid_phi`` ± half its oscillation range).

    :param spots: From :func:`read_spt`.
    :param pixel_mm: Pixel size (mm); ``.spt`` positions are mm along the
        image's fast and slow axes.
    :param frame: Frame number shown.
    :param first_frame: The sweep's first frame number.
    :param phi_start_deg: φ at the start of *first_frame*.
    :param osc_deg: Oscillation per frame.
    :returns: ``(x_px, y_px, False)`` markers (found spots).
    """
    if pixel_mm <= 0 or osc_deg <= 0:
        return []
    lo = phi_start_deg + (frame - first_frame) * osc_deg
    hi = lo + osc_deg
    eps = 1e-6 * max(1.0, abs(hi))
    out = []
    for s in spots:
        half = 0.5 * max(s.osc_range_deg, 0.0)
        if s.mid_phi_deg + half > lo + eps and s.mid_phi_deg - half < hi - eps:
            out.append((s.x_mm / pixel_mm, s.y_mm / pixel_mm, False))
        elif half == 0.0 and math.isclose(s.mid_phi_deg, lo):
            out.append((s.x_mm / pixel_mm, s.y_mm / pixel_mm, False))
    return out


def scan_of(record: RunRecord) -> tuple[int, float, float] | None:
    """``(first_frame, phi_start_deg, osc_deg)`` of a stage's sweep, from ``input.json``."""
    try:
        experiment = json.loads((record.directory / "input.json").read_text(encoding="utf-8"))["experiment"]
        # The first frame on disk (phi_start_deg is its φ); older records
        # lack it, so fall back to the run's first frame.
        first = int(experiment.get("first_frame") or experiment["frame_range"]["value"]["first"])
        osc = float(experiment["oscillation_width"]["value"])
        phi0 = float(experiment.get("phi_start_deg") or 0.0)
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return None
    return first, phi0, osc
