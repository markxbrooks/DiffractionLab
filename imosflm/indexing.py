"""Mosflm-style autoindexing (findspots → DPS → refine), mirroring TCL.

Command templates follow ``Mosflm::findspots``, ``Mosflm::index`` and
``Mosflm::index2`` in ``src/mosflm.tcl``. XML parsing follows
``Solution::parseSolution`` / ``Latticetab`` in the iMosflm sources.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from imosflm.mosflm_client import MosflmClient, MosflmMessage, find_ipmosflm
from imosflm.spotfile import (
    Spot,
    SpotFileGeometry,
    SpotList,
    find_spots_in_frame,
    new_temp_spotfilename,
    write_merged_spotfile,
)

logger = logging.getLogger(__name__)


def as_float(value) -> float:
    """Coerce NeXus scalars / length-1 arrays to ``float``."""
    arr = np.asarray(value)
    if arr.size == 0:
        raise ValueError("empty numeric value")
    return float(arr.reshape(-1)[0])


# First space group for each Bravais lattice type (indexing.tcl).
SPACEGROUP_FOR_LATTICE: dict[str, str] = {
    "aP": "P1",
    "mP": "P2",
    "mC": "C2",
    "mI": "C2",
    "oP": "P222",
    "oC": "C222",
    "oF": "F222",
    "oI": "I222",
    "tP": "P4",
    "tI": "I4",
    "hP": "P3",
    "hR": "h3",
    "cP": "P23",
    "cF": "F23",
    "cI": "I23",
}

# Full space-group lists per lattice (``::spacegroup($lattice)`` in indexing.tcl).
SPACEGROUPS_BY_LATTICE: dict[str, tuple[str, ...]] = {
    "aP": ("P1",),
    "mP": ("P2", "P21"),
    "mC": ("C2",),
    "mI": ("C2",),
    "oP": (
        "P222",
        "P2221",
        "P2122",
        "P2212",
        "P21212",
        "P22121",
        "P21221",
        "P212121",
    ),
    "oC": ("C222", "C2221"),
    "oF": ("F222",),
    "oI": ("I222", "I212121"),
    "tP": (
        "P4",
        "P41",
        "P42",
        "P43",
        "P422",
        "P4212",
        "P4122",
        "P41212",
        "P4222",
        "P42212",
        "P4322",
        "P43212",
    ),
    "tI": ("I4", "I41", "I422", "I4122"),
    "hP": (
        "P3",
        "P31",
        "P32",
        "P312",
        "P321",
        "P3112",
        "P3121",
        "P3212",
        "P3221",
        "P6",
        "P61",
        "P65",
        "P62",
        "P64",
        "P63",
        "P622",
        "P6122",
        "P6522",
        "P6222",
        "P6422",
        "P6322",
    ),
    "hR": ("h3", "h32"),
    "cP": ("P23", "P213", "P432", "P4232", "P4332", "P4132"),
    "cF": ("F23", "F432", "F4132"),
    "cI": ("I23", "I213", "I432", "I4132"),
}

KNOWN_SPACEGROUPS: frozenset[str] = frozenset(
    sg for groups in SPACEGROUPS_BY_LATTICE.values() for sg in groups
)


def spacegroups_for_lattice(lattice: str) -> tuple[str, ...]:
    """Return candidate space groups for a Bravais lattice code (e.g. ``tP``)."""
    return SPACEGROUPS_BY_LATTICE.get(lattice, ("P1",))


def normalize_spacegroup(name: str) -> str:
    """Normalise a space-group symbol the way iMosflm does (indexwizard.tcl)."""
    trimmed = "".join(str(name).split())
    if not trimmed:
        return ""
    if trimmed[0] in ("h", "H"):
        return "h" + trimmed[1:] if trimmed[0] == "H" else trimmed
    return trimmed.upper()


def default_spacegroup_for_lattice(lattice: str) -> str:
    """Return the default (first) space group for *lattice*."""
    groups = spacegroups_for_lattice(lattice)
    return groups[0] if groups else "P1"


@dataclass
class IndexingGeometry:
    """Experiment geometry used for spotfinding / autoindexing."""

    directory: Path
    template: str
    image_number: int
    beam_x_mm: float
    beam_y_mm: float
    distance_mm: float
    wavelength_a: float
    pixel_size_mm: float
    image_width: int
    image_height: int
    two_theta: float = 0.0
    phi_start: float = 0.0
    phi_end: float = 0.5
    gain: float = 1.0
    nullpix: float = -1.0
    yscale: float = 1.0
    is_hdf5_master: bool = False


@dataclass
class IndexingSettings:
    """Defaults matching iMosflm session variables for indexing / spots."""

    threshold: float = 5.0
    i_sig_i: float = 5.0
    hkldev_max: float = 0.300
    numvectors: int = 30
    fix_distance_indexing: bool = True
    exclude_auto: bool = False
    sigma_cutoff: float = 2.50
    max_penalty_to_refine: float = 50.0
    spot_threshold_sigma: float = 5.0
    max_spots_per_frame: int = 400
    mosaic_block_size: float = 100.0
    min_spots_to_index: int = 12


@dataclass(slots=True)
class IndexSolution:
    """One Bravais-lattice solution from Mosflm autoindexing."""

    number: int
    kind: str  # raw | reg | ref
    penalty: float
    lattice: str
    spacegroup: str
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    matrix: tuple[float, ...] = field(default_factory=tuple)
    reflections_used: int | None = None
    spot_deviation_pos: float | None = None
    spot_deviation_phi: float | None = None
    beam_x: float | None = None
    beam_y: float | None = None
    volume: float | None = None

    @property
    def cell_tuple(self) -> tuple[float, float, float, float, float, float]:
        return (self.a, self.b, self.c, self.alpha, self.beta, self.gamma)


@dataclass
class SpotFindingResult:
    """Spot search on one or more frames (iMosflm §5.1), before DPS indexing."""

    spotfile: Path
    n_spots: int
    spotlists: list[SpotList]
    frame_indices: list[int]
    geometry: IndexingGeometry | None = None
    source_geometry: IndexingGeometry | None = None


@dataclass
class MosaicityEstimation:
    """Result of Mosflm ``mosaic estimate`` (iMosflm §5.6)."""

    mosaicity_deg: float
    scan_points: list[tuple[float, float]] = field(default_factory=list)
    message: str = ""


@dataclass
class IntegrationImageRange:
    """Contiguous Mosflm image numbers chosen for a trial integration run."""

    first_image: int
    last_image: int
    selection_method: str = "heuristic"
    segment_count: int = 0


@dataclass
class IntegrationResult:
    """Output from a short Mosflm integration (tutorial §9)."""

    mtz_path: Path
    image_range: IntegrationImageRange
    n_images: int
    message: str = ""
    reflections_measured: int | None = None


@dataclass
class IndexingResult:
    """Combined preselection + refined solutions from one indexing run."""

    spotfile: Path
    n_spots: int
    suggested_number: int | None
    solutions: list[IndexSolution]
    refined: list[IndexSolution]
    messages: list[str] = field(default_factory=list)
    spotlists: list[SpotList] = field(default_factory=list)
    frame_indices: list[int] = field(default_factory=list)
    geometry: IndexingGeometry | None = None
    uses_placeholder_template: bool = False
    estimated_mosaicity: float | None = None
    mosaicity_scan: list[tuple[float, float]] = field(default_factory=list)


def scattering_vector(
    x_mm: float,
    y_mm: float,
    *,
    beam_x_mm: float,
    beam_y_mm: float,
    distance_mm: float,
    wavelength_a: float,
) -> np.ndarray:
    """Lab-frame scattering vector ``(s1 - s0) / λ`` for a detector spot (Å⁻¹)."""
    delta_x = float(x_mm) - float(beam_x_mm)
    delta_y = float(y_mm) - float(beam_y_mm)
    z = float(distance_mm)
    norm = float(np.sqrt(delta_x * delta_x + delta_y * delta_y + z * z))
    if norm <= 0.0 or wavelength_a <= 0.0:
        return np.zeros(3, dtype=np.float64)
    s1 = np.array([delta_x / norm, delta_y / norm, z / norm], dtype=np.float64)
    s0 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return (s1 - s0) / float(wavelength_a)


def match_spots_to_solution(
    spots: Sequence[Spot],
    geometry: IndexingGeometry,
    solution: IndexSolution,
    *,
    hkldev: float = 0.3,
) -> list[bool]:
    """Return per-spot flags for spots indexed by *solution*'s A-matrix.

    Uses Mosflm's ``a_matrix`` with ``S ≈ Aᵀ h`` (so ``h = (Aᵀ)⁻¹ S``).
    A spot matches when the maximum fractional-index residual is ≤ *hkldev*
    and the nearest Miller index is not ``(0,0,0)``.
    """
    if not spots:
        return []
    if len(solution.matrix) != 9:
        return [False] * len(spots)
    matrix = np.asarray(solution.matrix, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-18:
        return [False] * len(spots)
    try:
        # S = A.T @ h  ⇒  h = inv(A.T) @ S
        a_t_inv = np.linalg.inv(matrix.T)
    except np.linalg.LinAlgError:
        return [False] * len(spots)

    matched: list[bool] = []
    for spot in spots:
        scat = scattering_vector(
            spot.x_mm,
            spot.y_mm,
            beam_x_mm=geometry.beam_x_mm,
            beam_y_mm=geometry.beam_y_mm,
            distance_mm=geometry.distance_mm,
            wavelength_a=geometry.wavelength_a,
        )
        hkl = a_t_inv @ scat
        nearest = np.rint(hkl)
        if np.allclose(nearest, 0.0):
            matched.append(False)
            continue
        residual = float(np.max(np.abs(hkl - nearest)))
        matched.append(residual <= float(hkldev))
    return matched


def spots_for_frame(
    result: IndexingResult,
    frame_index: int,
) -> list[Spot]:
    """Return observed spots associated with *frame_index*, if any."""
    if not result.spotlists:
        return []
    if result.frame_indices:
        for idx, spotlist in zip(result.frame_indices, result.spotlists):
            if int(idx) == int(frame_index):
                return list(spotlist.spots)
        return []
    if len(result.spotlists) == 1:
        return list(result.spotlists[0].spots)
    return []


def _text(node: ET.Element | None, path: str, default: str = "") -> str:
    if node is None:
        return default
    found = node.find(path)
    if found is None or found.text is None:
        return default
    return found.text.strip()


def _float(node: ET.Element | None, path: str, default: float = 0.0) -> float:
    text = _text(node, path, "")
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _int(node: ET.Element | None, path: str, default: int = 0) -> int:
    text = _text(node, path, "")
    if not text:
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def parse_lattice_character(node: ET.Element, kind: str) -> IndexSolution:
    """Parse one ``lattice_character`` node into an :class:`IndexSolution`."""
    lattice = _text(node, "lattice", "aP")
    if kind == "raw":
        a = _float(node, "cell/a")
        b = _float(node, "cell/b")
        c = _float(node, "cell/c")
        alpha = _float(node, "cell/alpha", 90.0)
        beta = _float(node, "cell/beta", 90.0)
        gamma = _float(node, "cell/gamma", 90.0)
    else:
        a = _float(node, "regularised_cell/reg_a")
        b = _float(node, "regularised_cell/reg_b")
        c = _float(node, "regularised_cell/reg_c")
        alpha = _float(node, "regularised_cell/reg_alpha", 90.0)
        beta = _float(node, "regularised_cell/reg_beta", 90.0)
        gamma = _float(node, "regularised_cell/reg_gamma", 90.0)
    matrix = tuple(
        _float(node, f"a_matrix/a{i}{j}")
        for i in (1, 2, 3)
        for j in (1, 2, 3)
    )
    return IndexSolution(
        number=_int(node, "number"),
        kind=kind,
        penalty=_float(node, "penalty"),
        lattice=lattice,
        spacegroup=SPACEGROUP_FOR_LATTICE.get(lattice, "P1"),
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        matrix=matrix,
    )


def parse_preselection_index_response(root: ET.Element) -> tuple[list[IndexSolution], int | None, str]:
    """Parse ``preselection_index_response`` XML."""
    status = _text(root, "status/code", "ok")
    message = _text(root, "status/message", "")
    suggested = _text(root, "suggested_solution/number", "")
    suggested_number = int(float(suggested)) if suggested else None
    solutions: list[IndexSolution] = []
    for node in root.findall(".//lattice_character"):
        solutions.append(parse_lattice_character(node, "raw"))
        solutions.append(parse_lattice_character(node, "reg"))
    return solutions, suggested_number, f"{status}: {message}".strip(": ")


def parse_refined_index_response(
    root: ET.Element,
    *,
    spacegroup: str | None = None,
) -> IndexSolution:
    """Parse ``refined_index_response`` XML into one refined solution."""
    lattice = _text(root, "lattice", "aP")
    sg_from_xml = _text(root, "spacegroup", "")
    # Refined responses often put cell at top level.
    a = _float(root, "a", _float(root, "cell/a"))
    b = _float(root, "b", _float(root, "cell/b"))
    c = _float(root, "c", _float(root, "cell/c"))
    alpha = _float(root, "alpha", _float(root, "cell/alpha", 90.0))
    beta = _float(root, "beta", _float(root, "cell/beta", 90.0))
    gamma = _float(root, "gamma", _float(root, "cell/gamma", 90.0))
    matrix = tuple(
        _float(root, f"a{i}{j}", _float(root, f"a_matrix/a{i}{j}"))
        for i in (1, 2, 3)
        for j in (1, 2, 3)
    )
    resolved_sg = (
        normalize_spacegroup(spacegroup or sg_from_xml)
        or default_spacegroup_for_lattice(lattice)
    )
    return IndexSolution(
        number=_int(root, "solution_number", _int(root, "number")),
        kind="ref",
        penalty=_float(root, "penalty"),
        lattice=lattice or _text(root, "lattice_type", "aP"),
        spacegroup=resolved_sg,
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        matrix=matrix,
        reflections_used=_int(root, "reflections_used") or None,
        spot_deviation_pos=_float(root, "spot_deviation_pos") or None,
        spot_deviation_phi=_float(root, "spot_deviation_phi") or None,
        beam_x=_float(root, "beam_x") or None,
        beam_y=_float(root, "beam_y") or None,
        volume=_float(root, "volume") or None,
    )


def build_findspots_commands(geometry: IndexingGeometry, settings: IndexingSettings, spot_path: Path) -> list[str]:
    """Build Mosflm commands equivalent to ``Mosflm::findspots``."""
    beam = (
        f"beam {geometry.beam_x_mm:.4f} {geometry.beam_y_mm:.4f}"
        if abs(geometry.two_theta) < 1e-6
        else f"beam swungout {geometry.beam_x_mm:.4f} {geometry.beam_y_mm:.4f}"
    )
    findspots_params = (
        f"threshold {settings.threshold:.2f} split 0.00 0.00 "
        f"minx 0.50 maxx 10.00 miny 0.50 maxy 10.00 "
        f"rmin 0.00 rmax 0.00 LOCAL AUTORESLN AUTORING NOICE"
    )
    return [
        f"directory {geometry.directory}",
        f"template {geometry.template}",
        beam,
        f"pixel {geometry.pixel_size_mm:.6f}",
        f"gain {geometry.gain:.4f}",
        f"distance {geometry.distance_mm:.4f}",
        f"nullpix {geometry.nullpix:g}",
        f"findspots {findspots_params}",
        (
            f"findspots find {geometry.image_number} "
            f"phi {geometry.phi_start:.4f} {geometry.phi_end:.4f} "
            f'file "{spot_path}"'
        ),
    ]


def ensure_indexable_template(
    geometry: IndexingGeometry,
    work_dir: Path,
) -> tuple[IndexingGeometry, bool]:
    """Return geometry pointing at a Mosflm-readable template.

    Native Eiger HDF5 masters often crash or fail header reads in
    ``ipmosflm``.  When *geometry* is an HDF5 master, write a tiny ADSC
    ``.img`` placeholder (same dimensions / header values) under
    *work_dir* and point Mosflm at that.  Spot coordinates still come
    from the spotfile.

    Returns
    -------
    tuple[IndexingGeometry, bool]
        Updated geometry and whether a placeholder image was substituted.
    """
    if not geometry.is_hdf5_master:
        return geometry, False
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    stem = f"qtmosflm_{geometry.image_width}x{geometry.image_height}"
    template = f"{stem}_####.img"
    image_path = work_dir / f"{stem}_0001.img"
    if not image_path.is_file():
        _write_dummy_adsc_image(image_path, geometry)
    return (
        replace(
            geometry,
            directory=work_dir,
            template=template,
            is_hdf5_master=False,
        ),
        True,
    )


def _write_dummy_adsc_image(path: Path, geometry: IndexingGeometry) -> None:
    """Write a minimal ADSC image Mosflm can open for header/template use."""
    try:
        from fabio.adscimage import adscimage
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "fabio is required to create a placeholder image for HDF5 indexing"
        ) from exc

    data = np.zeros((geometry.image_height, geometry.image_width), dtype=np.uint16)
    # Non-zero pixel so the file is recognised as real detector data.
    cx = min(max(int(geometry.beam_x_mm / geometry.pixel_size_mm), 0), geometry.image_width - 1)
    cy = min(max(int(geometry.beam_y_mm / geometry.pixel_size_mm), 0), geometry.image_height - 1)
    data[cy, cx] = 100
    img = adscimage(data=data)
    img.header.update(
        {
            "HEADER_BYTES": "512",
            "DIM": "2",
            "BYTE_ORDER": "little_endian",
            "TYPE": "unsigned_short",
            "SIZE1": str(geometry.image_width),
            "SIZE2": str(geometry.image_height),
            "PIXEL_SIZE": f"{geometry.pixel_size_mm:g}",
            "DISTANCE": f"{geometry.distance_mm:.4f}",
            "TWOTHETA": f"{geometry.two_theta:.4f}",
            "PHI": f"{geometry.phi_start:.4f}",
            "OSC_START": f"{geometry.phi_start:.4f}",
            "OSC_RANGE": f"{abs(geometry.phi_end - geometry.phi_start):.4f}",
            "WAVELENGTH": f"{geometry.wavelength_a:.6f}",
            "BEAM_CENTER_X": f"{geometry.beam_x_mm:.4f}",
            "BEAM_CENTER_Y": f"{geometry.beam_y_mm:.4f}",
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    img.write(str(path))
    logger.info("Wrote Mosflm placeholder image %s", path)


def build_index_commands(
    geometry: IndexingGeometry,
    settings: IndexingSettings,
    spotfilename: Path,
    *,
    solution: int = 0,
) -> list[str]:
    """Build Mosflm commands equivalent to ``Mosflm::index``."""
    beam = (
        f"beam {geometry.beam_x_mm:.4f} {geometry.beam_y_mm:.4f}"
        if abs(geometry.two_theta) < 1e-6
        else f"beam swungout {geometry.beam_x_mm:.4f} {geometry.beam_y_mm:.4f}"
    )
    sub = (
        f"dps threshold {settings.i_sig_i:g} hkldev {settings.hkldev_max:g} "
        f"nvect {settings.numvectors} nvref {settings.numvectors}"
    )
    if not settings.fix_distance_indexing:
        sub += " unfixdist"
    if settings.exclude_auto:
        sub += " exclude auto"
    # Prefer basename when the spotfile lives in the working directory.
    spot_arg = Path(spotfilename).name
    return [
        f"directory {geometry.directory}",
        f"template {geometry.template}",
        beam,
        f"distance {geometry.distance_mm:.4f}",
        f"wavelength {geometry.wavelength_a:.6f}",
        f"twotheta {geometry.two_theta:.4f}",
        f"pixel {geometry.pixel_size_mm:.6f}",
        (
            f"image {geometry.image_number} "
            f"phi {geometry.phi_start:.4f} {geometry.phi_end:.4f}"
        ),
        (
            f"autoindex {sub} solution {solution} "
            f"image {geometry.image_number} file \"{spot_arg}\""
        ),
    ]


def build_index2_commands(
    solution_number: int,
    settings: IndexingSettings,
    *,
    lattice: int = 1,
    sigma_cutoff: float | None = None,
) -> list[str]:
    """Build Mosflm commands equivalent to ``Mosflm::index2``."""
    unfix = "" if settings.fix_distance_indexing else " unfixdist"
    sigma = settings.sigma_cutoff if sigma_cutoff is None else float(sigma_cutoff)
    return [
        (
            f"autoindex dps solution {solution_number} "
            f"sdcutoff {sigma:g}{unfix} lattice {lattice}"
        )
    ]


def auto_integration_image_count(
    mosaicity_deg: float,
    oscillation_deg: float,
    total_images: int,
    *,
    minimum: int = 4,
) -> int:
    """Heuristic image count (``integrationwizard.tcl`` post-refinement formula)."""
    if total_images <= 0:
        return 0
    osc = float(oscillation_deg) if oscillation_deg > 0 else 0.15
    mosaic = float(mosaicity_deg) if mosaicity_deg > 0 else 0.05
    count = int(2.5 * mosaic / osc)
    if count < minimum:
        count = minimum
    if count >= 20:
        count = count // 2
    return min(count, int(total_images))


def parse_segment_setup_response(root: ET.Element) -> list[tuple[int, int]]:
    """Parse ``segment_setup_response`` into ``(start_image, n_images)`` segments."""
    segments: list[tuple[int, int]] = []
    for node in root.findall(".//segment"):
        start = _int(node, "start")
        span = _int(node, "range")
        if start > 0 and span > 0:
            segments.append((start, span))
    return segments


def integration_range_from_segments(
    segments: Sequence[tuple[int, int]],
    *,
    total_images: int,
) -> IntegrationImageRange:
    """Build one contiguous range from Mosflm segment list (first segment)."""
    if not segments:
        raise ValueError("No segments returned from Mosflm")
    start, span = segments[0]
    last = start + span - 1
    if last > total_images:
        last = total_images
    return IntegrationImageRange(
        first_image=int(start),
        last_image=int(last),
        selection_method="mosflm_segment",
        segment_count=len(segments),
    )


def heuristic_integration_range(
    *,
    total_images: int,
    mosaicity_deg: float,
    oscillation_deg: float,
    first_image: int = 1,
) -> IntegrationImageRange:
    """Pick a short integration wedge when Mosflm segment setup is unavailable."""
    count = auto_integration_image_count(
        mosaicity_deg,
        oscillation_deg,
        total_images,
    )
    first = max(1, int(first_image))
    last = min(first + count - 1, int(total_images))
    return IntegrationImageRange(
        first_image=first,
        last_image=last,
        selection_method="heuristic",
        segment_count=0,
    )


def build_matrix_command(matrix: Sequence[float]) -> str:
    """Format a Mosflm ``matrix`` line from nine A-matrix elements."""
    if len(matrix) != 9:
        raise ValueError("A-matrix must have nine elements")
    return "matrix " + " ".join(f"{float(v):.8g}" for v in matrix)


def build_segment_setup_commands(
    geometry: IndexingGeometry,
    solution: IndexSolution,
    *,
    spacegroup: str,
    mosaicity_deg: float,
    settings: IndexingSettings,
    first_image: int,
    last_image: int,
    genfile: str,
) -> list[str]:
    """Build Mosflm commands for ``getCellRefinementSegments`` / segment setup."""
    if len(solution.matrix) != 9:
        raise ValueError("Solution has no A-matrix for segment setup")
    beam = (
        f"beam {geometry.beam_x_mm:.4f} {geometry.beam_y_mm:.4f}"
        if abs(geometry.two_theta) < 1e-6
        else f"beam swungout {geometry.beam_x_mm:.4f} {geometry.beam_y_mm:.4f}"
    )
    sg = normalize_spacegroup(spacegroup)
    a, b, c, alpha, beta, gamma = solution.cell_tuple
    return [
        beam,
        f"distance {geometry.distance_mm:.4f}",
        f"wavelength {geometry.wavelength_a:.6f}",
        "resolution 50",
        f"twotheta {geometry.two_theta:.4f}",
        f"symmetry {sg}",
        f'genfile "{genfile}"',
        f"directory {geometry.directory}",
        f"template {geometry.template}",
        (
            f"image {first_image} "
            f"phi {geometry.phi_start:.4f} {geometry.phi_end:.4f}"
        ),
        "load",
        build_matrix_command(solution.matrix),
        f"mosaic {mosaicity_deg:g} blocksize {settings.mosaic_block_size:g}",
        f"segment {first_image} {last_image}",
    ]


def build_integrate_commands(
    geometry: IndexingGeometry,
    solution: IndexSolution,
    *,
    spacegroup: str,
    mosaicity_deg: float,
    settings: IndexingSettings,
    image_range: IntegrationImageRange,
    phi_start: float,
    oscillation_deg: float,
    mtz_path: Path,
    genfile: str,
) -> list[str]:
    """Build a minimal Mosflm integration script (``Mosflm::integrate``)."""
    if len(solution.matrix) != 9:
        raise ValueError("Solution has no A-matrix for integration")
    beam_x = solution.beam_x if solution.beam_x is not None else geometry.beam_x_mm
    beam_y = solution.beam_y if solution.beam_y is not None else geometry.beam_y_mm
    beam = (
        f"beam {beam_x:.4f} {beam_y:.4f}"
        if abs(geometry.two_theta) < 1e-6
        else f"beam swungout {beam_x:.4f} {beam_y:.4f}"
    )
    sg = normalize_spacegroup(spacegroup)
    a, b, c, alpha, beta, gamma = solution.cell_tuple
    mtz_name = Path(mtz_path).name
    first = image_range.first_image
    last = image_range.last_image
    osc = max(float(oscillation_deg), 1e-4)
    return [
        f"directory {geometry.directory}",
        f"template {geometry.template}",
        beam,
        f"distance {geometry.distance_mm:.4f}",
        f"wavelength {geometry.wavelength_a:.6f}",
        f"twotheta {geometry.two_theta:.4f}",
        f"pixel {geometry.pixel_size_mm:.6f}",
        f"gain {geometry.gain:.4f}",
        build_matrix_command(solution.matrix),
        f"cell {a:.4f} {b:.4f} {c:.4f} {alpha:.2f} {beta:.2f} {gamma:.2f}",
        f"symmetry {sg}",
        (
            f"mosaicity {mosaicity_deg:g} "
            f"blocksize {settings.mosaic_block_size:g}"
        ),
        f'mtzdirectory "{Path(mtz_path).parent}"',
        f"hklout {mtz_name} nomultiple",
        f'genfile "{genfile}"',
        "resolution exclude none",
        f"nullpix {geometry.nullpix:g}",
        "norefine off",
        (
            f"process {first} {last} start {phi_start:.4f} "
            f"angle {osc:.4f} add 0"
        ),
    ]


def build_mosaic_estimate_commands(
    geometry: IndexingGeometry,
    solution: IndexSolution,
    *,
    spacegroup: str,
    image_number: int,
    phi_start: float,
    phi_end: float,
    phi_scan_end: float,
    settings: IndexingSettings,
) -> list[str]:
    """Build Mosflm lines for ``Mosflm::estimateMosaicity`` (mosaic estimate)."""
    if len(solution.matrix) != 9:
        raise ValueError("Selected solution has no A-matrix for mosaicity estimation")
    beam = (
        f"beam {geometry.beam_x_mm:.4f} {geometry.beam_y_mm:.4f}"
        if abs(geometry.two_theta) < 1e-6
        else f"beam swungout {geometry.beam_x_mm:.4f} {geometry.beam_y_mm:.4f}"
    )
    a, b, c, alpha, beta, gamma = solution.cell_tuple
    sg = normalize_spacegroup(spacegroup)
    return [
        f"directory {geometry.directory}",
        f"template {geometry.template}",
        beam,
        f"distance {geometry.distance_mm:.4f}",
        f"wavelength {geometry.wavelength_a:.6f}",
        f"twotheta {geometry.two_theta:.4f}",
        f"pixel {geometry.pixel_size_mm:.6f}",
        build_matrix_command(solution.matrix),
        f"cell {a:.4f} {b:.4f} {c:.4f} {alpha:.2f} {beta:.2f} {gamma:.2f}",
        "misset 0 0 0",
        f"symmetry {sg}",
        (
            f"image {image_number} "
            f"phi {phi_start:.4f} {phi_end:.4f}"
        ),
        (
            f"mosaicity 0.05 blocksize {settings.mosaic_block_size:g}"
        ),
        (
            f"mosaic estimate {image_number} "
            f"phi {phi_start:.4f} to {phi_scan_end:.4f}"
        ),
    ]


def parse_mosaicity_feedback(root: ET.Element) -> tuple[float, float]:
    """Parse one ``mosaicity_estimation_response`` trial point."""
    mosaic = _float(root, "mosaicity_trial_value")
    intensity = _float(root, "total_intensity")
    return mosaic, intensity


def parse_mosaicity_estimation(root: ET.Element) -> MosaicityEstimation:
    """Parse final ``mosaicity_response`` XML."""
    status = _text(root, "status/code", "ok")
    message = _text(root, "status/message", "")
    if status.lower() == "error":
        raise RuntimeError(message or "Mosflm mosaicity estimation failed")
    mosaic = _float(root, "mosaicity_value", _float(root, "mosaicity"))
    return MosaicityEstimation(mosaicity_deg=mosaic, message=message.strip(": "))


def build_cell_symmetry_validate_commands(
    cell: Sequence[float],
    spacegroup: str,
) -> list[str]:
    """Build ``cell`` / ``symmetry`` / ``validate`` (Session::validateCellAndSpacegroup)."""
    if len(cell) != 6:
        raise ValueError("cell must have six elements (a, b, c, alpha, beta, gamma)")
    a, b, c, alpha, beta, gamma = (float(x) for x in cell)
    sg = normalize_spacegroup(spacegroup)
    if not sg:
        raise ValueError("space group must not be empty")
    return [
        f"cell {a:.4f} {b:.4f} {c:.4f} {alpha:.2f} {beta:.2f} {gamma:.2f}",
        f"symmetry {sg}",
        "validate",
    ]


def preselection_solution(
    result: IndexingResult,
    solution_number: int,
    *,
    prefer: str = "reg",
) -> IndexSolution | None:
    """Return the raw or regularised preselection row for *solution_number*."""
    kinds: list[str] = [prefer]
    for kind in ("reg", "raw"):
        if kind not in kinds:
            kinds.append(kind)
    for kind in kinds:
        for sol in result.solutions:
            if sol.number == solution_number and sol.kind == kind:
                return sol
    return None


def _first_xml(messages: Sequence[MosflmMessage], doctype: str) -> ET.Element:
    for msg in messages:
        if msg.kind == "xml" and msg.doctype == doctype and msg.root is not None:
            return msg.root
    raise RuntimeError(f"Mosflm did not return {doctype}")


class IndexingEngine:
    """High-level indexing orchestrator using ``ipmosflm``."""

    def __init__(
        self,
        *,
        work_dir: Path | None = None,
        settings: IndexingSettings | None = None,
        executable: Path | None = None,
    ) -> None:
        self.settings = settings or IndexingSettings()
        self.work_dir = Path(work_dir or (Path.home() / ".qtmosflm" / "work"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.executable = executable or find_ipmosflm()
        if self.executable is None:
            raise FileNotFoundError(
                "ipmosflm not found. Install CCP4 or set MOSFLM_EXEC / ELMO_IPMOSFLM."
            )

    def find_spots_from_frames(
        self,
        frames: Sequence[np.ndarray],
        geometry: IndexingGeometry,
        *,
        phis: Sequence[tuple[float, float]] | None = None,
    ) -> tuple[Path, int, list[SpotList]]:
        """Peak-find in numpy frames and write a Mosflm merged spotfile."""
        if not frames:
            raise ValueError("No frames provided for spot finding")
        spotlists: list[SpotList] = []
        total = 0
        for i, frame in enumerate(frames):
            if phis is not None and i < len(phis):
                phi0, phi1 = phis[i]
            else:
                width = geometry.phi_end - geometry.phi_start
                phi0 = geometry.phi_start + i * width
                phi1 = phi0 + width
            spots = find_spots_in_frame(
                frame,
                pixel_size_mm=geometry.pixel_size_mm,
                threshold_sigma=self.settings.spot_threshold_sigma,
                max_spots=self.settings.max_spots_per_frame,
            )
            spotlists.append(SpotList(spots=spots, phi_start=phi0, phi_end=phi1))
            total += len(spots)
            logger.info("Frame %d: found %d spots (phi %.3f–%.3f)", i, len(spots), phi0, phi1)
        if total < 10:
            raise RuntimeError(
                f"Only {total} spots found — need more spots for Mosflm indexing."
            )
        spot_geom = SpotFileGeometry(
            image_width=geometry.image_width,
            image_height=geometry.image_height,
            pixel_size_mm=geometry.pixel_size_mm,
            beam_x_mm=geometry.beam_x_mm,
            beam_y_mm=geometry.beam_y_mm,
            yscale=geometry.yscale,
        )
        spotfile = new_temp_spotfilename(self.work_dir)
        write_merged_spotfile(spotfile, spot_geom, spotlists)
        logger.info("Wrote spotfile %s (%d spots)", spotfile, total)
        return spotfile, total, spotlists

    def find_spots_step(
        self,
        frames: Sequence[np.ndarray],
        geometry: IndexingGeometry,
        *,
        phis: Sequence[tuple[float, float]] | None = None,
        frame_indices: Sequence[int] | None = None,
    ) -> SpotFindingResult:
        """Run spot finding only (tutorial §5.1) and return a spotfile."""
        spotfile, n_spots, spotlists = self.find_spots_from_frames(
            frames, geometry, phis=phis
        )
        mosflm_geometry, _ = ensure_indexable_template(geometry, self.work_dir)
        indices = (
            [int(i) for i in frame_indices]
            if frame_indices is not None
            else list(range(len(spotlists)))
        )
        return SpotFindingResult(
            spotfile=spotfile,
            n_spots=n_spots,
            spotlists=list(spotlists),
            frame_indices=indices,
            geometry=mosflm_geometry,
            source_geometry=geometry,
        )

    def index_spotfile(
        self,
        spotfinding: SpotFindingResult,
        *,
        refine: bool = False,
    ) -> IndexingResult:
        """DPS autoindexing from an existing spotfile (after spot finding)."""
        geometry = spotfinding.geometry or spotfinding.source_geometry
        if geometry is None:
            raise ValueError("Spot finding result has no geometry")
        result = self.autoindex(geometry, spotfinding.spotfile, refine=refine)
        result.spotlists = list(spotfinding.spotlists)
        result.n_spots = spotfinding.n_spots
        result.frame_indices = list(spotfinding.frame_indices)
        if result.geometry is None:
            result.geometry = geometry
        return result

    def estimate_mosaicity(
        self,
        geometry: IndexingGeometry,
        solution: IndexSolution,
        *,
        spacegroup: str,
        image_number: int | None = None,
        phi_start: float | None = None,
        phi_end: float | None = None,
        phi_scan_end: float | None = None,
    ) -> MosaicityEstimation:
        """Estimate mosaic spread for *solution* (``mosaic estimate``, tutorial §5.6)."""
        geometry, used_placeholder = ensure_indexable_template(geometry, self.work_dir)
        if used_placeholder:
            raise RuntimeError(
                "Mosaicity estimation needs Mosflm to read detector images; "
                "it is not supported with the HDF5 placeholder template in DiffractionLab."
            )
        img_num = int(image_number if image_number is not None else geometry.image_number)
        p0 = float(phi_start if phi_start is not None else geometry.phi_start)
        p1 = float(phi_end if phi_end is not None else geometry.phi_end)
        if phi_scan_end is not None:
            p_scan = float(phi_scan_end)
        else:
            width = max(p1 - p0, 0.01)
            p_scan = p1 + width

        with MosflmClient(self.executable, self.work_dir) as client:
            stamp = time_stamp()
            client.send(f"GENFILE mosflm_{stamp}.gen")
            client.wait_for(doctypes=(), also_done=True, timeout_s=10.0)
            client.send(f"NEWMAT mosflm_{stamp}.mat")
            client.wait_for(doctypes=(), also_done=True, timeout_s=10.0)
            client.send_many(
                build_mosaic_estimate_commands(
                    geometry,
                    solution,
                    spacegroup=spacegroup,
                    image_number=img_num,
                    phi_start=p0,
                    phi_end=p1,
                    phi_scan_end=p_scan,
                    settings=self.settings,
                )
            )
            return _wait_for_mosaicity_estimation(client, timeout_s=180.0)

    def auto_select_integration_range(
        self,
        geometry: IndexingGeometry,
        solution: IndexSolution,
        *,
        spacegroup: str,
        mosaicity_deg: float,
        total_images: int,
        first_image: int = 1,
    ) -> IntegrationImageRange:
        """Ask Mosflm for segments, or fall back to the iMosflm heuristic."""
        if total_images <= 0:
            raise ValueError("No images in dataset")
        if len(solution.matrix) != 9:
            raise ValueError("Indexing solution has no A-matrix")
        osc = max(abs(geometry.phi_end - geometry.phi_start), 0.01)
        try:
            with MosflmClient(self.executable, self.work_dir) as client:
                stamp = time_stamp()
                gen = f"mosflm_{stamp}.gen"
                client.send(f"GENFILE {gen}")
                client.wait_for(doctypes=(), also_done=True, timeout_s=10.0)
                client.send(f"NEWMAT mosflm_{stamp}.mat")
                client.wait_for(doctypes=(), also_done=True, timeout_s=10.0)
                setup = build_segment_setup_commands(
                    geometry,
                    solution,
                    spacegroup=spacegroup,
                    mosaicity_deg=mosaicity_deg,
                    settings=self.settings,
                    first_image=1,
                    last_image=int(total_images),
                    genfile=gen,
                )
                load_idx = setup.index("load") + 1
                client.send_many(setup[:load_idx])
                client.go(expect=(), timeout_s=60.0)
                client.send_many(setup[load_idx:])
                segments = _wait_for_segment_setup(client, timeout_s=120.0)
                return integration_range_from_segments(
                    segments,
                    total_images=total_images,
                )
        except Exception as ex:
            logger.warning("Mosflm segment setup failed (%s); using heuristic", ex)
            return heuristic_integration_range(
                total_images=total_images,
                mosaicity_deg=mosaicity_deg,
                oscillation_deg=osc,
                first_image=first_image,
            )

    def integrate_images(
        self,
        geometry: IndexingGeometry,
        solution: IndexSolution,
        image_range: IntegrationImageRange,
        *,
        spacegroup: str,
        mosaicity_deg: float,
        phi_start: float,
        oscillation_deg: float,
        mtz_path: Path | None = None,
    ) -> IntegrationResult:
        """Integrate *image_range* and write an MTZ under the work directory."""
        if len(solution.matrix) != 9:
            raise ValueError("Indexing solution has no A-matrix")
        if geometry.is_hdf5_master:
            raise RuntimeError(
                "Mosflm cannot integrate HDF5 / NeXus masters from DiffractionLab "
                "(ipmosflm crashes when PROCESS opens the master file). "
                "Use a native Mosflm image template (.img / .cbf / etc.), "
                "or run integration in iMosflm on this dataset."
            )
        mtz_path = Path(mtz_path or (self.work_dir / "qtmosflm_integrate.mtz"))
        mtz_path.parent.mkdir(parents=True, exist_ok=True)
        with MosflmClient(self.executable, self.work_dir) as client:
            stamp = time_stamp()
            gen = f"mosflm_{stamp}.gen"
            client.send(f"GENFILE {gen}")
            client.wait_for(doctypes=(), also_done=True, timeout_s=10.0)
            client.send(f"NEWMAT mosflm_{stamp}.mat")
            client.wait_for(doctypes=(), also_done=True, timeout_s=10.0)
            client.send_many(
                build_integrate_commands(
                    geometry,
                    solution,
                    spacegroup=spacegroup,
                    mosaicity_deg=mosaicity_deg,
                    settings=self.settings,
                    image_range=image_range,
                    phi_start=phi_start,
                    oscillation_deg=oscillation_deg,
                    mtz_path=mtz_path,
                    genfile=gen,
                )
            )
            message = _wait_for_integration(client, timeout_s=600.0)
        n_images = image_range.last_image - image_range.first_image + 1
        return IntegrationResult(
            mtz_path=mtz_path,
            image_range=image_range,
            n_images=n_images,
            message=message,
        )

    def autoindex(
        self,
        geometry: IndexingGeometry,
        spotfile: Path,
        *,
        refine: bool = True,
    ) -> IndexingResult:
        """Run Mosflm DPS autoindexing (+ optional refinement of low-penalty solutions)."""
        messages_log: list[str] = []
        geometry, used_placeholder = ensure_indexable_template(geometry, self.work_dir)
        if used_placeholder:
            # Refinement re-opens the template image; a blank placeholder hangs
            # Mosflm.  Preselection solutions are still valid from the spotfile.
            if refine:
                messages_log.append(
                    "Skipping Mosflm solution refinement (HDF5 placeholder template)."
                )
            refine = False
        spotfile = Path(spotfile)
        # Spotfile must be visible from Mosflm's cwd (work_dir).
        if spotfile.parent.resolve() != self.work_dir.resolve():
            target = self.work_dir / spotfile.name
            if spotfile.resolve() != target.resolve():
                target.write_bytes(spotfile.read_bytes())
            spotfile = target

        with MosflmClient(self.executable, self.work_dir) as client:
            stamp = time_stamp()
            client.send(f"GENFILE mosflm_{stamp}.gen")
            client.wait_for(doctypes=(), also_done=True, timeout_s=10.0)
            client.send(f"NEWMAT mosflm_{stamp}.mat")
            client.wait_for(doctypes=(), also_done=True, timeout_s=10.0)

            client.send_many(build_index_commands(geometry, self.settings, spotfile))
            pre_msgs = client.go(
                expect=("preselection_index_response",),
                timeout_s=180.0,
            )
            pre_root = _first_xml(pre_msgs, "preselection_index_response")
            solutions, suggested, status = parse_preselection_index_response(pre_root)
            messages_log.append(status)
            if status.lower().startswith("error"):
                raise RuntimeError(f"Mosflm indexing failed: {status}")

            refined: list[IndexSolution] = []
            if refine and solutions:
                # Prefer regularised solutions with penalty <= cutoff (TCL refineAll).
                candidates = [
                    sol
                    for sol in solutions
                    if sol.kind == "reg" and sol.penalty <= self.settings.max_penalty_to_refine
                ]
                if not candidates and suggested is not None:
                    candidates = [
                        sol
                        for sol in solutions
                        if sol.kind == "reg" and sol.number == suggested
                    ]
                # Deduplicate by solution number; prefer suggested, then low penalty.
                seen: set[int] = set()
                unique: list[IndexSolution] = []
                ordered = sorted(
                    candidates,
                    key=lambda s: (
                        0 if suggested is not None and s.number == suggested else 1,
                        s.penalty,
                        s.number,
                    ),
                )
                for sol in ordered:
                    if sol.number in seen:
                        continue
                    seen.add(sol.number)
                    unique.append(sol)
                # Cap refinements for interactivity (Mosflm can hang on bad lattices).
                for sol in unique[:4]:
                    try:
                        client.send_many(
                            build_index2_commands(sol.number, self.settings)
                        )
                        ref_msgs = client.go(
                            expect=("refined_index_response",),
                            timeout_s=45.0,
                        )
                        ref_root = _first_xml(ref_msgs, "refined_index_response")
                        refined_sol = parse_refined_index_response(ref_root)
                        if not refined_sol.lattice:
                            refined_sol.lattice = sol.lattice
                            refined_sol.spacegroup = sol.spacegroup
                        if refined_sol.penalty <= 0 and sol.penalty:
                            refined_sol.penalty = sol.penalty
                        refined.append(refined_sol)
                    except Exception as ex:
                        logger.warning("Refine solution %s failed: %s", sol.number, ex)
                        messages_log.append(f"refine {sol.number}: {ex}")
                        # Further refine attempts are unreliable after a hang/crash.
                        break

        n_spots = count_spots_in_spotfile(spotfile)
        return IndexingResult(
            spotfile=spotfile,
            n_spots=n_spots,
            suggested_number=suggested,
            solutions=solutions,
            refined=refined,
            messages=messages_log,
            geometry=geometry,
            uses_placeholder_template=used_placeholder,
        )

    def refine_index_solution(
        self,
        geometry: IndexingGeometry,
        spotfile: Path,
        solution_number: int,
        cell: Sequence[float],
        spacegroup: str,
        *,
        lattice_number: int = 1,
        sigma_cutoff: float | None = None,
        replay_preselection: bool = True,
    ) -> IndexSolution:
        """Refine one DPS solution with a chosen cell and space group (``index2``).

        Mirrors ``Latticetab::refine`` plus ``Session::validateCellAndSpacegroup``
        and the space-group combo in iMosflm's index wizard.  Mosflm must still
        have indexing context, so by default this replays DPS preselection in the
        same socket session before sending ``cell``, ``symmetry``, ``validate``,
        and ``autoindex dps solution …``.
        """
        geometry, used_placeholder = ensure_indexable_template(geometry, self.work_dir)
        if used_placeholder:
            raise RuntimeError(
                "Mosflm cannot refine indexing solutions when using the HDF5 "
                "placeholder image template (refinement re-searches detector "
                "images). Index from a native Mosflm image template (.img / "
                ".cbf / etc.) or use iMosflm for HDF5 refinement."
            )
        spotfile = Path(spotfile)
        if spotfile.parent.resolve() != self.work_dir.resolve():
            target = self.work_dir / spotfile.name
            if spotfile.resolve() != target.resolve():
                target.write_bytes(spotfile.read_bytes())
            spotfile = target

        sg = normalize_spacegroup(spacegroup)
        if not sg:
            raise ValueError("space group must not be empty")

        settings = self.settings
        if sigma_cutoff is not None:
            settings = replace(self.settings, sigma_cutoff=float(sigma_cutoff))

        with MosflmClient(self.executable, self.work_dir) as client:
            stamp = time_stamp()
            client.send(f"GENFILE mosflm_{stamp}.gen")
            client.wait_for(doctypes=(), also_done=True, timeout_s=10.0)
            client.send(f"NEWMAT mosflm_{stamp}.mat")
            client.wait_for(doctypes=(), also_done=True, timeout_s=10.0)

            if replay_preselection:
                client.send_many(
                    build_index_commands(geometry, settings, spotfile)
                )
                pre_msgs = client.go(
                    expect=("preselection_index_response",),
                    timeout_s=180.0,
                )
                pre_root = _first_xml(pre_msgs, "preselection_index_response")
                _, _, status = parse_preselection_index_response(pre_root)
                if status.lower().startswith("error"):
                    raise RuntimeError(f"Mosflm indexing failed: {status}")

            client.send_many(build_cell_symmetry_validate_commands(cell, sg))
            client.send_many(
                build_index2_commands(
                    solution_number,
                    settings,
                    lattice=lattice_number,
                    sigma_cutoff=settings.sigma_cutoff,
                )
            )
            ref_root = _wait_for_refined_index_response(client, timeout_s=120.0)
            return parse_refined_index_response(ref_root, spacegroup=sg)

    def index_frames(
        self,
        frames: Sequence[np.ndarray],
        geometry: IndexingGeometry,
        *,
        phis: Sequence[tuple[float, float]] | None = None,
        frame_indices: Sequence[int] | None = None,
        refine: bool = True,
    ) -> IndexingResult:
        """Find spots in *frames* then run Mosflm autoindexing."""
        spotfile, n_spots, spotlists = self.find_spots_from_frames(
            frames, geometry, phis=phis
        )
        result = self.autoindex(geometry, spotfile, refine=refine)
        result.spotlists = list(spotlists)
        result.n_spots = n_spots
        if frame_indices is not None:
            result.frame_indices = [int(i) for i in frame_indices]
        else:
            result.frame_indices = list(range(len(spotlists)))
        if result.geometry is None:
            result.geometry = geometry
        return result


def count_spots_in_spotfile(path: Path) -> int:
    """Count non-sentinel spot lines in a Mosflm spotfile."""
    count = 0
    with Path(path).open("r", encoding="ascii", errors="replace") as handle:
        # Skip 3-line header.
        for _ in range(3):
            handle.readline()
        for line in handle:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
            except ValueError:
                continue
            if abs(x + 99.0) < 1e-3 or abs(x + 999.0) < 1e-3:
                continue
            # Trailer line starts with image width (integer-ish, few fields).
            if len(parts) <= 4:
                continue
            count += 1
    return count


def time_stamp() -> str:
    import time

    return time.strftime("%Y%m%d_%H%M%S")


def _wait_for_refined_index_response(
    client: MosflmClient,
    *,
    timeout_s: float = 120.0,
) -> ET.Element:
    """Send ``go`` and collect XML until ``refined_index_response`` or failure."""
    import time

    client.send("go")
    deadline = time.time() + timeout_s
    saw_spot_search = False
    validate_error = ""
    while time.time() < deadline:
        remaining = max(0.05, deadline - time.time())
        msg = client.recv_message(timeout_s=remaining)
        if msg is None:
            break
        if msg.kind == "xml" and msg.root is not None:
            if msg.doctype == "refined_index_response":
                status = _text(msg.root, "status/code", "ok")
                message = _text(msg.root, "status/message", "")
                if status.lower() == "error":
                    raise RuntimeError(message or "Mosflm refinement failed")
                return msg.root
            if msg.doctype == "interface_input_response":
                status = _text(msg.root, "status/code", "ok")
                message = _text(msg.root, "status/message", "")
                if status.lower() == "error" and message:
                    validate_error = message
            if msg.doctype == "spot_search_response":
                saw_spot_search = True
            if msg.doctype == "fatal_condition_response":
                raise RuntimeError(
                    _text(msg.root, "status/message", "Mosflm fatal error")
                )
        if msg.kind == "done":
            continue
    if validate_error:
        raise RuntimeError(validate_error)
    if saw_spot_search:
        raise RuntimeError(
            "Mosflm did not return a refined indexing solution (spot search "
            "completed without refinement). This often happens with HDF5 "
            "masters when Mosflm cannot read the image template."
        )
    raise TimeoutError("Timed out waiting for Mosflm refined_index_response")


def _wait_for_mosaicity_estimation(
    client: MosflmClient,
    *,
    timeout_s: float = 180.0,
) -> MosaicityEstimation:
    """Send ``go`` and collect mosaicity trial/final XML responses."""
    import time

    client.send("go")
    deadline = time.time() + timeout_s
    scan: list[tuple[float, float]] = []
    while time.time() < deadline:
        remaining = max(0.05, deadline - time.time())
        msg = client.recv_message(timeout_s=remaining)
        if msg is None:
            break
        if msg.kind == "xml" and msg.root is not None:
            if msg.doctype == "mosaicity_estimation_response":
                scan.append(parse_mosaicity_feedback(msg.root))
            elif msg.doctype == "mosaicity_response":
                final = parse_mosaicity_estimation(msg.root)
                final.scan_points = scan
                return final
            elif msg.doctype == "fatal_condition_response":
                raise RuntimeError(
                    _text(msg.root, "status/message", "Mosflm fatal error")
                )
        if msg.kind == "done":
            continue
    raise TimeoutError(
        "Timed out waiting for Mosflm mosaicity_response "
        f"(collected {len(scan)} trial point(s))"
    )


def _wait_for_segment_setup(
    client: MosflmClient,
    *,
    timeout_s: float = 120.0,
) -> list[tuple[int, int]]:
    """Collect ``segment_setup_response`` segments."""
    import time

    client.send("go")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        remaining = max(0.05, deadline - time.time())
        msg = client.recv_message(timeout_s=remaining)
        if msg is None:
            break
        if msg.kind == "xml" and msg.root is not None:
            if msg.doctype == "segment_setup_response":
                status = _text(msg.root, "status/code", "ok")
                message = _text(msg.root, "status/message", "")
                if status.lower() == "error":
                    raise RuntimeError(message or "Segment setup failed")
                segments = parse_segment_setup_response(msg.root)
                if segments:
                    return segments
                raise RuntimeError(message or "Segment setup returned no segments")
        if msg.kind == "done":
            continue
    raise TimeoutError("Timed out waiting for Mosflm segment_setup_response")


def _wait_for_integration(
    client: MosflmClient,
    *,
    timeout_s: float = 600.0,
) -> str:
    """Run ``go`` and wait until Mosflm finishes the integration block."""
    import time

    client.send("go")
    deadline = time.time() + timeout_s
    last_message = ""
    saw_integration = False
    seen: list[str] = []
    while time.time() < deadline:
        remaining = max(0.05, deadline - time.time())
        msg = client.recv_message(timeout_s=remaining)
        if msg is None:
            exit_code = None
            if client._proc is not None:
                exit_code = client._proc.poll()
            detail = f" (exit code {exit_code})" if exit_code is not None else ""
            raise RuntimeError(
                "Mosflm closed the connection during integration"
                f"{detail}. Seen: {seen or ['(nothing)']}. "
                "HDF5 masters usually cause this; use a native image template "
                "or iMosflm for integration."
            )
        seen.append(msg.doctype or msg.kind)
        if msg.kind == "xml" and msg.root is not None:
            if msg.doctype == "integration_response":
                saw_integration = True
                status = _text(msg.root, "status/code", "ok")
                last_message = _text(msg.root, "status/message", "")
                if status.lower() == "error":
                    raise RuntimeError(last_message or "Integration failed")
                # Mosflm expects continue between blocks; send for single wedge.
                try:
                    client.send("continue")
                except OSError:
                    pass
            elif msg.doctype == "processing_task_end":
                return last_message or "Integration finished"
            elif msg.doctype == "fatal_condition_response":
                raise RuntimeError(
                    _text(msg.root, "status/message", "Mosflm fatal error")
                )
        if msg.kind == "done":
            continue
    if saw_integration:
        return last_message or "Integration finished (timed out waiting for end)"
    raise TimeoutError(
        f"Timed out waiting for Mosflm integration to finish after {timeout_s:g}s. "
        f"Seen: {seen or ['(nothing)']}"
    )
