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


def parse_refined_index_response(root: ET.Element) -> IndexSolution:
    """Parse ``refined_index_response`` XML into one refined solution."""
    lattice = _text(root, "lattice", "aP")
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
    return IndexSolution(
        number=_int(root, "solution_number", _int(root, "number")),
        kind="ref",
        penalty=_float(root, "penalty"),
        lattice=lattice or _text(root, "lattice_type", "aP"),
        spacegroup=SPACEGROUP_FOR_LATTICE.get(lattice, "P1"),
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
) -> list[str]:
    """Build Mosflm commands equivalent to ``Mosflm::index2``."""
    unfix = "" if settings.fix_distance_indexing else " unfixdist"
    return [
        (
            f"autoindex dps solution {solution_number} "
            f"sdcutoff {settings.sigma_cutoff:g}{unfix} lattice {lattice}"
        )
    ]


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
        )

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
