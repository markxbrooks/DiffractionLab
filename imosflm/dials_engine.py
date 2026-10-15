"""DIALS processing pipeline for datasets Mosflm cannot open (e.g. Eiger HDF5).

Mirrors the DUI tutorial flow:
``import → find_spots → index → refine → integrate → export``
(https://dials.github.io/documentation/tutorials/processing_in_detail_betalactamase_dui.html).
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)


def find_dials_bin(name: str = "dials.import") -> Path | None:
    """Locate a ``dials.*`` executable (CCP4 or PATH)."""
    for key in ("DIALS_DIR", "CCP4"):
        root = os.environ.get(key)
        if not root:
            continue
        candidate = Path(root) / "bin" / name
        if candidate.is_file():
            return candidate.resolve()
    which = shutil.which(name)
    if which:
        return Path(which).resolve()
    for root in (
        Path("/Applications/ccp4-9"),
        Path("/Applications/ccp4-8"),
        Path(os.environ.get("CBIN", "")).parent if os.environ.get("CBIN") else None,
    ):
        if root is None:
            continue
        candidate = root / "bin" / name
        try:
            if candidate.is_file():
                return candidate.resolve()
        except OSError:
            continue
    return None


def find_dials_prefix() -> Path | None:
    """Return the directory containing ``dials.import``, if any."""
    exe = find_dials_bin("dials.import")
    return exe.parent if exe is not None else None


@dataclass
class DialsImageRange:
    """1-based inclusive DialS scan range (image numbers)."""

    first_image: int
    last_image: int

    @property
    def n_images(self) -> int:
        return max(0, self.last_image - self.first_image + 1)

    def as_scan_range(self) -> str:
        return f"{self.first_image},{self.last_image}"


@dataclass
class DialsCrystalSolution:
    """Crystal model extracted from a DIALS ``.expt`` file."""

    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    spacegroup: str
    lattice: str = ""
    n_indexed: int | None = None
    n_unindexed: int | None = None
    rmsd_x: float | None = None
    rmsd_y: float | None = None
    rmsd_phi: float | None = None

    @property
    def cell_tuple(self) -> tuple[float, float, float, float, float, float]:
        return (self.a, self.b, self.c, self.alpha, self.beta, self.gamma)


@dataclass
class DialsSpotFindingResult:
    """Result of ``dials.import`` + ``dials.find_spots``."""

    work_dir: Path
    imported_expt: Path
    strong_refl: Path
    n_spots: int
    image_range: DialsImageRange
    log: str = ""
    #: Optional ``.npz`` with arrays ``x``, ``y``, ``z`` (pixel centroids).
    centroids_npz: Path | None = None


@dataclass
class DialsIndexingResult:
    """Result of ``dials.index`` (+ optional ``dials.refine``)."""

    work_dir: Path
    experiments: Path
    reflections: Path
    solution: DialsCrystalSolution
    refined: bool = False
    log: str = ""
    #: Optional ``.npz`` with predicted ``x``, ``y``, ``z`` (``xyzcal.px``).
    predictions_npz: Path | None = None


@dataclass
class DialsIntegrationResult:
    """Result of ``dials.integrate`` + ``dials.export``."""

    work_dir: Path
    experiments: Path
    reflections: Path
    mtz_path: Path
    image_range: DialsImageRange
    n_reflections: int | None = None
    log: str = ""


@dataclass
class DialsSettings:
    """User-facing DIALS defaults for DiffractionLab."""

    nproc: int = 4
    min_spot_size: int = 3
    gain: float | None = None
    d_min: float | None = None
    space_group: str | None = None
    unit_cell: str | None = None  # "a,b,c,alpha,beta,gamma"


def auto_dials_image_range(
    total_images: int,
    *,
    oscillation_deg: float = 0.15,
    mosaicity_deg: float = 0.2,
    first_image: int = 1,
    minimum: int = 20,
    maximum: int = 90,
) -> DialsImageRange:
    """Pick a short wedge for interactive DIALS indexing / trial integration.

    Uses a similar ``2.5 × mosaicity / oscillation`` heuristic as iMosflm,
    then clamps to *minimum*…*maximum* images for interactive runs.
    """
    if total_images <= 0:
        raise ValueError("No images in dataset")
    osc = float(oscillation_deg) if oscillation_deg > 0 else 0.15
    mosaic = float(mosaicity_deg) if mosaicity_deg > 0 else 0.2
    count = int(2.5 * mosaic / osc)
    if count < minimum:
        count = minimum
    if count > maximum:
        count = maximum
    count = min(count, int(total_images))
    first = max(1, int(first_image))
    last = min(first + count - 1, int(total_images))
    return DialsImageRange(first_image=first, last_image=last)


def _vec_len(v: Sequence[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in v))


def _angle_deg(u: Sequence[float], v: Sequence[float]) -> float:
    lu = _vec_len(u)
    lv = _vec_len(v)
    if lu <= 0 or lv <= 0:
        return 90.0
    dot = sum(float(a) * float(b) for a, b in zip(u, v)) / (lu * lv)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def parse_crystal_from_expt(path: Path) -> DialsCrystalSolution:
    """Read unit cell / space group from a DIALS experiments JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    crystals = data.get("crystal") or []
    if not crystals:
        raise RuntimeError(f"No crystal model in {path}")
    cryst = crystals[0]
    a_vec = cryst["real_space_a"]
    b_vec = cryst["real_space_b"]
    c_vec = cryst["real_space_c"]
    a = _vec_len(a_vec)
    b = _vec_len(b_vec)
    c = _vec_len(c_vec)
    alpha = _angle_deg(b_vec, c_vec)
    beta = _angle_deg(a_vec, c_vec)
    gamma = _angle_deg(a_vec, b_vec)
    sg = str(cryst.get("space_group_hall_symbol") or "P 1").strip()
    # Hall → compact display (keep spaces for DIALS re-use).
    return DialsCrystalSolution(
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        spacegroup=sg.replace(" ", "") if sg else "P1",
        lattice=_guess_lattice(a, b, c, alpha, beta, gamma),
    )


def _guess_lattice(
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> str:
    """Rough Bravais hint from cell metrics (P1 fallback)."""
    tol_l = 0.5  # Å
    tol_a = 1.0  # deg
    ab = abs(a - b) < tol_l
    ac = abs(a - c) < tol_l
    bc = abs(b - c) < tol_l
    aa90 = abs(alpha - 90.0) < tol_a
    bb90 = abs(beta - 90.0) < tol_a
    gg90 = abs(gamma - 90.0) < tol_a
    if ab and bc and aa90 and bb90 and gg90:
        return "cP"
    if ab and aa90 and bb90 and gg90:
        return "tP"
    if aa90 and bb90 and gg90:
        return "oP"
    if aa90 and gg90 and abs(beta - 90.0) >= tol_a:
        return "mP"
    return "aP"


def count_strong_spots(refl_path: Path) -> int:
    """Count reflections in a ``.refl`` msgpack via ``dials.show`` if needed.

    Falls back to parsing ``dials.find_spots.log`` in the same directory.
    """
    log = Path(refl_path).with_name("dials.find_spots.log")
    if log.is_file():
        text = log.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            # "Saved 2261 reflections to strong.refl"
            if "Saved" in line and "reflection" in line and Path(refl_path).name in line:
                parts = line.split()
                for i, tok in enumerate(parts):
                    if tok == "Saved" and i + 1 < len(parts):
                        try:
                            return int(parts[i + 1])
                        except ValueError:
                            break
            if line.strip().startswith("Filtered ") and " spots" in line:
                # "Filtered 2261 of 2272 spots by peak-centroid distance"
                try:
                    return int(line.split()[1])
                except (IndexError, ValueError):
                    pass
    # Last resort: file size heuristic is unreliable; return 0 and let UI warn.
    return 0


def find_ccp4_python() -> Path | None:
    """Locate the CCP4/DIALS ``python3`` that can import ``dials``."""
    prefix = find_dials_prefix()
    roots: list[Path] = []
    if prefix is not None:
        roots.append(prefix.parent)
    for key in ("CCP4", "DIALS_DIR"):
        root = os.environ.get(key)
        if root:
            roots.append(Path(root))
    roots.extend(
        (
            Path("/Applications/ccp4-9"),
            Path("/Applications/ccp4-8"),
        )
    )
    seen: set[Path] = set()
    for root in roots:
        try:
            root = root.resolve()
        except OSError:
            continue
        if root in seen:
            continue
        seen.add(root)
        for candidate in (
            root / "Frameworks/Python.framework/Versions/3.9/bin/python3.9",
            root / "Frameworks/Python.framework/Versions/3.10/bin/python3.10",
            root / "bin/python3",
            root / "bin/python",
        ):
            try:
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    return candidate.resolve()
            except OSError:
                continue
    return None


_EXPORT_CENTROIDS_SCRIPT = r"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from dials.array_family import flex

refl_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
refl = flex.reflection_table.from_file(str(refl_path))
xs, ys, zs = refl["xyzobs.px.value"].parts()
np.savez_compressed(
    out_path,
    x=np.asarray(xs, dtype=np.float64),
    y=np.asarray(ys, dtype=np.float64),
    z=np.asarray(zs, dtype=np.float64),
)
print(len(xs))
"""


_EXPORT_PREDICTIONS_SCRIPT = r"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from dials.array_family import flex

refl_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
refl = flex.reflection_table.from_file(str(refl_path))
if "xyzcal.px" not in refl:
    raise SystemExit("reflection table has no xyzcal.px column")
indexed = refl.get_flags(refl.flags.indexed)
sel = refl.select(indexed)
xs, ys, zs = sel["xyzcal.px"].parts()
np.savez_compressed(
    out_path,
    x=np.asarray(xs, dtype=np.float64),
    y=np.asarray(ys, dtype=np.float64),
    z=np.asarray(zs, dtype=np.float64),
)
print(len(xs))
"""


def _run_ccp4_python_script(script: str, *args: str) -> str:
    """Run *script* with CCP4 Python; return stdout. Raises on failure."""
    python = find_ccp4_python()
    if python is None:
        raise FileNotFoundError(
            "CCP4/DIALS Python not found; cannot export reflection tables."
        )
    proc = subprocess.run(
        [str(python), "-c", script, *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-20:]
        raise RuntimeError("CCP4 Python script failed:\n" + "\n".join(tail))
    return proc.stdout or ""


def export_strong_centroids_npz(
    strong_refl: Path,
    out_path: Path | None = None,
) -> Path:
    """Export ``xyzobs.px`` centroids from *strong_refl* to a NumPy ``.npz``.

    DiffractionLab cannot import DIALS into its own venv (boost mismatch), so this
    shells out to CCP4's Python. The cache enables fast per-frame overlays.
    """
    strong_refl = Path(strong_refl)
    if not strong_refl.is_file():
        raise FileNotFoundError(strong_refl)
    out_path = Path(out_path) if out_path is not None else strong_refl.with_name(
        "strong_centroids.npz"
    )
    _run_ccp4_python_script(
        _EXPORT_CENTROIDS_SCRIPT, str(strong_refl), str(out_path)
    )
    if not out_path.is_file():
        raise RuntimeError(f"Failed to write centroids cache: {out_path}")
    return out_path


def export_indexed_predictions_npz(
    indexed_refl: Path,
    out_path: Path | None = None,
) -> Path:
    """Export indexed ``xyzcal.px`` predictions to a NumPy ``.npz``.

    Only reflections with ``flags.indexed`` are written. Frame filtering uses
    the same half-open z window as :func:`centroids_px_for_image`.
    """
    indexed_refl = Path(indexed_refl)
    if not indexed_refl.is_file():
        raise FileNotFoundError(indexed_refl)
    out_path = Path(out_path) if out_path is not None else indexed_refl.with_name(
        "indexed_predictions.npz"
    )
    _run_ccp4_python_script(
        _EXPORT_PREDICTIONS_SCRIPT, str(indexed_refl), str(out_path)
    )
    if not out_path.is_file():
        raise RuntimeError(f"Failed to write predictions cache: {out_path}")
    return out_path


def centroids_px_for_image(
    centroids_npz: Path,
    image_index_0based: int,
) -> list[tuple[float, float]]:
    """Return ``(x_px, y_px)`` centroids for one 0-based image index.

    DIALS stores the slow/fast pixel centroid in ``xyzobs.px.value``; the z
    coordinate uses half-integer frame indices (``image + 0.5``), so a frame
    matches when ``image_index <= z < image_index + 1``.
    """
    import numpy as np

    path = Path(centroids_npz)
    if not path.is_file():
        return []
    data = np.load(path)
    try:
        x = data["x"]
        y = data["y"]
        z = data["z"]
    finally:
        data.close()
    index = int(image_index_0based)
    mask = (z >= index) & (z < index + 1)
    if not np.any(mask):
        return []
    xs = x[mask]
    ys = y[mask]
    return [(float(px), float(py)) for px, py in zip(xs, ys)]


class DialsEngine:
    """Orchestrate DIALS CLI steps in a working directory."""

    def __init__(
        self,
        *,
        work_dir: Path | None = None,
        settings: DialsSettings | None = None,
        dials_bin_dir: Path | None = None,
    ) -> None:
        self.settings = settings or DialsSettings()
        self.work_dir = Path(work_dir or (Path.home() / ".qtmosflm" / "dials"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.dials_bin_dir = dials_bin_dir or find_dials_prefix()
        if self.dials_bin_dir is None:
            raise FileNotFoundError(
                "DIALS not found. Install CCP4/DIALS or ensure dials.import is on PATH."
            )

    def _exe(self, name: str) -> Path:
        path = self.dials_bin_dir / name
        if not path.is_file():
            found = find_dials_bin(name)
            if found is None:
                raise FileNotFoundError(f"{name} not found under {self.dials_bin_dir}")
            return found
        return path

    def _run(
        self,
        name: str,
        args: Sequence[str],
        *,
        log_name: str | None = None,
    ) -> str:
        cmd = [str(self._exe(name)), *args]
        logger.info("Running: %s", " ".join(cmd))
        proc = subprocess.run(
            cmd,
            cwd=str(self.work_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        combined = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        if log_name:
            (self.work_dir / log_name).write_text(combined, encoding="utf-8")
        if proc.returncode != 0:
            tail = "\n".join(combined.strip().splitlines()[-40:])
            raise RuntimeError(
                f"{name} failed (exit {proc.returncode}). Last output:\n{tail}"
            )
        return combined

    def import_images(self, image_path: Path) -> Path:
        """Run ``dials.import`` on *image_path* (master HDF5 or image template)."""
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        self._run(
            "dials.import",
            [str(image_path)],
            log_name="dials.import.log",
        )
        out = self.work_dir / "imported.expt"
        if not out.is_file():
            raise RuntimeError("dials.import did not write imported.expt")
        return out

    def find_spots(
        self,
        imported_expt: Path,
        image_range: DialsImageRange,
    ) -> DialsSpotFindingResult:
        """Run ``dials.find_spots`` on *image_range*."""
        args = [
            str(imported_expt),
            f"nproc={self.settings.nproc}",
            f"spotfinder.scan_range={image_range.as_scan_range()}",
            f"spotfinder.filter.min_spot_size={self.settings.min_spot_size}",
        ]
        if self.settings.gain is not None:
            args.append(f"spotfinder.threshold.dispersion.gain={self.settings.gain:g}")
        log = self._run("dials.find_spots", args)
        strong = self.work_dir / "strong.refl"
        if not strong.is_file():
            raise RuntimeError("dials.find_spots did not write strong.refl")
        n_spots = count_strong_spots(strong)
        centroids_npz: Path | None = None
        try:
            centroids_npz = export_strong_centroids_npz(strong)
        except Exception as ex:  # noqa: BLE001
            logger.warning("Could not export DIALS centroids for overlay: %s", ex)
        return DialsSpotFindingResult(
            work_dir=self.work_dir,
            imported_expt=Path(imported_expt),
            strong_refl=strong,
            n_spots=n_spots,
            image_range=image_range,
            log=log,
            centroids_npz=centroids_npz,
        )

    def index(
        self,
        imported_expt: Path,
        strong_refl: Path,
        *,
        refine: bool = True,
    ) -> DialsIndexingResult:
        """Run ``dials.index`` and optionally ``dials.refine``."""
        args = [
            str(imported_expt),
            str(strong_refl),
            # Short interactive wedges can trip the volume sanity check.
            "refinement_protocol.disable_unit_cell_volume_sanity_check=True",
        ]
        if self.settings.space_group:
            args.append(f"indexing.known_symmetry.space_group={self.settings.space_group}")
        if self.settings.unit_cell:
            args.append(f"indexing.known_symmetry.unit_cell={self.settings.unit_cell}")
        log = self._run("dials.index", args)
        indexed_expt = self.work_dir / "indexed.expt"
        indexed_refl = self.work_dir / "indexed.refl"
        if not indexed_expt.is_file():
            raise RuntimeError("dials.index did not write indexed.expt")
        experiments = indexed_expt
        reflections = indexed_refl
        did_refine = False
        if refine:
            refine_log = self._run(
                "dials.refine",
                [str(indexed_expt), str(indexed_refl)],
            )
            log = log + "\n" + refine_log
            refined_expt = self.work_dir / "refined.expt"
            refined_refl = self.work_dir / "refined.refl"
            if refined_expt.is_file():
                experiments = refined_expt
                reflections = refined_refl
                did_refine = True
        solution = parse_crystal_from_expt(experiments)
        # Pull indexed counts from log if present.
        for line in log.splitlines():
            if "|          0 |" in line and "indexed" not in line.lower():
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 3:
                    try:
                        solution.n_indexed = int(parts[1])
                        solution.n_unindexed = int(parts[2])
                    except ValueError:
                        pass
        predictions_npz: Path | None = None
        try:
            predictions_npz = export_indexed_predictions_npz(reflections)
        except Exception as ex:  # noqa: BLE001
            logger.warning("Could not export DIALS predictions for overlay: %s", ex)
        return DialsIndexingResult(
            work_dir=self.work_dir,
            experiments=experiments,
            reflections=reflections,
            solution=solution,
            refined=did_refine,
            log=log,
            predictions_npz=predictions_npz,
        )

    def integrate(
        self,
        experiments: Path,
        reflections: Path,
        *,
        image_range: DialsImageRange | None = None,
        mtz_name: str = "integrated.mtz",
    ) -> DialsIntegrationResult:
        """Run ``dials.integrate`` then ``dials.export`` to MTZ."""
        args = [
            str(experiments),
            str(reflections),
            f"nproc={self.settings.nproc}",
        ]
        if self.settings.d_min is not None:
            args.append(f"prediction.d_min={self.settings.d_min:g}")
        # scan_range for integrate uses prediction; for a trial wedge the
        # experiments already encode the scan — integrating the refined model
        # over the indexed range is the usual interactive approach.
        log = self._run("dials.integrate", args)
        integrated_expt = self.work_dir / "integrated.expt"
        integrated_refl = self.work_dir / "integrated.refl"
        if not integrated_expt.is_file() or not integrated_refl.is_file():
            raise RuntimeError("dials.integrate did not write integrated.{expt,refl}")
        export_log = self._run(
            "dials.export",
            [
                str(integrated_expt),
                str(integrated_refl),
                "format=mtz",
                f"mtz.hklout={mtz_name}",
            ],
        )
        log = log + "\n" + export_log
        mtz_path = self.work_dir / mtz_name
        if not mtz_path.is_file():
            # Older dials.export may write integrated.mtz regardless of hklout.
            fallback = self.work_dir / "integrated.mtz"
            if fallback.is_file():
                mtz_path = fallback
            else:
                raise RuntimeError("dials.export did not write an MTZ file")
        n_ref = None
        for line in log.splitlines():
            if "Saving" in line and "reflections to integrated.refl" in line:
                try:
                    n_ref = int(line.split()[1])
                except (IndexError, ValueError):
                    pass
        if image_range is None:
            image_range = DialsImageRange(1, 1)
        return DialsIntegrationResult(
            work_dir=self.work_dir,
            experiments=integrated_expt,
            reflections=integrated_refl,
            mtz_path=mtz_path,
            image_range=image_range,
            n_reflections=n_ref,
            log=log,
        )

    def process_hdf5(
        self,
        master_h5: Path,
        image_range: DialsImageRange,
        *,
        refine: bool = True,
        integrate: bool = True,
    ) -> tuple[DialsSpotFindingResult, DialsIndexingResult, DialsIntegrationResult | None]:
        """Full interactive pipeline on an HDF5 master for *image_range*."""
        imported = self.import_images(master_h5)
        spots = self.find_spots(imported, image_range)
        indexed = self.index(imported, spots.strong_refl, refine=refine)
        integ = None
        if integrate:
            integ = self.integrate(
                indexed.experiments,
                indexed.reflections,
                image_range=image_range,
            )
        return spots, indexed, integ
