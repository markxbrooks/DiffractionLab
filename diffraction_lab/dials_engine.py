"""DIALS processing pipeline for datasets Mosflm cannot open (e.g. Eiger HDF5).

Mirrors the DUI tutorial flow:
``import → find_spots → index → refine_bravais_settings → reindex → refine
→ integrate → export`` (+ optional ``symmetry → scale → merge`` / report).

See https://dials.github.io/documentation/tutorials/processing_in_detail_betalactamase_dui.html
and https://github.com/dials/dials/
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

logger = logging.getLogger(__name__)


def dials_run_root_for_master(master_path: Path | str) -> Path:
    """Return ``<master_dir>/<master_stem>/`` beside an HDF5 / NeXus master.

    Example: ``…/Downloads/thaumatin_9_1_master.h5`` →
    ``…/Downloads/thaumatin_9_1_master/``.
    """
    master = Path(master_path).expanduser().resolve()
    return master.parent / master.stem


def default_dials_work_dir(
    master_path: Path | str,
    *,
    when: datetime | None = None,
) -> Path:
    """Timestamped DIALS run directory beside the master file.

    Example: ``…/thaumatin_9_1_master.h5`` →
    ``…/thaumatin_9_1_master/2026-10-15-20-30-00-dials/``.
    """
    stamp = (when or datetime.now()).strftime("%Y-%m-%d-%H-%M-%S")
    return dials_run_root_for_master(master_path) / f"{stamp}-dials"


def latest_dials_run_dir(master_path: Path | str) -> Path | None:
    """Newest ``*-dials`` run folder under the master stem directory, if any."""
    root = dials_run_root_for_master(master_path)
    if not root.is_dir():
        return None
    runs = sorted(
        (p for p in root.glob("*-dials") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    return runs[-1] if runs else None


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
    #: Path to triclinic ``indexed.expt`` (before Bravais / reindex).
    indexed_expt: Path | None = None
    indexed_refl: Path | None = None


@dataclass
class DialsBravaisSolution:
    """One row from ``bravais_summary.json``."""

    setting_number: int
    bravais: str
    unit_cell: tuple[float, float, float, float, float, float]
    cb_op: str
    metric_fit: float
    rmsd: float
    nspots: int
    recommended: bool
    min_cc: float | None = None
    max_cc: float | None = None
    experiments: Path | None = None  # bravais_setting_N.expt


@dataclass
class DialsBravaisResult:
    """Result of ``dials.refine_bravais_settings``."""

    work_dir: Path
    summary_json: Path
    solutions: list[DialsBravaisSolution]
    log: str = ""


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
class DialsScaleResult:
    """Result of ``dials.symmetry`` → ``dials.scale`` → ``dials.merge``."""

    work_dir: Path
    symmetrized_expt: Path
    symmetrized_refl: Path
    scaled_expt: Path
    scaled_refl: Path
    merged_mtz: Path | None
    unmerged_mtz: Path | None
    symmetry_html: Path | None = None
    scale_html: Path | None = None
    log: str = ""


@dataclass
class DialsReportResult:
    """Result of ``dials.report`` / spot-count QA helpers."""

    work_dir: Path
    report_html: Path | None = None
    spot_counts_json: Path | None = None
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
    scan_varying: bool = False


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


def _optional_file(path: Path) -> Path | None:
    return path if path.is_file() else None


def _is_identity_cb_op(cb_op: str) -> bool:
    """Return True if *cb_op* is the identity change-of-basis."""
    normalized = (
        cb_op.strip().lower().replace(" ", "").replace("[", "").replace("]", "")
    )
    return normalized in {"a,b,c", "h,k,l", ""}


def parse_bravais_summary(
    path: Path,
    *,
    work_dir: Path | None = None,
) -> list[DialsBravaisSolution]:
    """Parse ``bravais_summary.json`` from ``dials.refine_bravais_settings``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected bravais_summary format in {path}")
    root = Path(work_dir) if work_dir is not None else Path(path).parent
    solutions: list[DialsBravaisSolution] = []
    for key, item in data.items():
        try:
            number = int(key)
        except (TypeError, ValueError):
            continue
        cell = item.get("unit_cell") or [0, 0, 0, 90, 90, 90]
        if len(cell) != 6:
            continue
        setting_path = root / f"bravais_setting_{number}.expt"
        solutions.append(
            DialsBravaisSolution(
                setting_number=number,
                bravais=str(item.get("bravais") or ""),
                unit_cell=(
                    float(cell[0]),
                    float(cell[1]),
                    float(cell[2]),
                    float(cell[3]),
                    float(cell[4]),
                    float(cell[5]),
                ),
                cb_op=str(item.get("cb_op") or "a,b,c"),
                metric_fit=float(item.get("max_angular_difference") or 0.0),
                rmsd=float(item.get("rmsd") or 0.0),
                nspots=int(item.get("nspots") or 0),
                recommended=bool(item.get("recommended")),
                min_cc=(
                    float(item["min_cc"]) if item.get("min_cc") is not None else None
                ),
                max_cc=(
                    float(item["max_cc"]) if item.get("max_cc") is not None else None
                ),
                experiments=setting_path if setting_path.is_file() else None,
            )
        )
    solutions.sort(key=lambda s: s.setting_number, reverse=True)
    return solutions


def pick_recommended_bravais(
    solutions: Sequence[DialsBravaisSolution],
) -> DialsBravaisSolution | None:
    """Pick the highest-numbered recommended Bravais solution (DIALS convention)."""
    recommended = [s for s in solutions if s.recommended]
    if recommended:
        return max(recommended, key=lambda s: s.setting_number)
    return solutions[0] if solutions else None


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
        on_chunk: Callable[[str], None] | None = None,
        session_id: str | None = None,
    ) -> None:
        self.settings = settings or DialsSettings()
        self.work_dir = Path(work_dir or (Path.home() / ".qtmosflm" / "dials"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.dials_bin_dir = dials_bin_dir or find_dials_prefix()
        self.on_chunk = on_chunk
        self.session_id = session_id
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
        on_chunk: Callable[[str], None] | None = None,
    ) -> str:
        """Run a ``dials.*`` command, streaming stdout/stderr to a log file.

        When SchedArray is enabled (``USE_SCHEDARRAY``, default on if the
        package is installed), the command is submitted to
        :class:`~diffraction_lab.jobs.service.JobService` so it is persisted
        and workers write stdout live to *log_name* for
        :class:`~diffraction_lab.terminal_log_viewer.TerminalLogViewer`.

        Otherwise chunks are written incrementally via local ``Popen``.
        Optional *on_chunk* receives decoded text as it arrives (local path
        only; SchedArray relies on file tailing).
        """
        cmd = [str(self._exe(name)), *args]
        logger.info("Running: %s", " ".join(cmd))
        try:
            from diffraction_lab.jobs.service import use_schedarray

            if use_schedarray():
                return self._run_via_schedarray(
                    name, cmd, log_name=log_name, on_chunk=on_chunk
                )
            logger.info(
                "SchedArray not active (package missing or USE_SCHEDARRAY=false); "
                "running %s via local Popen (Jobs tab will not record this step)",
                name,
            )
        except ImportError:
            logger.info(
                "SchedArray import failed; running %s via local Popen", name
            )
        return self._run_local(name, cmd, log_name=log_name, on_chunk=on_chunk)

    def _run_via_schedarray(
        self,
        name: str,
        cmd: Sequence[str],
        *,
        log_name: str | None = None,
        on_chunk: Callable[[str], None] | None = None,
    ) -> str:
        """Submit *cmd* through SchedArray and wait for completion."""
        import shlex

        from diffraction_lab.jobs.service import configure_job_service_for_work_dir

        log_path = (
            self.work_dir / log_name
            if log_name
            else self.work_dir / f"{name}.log"
        )
        err_path = log_path.with_suffix(log_path.suffix + ".err")
        # SchedArray opens output_file with mode "w", so echo the command
        # header inside the shell command rather than pre-writing the file.
        display = " ".join(str(c) for c in cmd)
        quoted = " ".join(shlex.quote(str(c)) for c in cmd)
        # Line-buffer stdout when redirected to a file so Terminal/Jobs log
        # tailing sees output during long steps (e.g. dials.integrate).
        shell_cmd = (
            f"echo {shlex.quote('$ ' + display)}; "
            f"if command -v stdbuf >/dev/null 2>&1; then "
            f"stdbuf -oL -eL {quoted}; else {quoted}; fi"
        )
        step = name.replace("dials.", "", 1) if name.startswith("dials.") else name
        chunk_cb = on_chunk if on_chunk is not None else self.on_chunk
        if chunk_cb is not None:
            chunk_cb(f"$ {display}\n")

        # Per-master DB: <stem_root>/schedarray_scheduler.db
        service = configure_job_service_for_work_dir(self.work_dir)
        status = service.submit_and_wait(
            shell_cmd,
            working_dir=self.work_dir,
            job_name=name,
            output_file=log_path,
            error_file=err_path,
            tool="dials",
            step=step,
            session_id=getattr(self, "session_id", None),
            max_retries=0,
            poll_interval=0.5,
        )
        combined = ""
        if log_path.is_file():
            combined = log_path.read_text(encoding="utf-8", errors="replace")
        state = str(status.get("state", "")).lower()
        if state != "completed":
            rc = status.get("return_code")
            tail = "\n".join(combined.strip().splitlines()[-40:])
            raise RuntimeError(
                f"{name} failed via SchedArray "
                f"(state={state!r}, exit {rc}). Last output:\n{tail}"
            )
        return combined

    def _run_local(
        self,
        name: str,
        cmd: Sequence[str],
        *,
        log_name: str | None = None,
        on_chunk: Callable[[str], None] | None = None,
    ) -> str:
        """Run *cmd* with local ``Popen``, streaming to a log file."""
        chunk_cb = on_chunk if on_chunk is not None else self.on_chunk
        header = f"$ {' '.join(str(c) for c in cmd)}\n"
        log_path = self.work_dir / log_name if log_name else None
        if log_path is not None:
            log_path.write_text(header, encoding="utf-8")
        if chunk_cb is not None:
            chunk_cb(header)

        chunks: list[str] = [header]
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(self.work_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                chunks.append(line)
                if log_path is not None:
                    with open(log_path, "a", encoding="utf-8") as handle:
                        handle.write(line)
                        handle.flush()
                if chunk_cb is not None:
                    chunk_cb(line)
        finally:
            proc.wait()
        combined = "".join(chunks)
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
        log = self._run("dials.find_spots", args, log_name="dials.find_spots.log")
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
        """Run ``dials.index`` and optionally ``dials.refine``.

        Prefer ``refine=False`` when following with
        :meth:`refine_bravais_settings` (tutorial order).
        """
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
        log = self._run("dials.index", args, log_name="dials.index.log")
        indexed_expt = self.work_dir / "indexed.expt"
        indexed_refl = self.work_dir / "indexed.refl"
        if not indexed_expt.is_file():
            raise RuntimeError("dials.index did not write indexed.expt")
        experiments = indexed_expt
        reflections = indexed_refl
        did_refine = False
        if refine:
            experiments, reflections, refine_log = self._refine_pair(
                indexed_expt, indexed_refl
            )
            log = log + "\n" + refine_log
            did_refine = experiments.name.startswith("refined")
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
            indexed_expt=indexed_expt,
            indexed_refl=indexed_refl if indexed_refl.is_file() else None,
        )

    def _refine_pair(
        self,
        experiments: Path,
        reflections: Path,
    ) -> tuple[Path, Path, str]:
        """Run ``dials.refine``; return refined paths when written."""
        args = [str(experiments), str(reflections)]
        if self.settings.scan_varying:
            args.append("scan_varying=True")
        refine_log = self._run(
            "dials.refine",
            args,
            log_name="dials.refine.log",
        )
        refined_expt = self.work_dir / "refined.expt"
        refined_refl = self.work_dir / "refined.refl"
        if refined_expt.is_file() and refined_refl.is_file():
            return refined_expt, refined_refl, refine_log
        return experiments, reflections, refine_log

    def refine_bravais_settings(
        self,
        experiments: Path | None = None,
        reflections: Path | None = None,
    ) -> DialsBravaisResult:
        """Run ``dials.refine_bravais_settings``; parse ``bravais_summary.json``."""
        expt = Path(experiments or (self.work_dir / "indexed.expt"))
        refl = Path(reflections or (self.work_dir / "indexed.refl"))
        if not expt.is_file() or not refl.is_file():
            raise FileNotFoundError(
                f"Need indexed.expt/refl for Bravais settings (got {expt}, {refl})"
            )
        log = self._run(
            "dials.refine_bravais_settings",
            [str(expt), str(refl), f"nproc={self.settings.nproc}"],
            log_name="dials.refine_bravais_settings.log",
        )
        summary = self.work_dir / "bravais_summary.json"
        if not summary.is_file():
            raise RuntimeError("dials.refine_bravais_settings did not write bravais_summary.json")
        solutions = parse_bravais_summary(summary, work_dir=self.work_dir)
        return DialsBravaisResult(
            work_dir=self.work_dir,
            summary_json=summary,
            solutions=solutions,
            log=log,
        )

    def apply_bravais_solution(
        self,
        solution: DialsBravaisSolution,
        *,
        indexed_expt: Path | None = None,
        indexed_refl: Path | None = None,
        refine: bool = True,
    ) -> DialsIndexingResult:
        """Reindex (if needed) + refine for a chosen Bravais solution.

        Uses ``bravais_setting_N.expt`` plus ``dials.reindex`` on the triclinic
        reflections when ``cb_op`` is not the identity ``a,b,c``.
        """
        expt = Path(indexed_expt or (self.work_dir / "indexed.expt"))
        refl = Path(indexed_refl or (self.work_dir / "indexed.refl"))
        setting_expt = solution.experiments or (
            self.work_dir / f"bravais_setting_{solution.setting_number}.expt"
        )
        if not setting_expt.is_file():
            raise FileNotFoundError(setting_expt)
        if not refl.is_file():
            raise FileNotFoundError(refl)

        cb = (solution.cb_op or "a,b,c").strip()
        log_parts: list[str] = []
        if _is_identity_cb_op(cb):
            # Reflections already match; copy/use setting experiments.
            experiments = setting_expt
            reflections = refl
            log_parts.append(f"Identity cb_op ({cb}); using {setting_expt.name}")
        else:
            reindex_log = self._run(
                "dials.reindex",
                [
                    str(setting_expt),
                    str(refl),
                    f"change_of_basis_op={cb}",
                ],
                log_name="dials.reindex.log",
            )
            log_parts.append(reindex_log)
            # dials.reindex writes reindexed.expt / reindexed.refl by default
            reindexed_expt = self.work_dir / "reindexed.expt"
            reindexed_refl = self.work_dir / "reindexed.refl"
            if not reindexed_expt.is_file():
                # Some versions only rewrite reflections; keep setting expt.
                experiments = setting_expt
            else:
                experiments = reindexed_expt
            if not reindexed_refl.is_file():
                raise RuntimeError("dials.reindex did not write reindexed.refl")
            reflections = reindexed_refl

        did_refine = False
        if refine:
            experiments, reflections, refine_log = self._refine_pair(
                experiments, reflections
            )
            log_parts.append(refine_log)
            did_refine = experiments.name.startswith("refined")

        solution_crystal = parse_crystal_from_expt(experiments)
        solution_crystal.lattice = solution.bravais or solution_crystal.lattice
        predictions_npz: Path | None = None
        try:
            predictions_npz = export_indexed_predictions_npz(reflections)
        except Exception as ex:  # noqa: BLE001
            logger.warning("Could not export DIALS predictions for overlay: %s", ex)
        return DialsIndexingResult(
            work_dir=self.work_dir,
            experiments=experiments,
            reflections=reflections,
            solution=solution_crystal,
            refined=did_refine,
            log="\n".join(log_parts),
            predictions_npz=predictions_npz,
            indexed_expt=expt if expt.is_file() else None,
            indexed_refl=refl if refl.is_file() else None,
        )

    def symmetry_scale_merge(
        self,
        experiments: Path,
        reflections: Path,
        *,
        d_min: float | None = None,
    ) -> DialsScaleResult:
        """Run ``dials.symmetry`` → ``dials.scale`` → ``dials.merge``.

        Writes ``symmetrized.*``, ``scaled.*``, ``merged.mtz``, and HTML reports.
        """
        expt = Path(experiments)
        refl = Path(reflections)
        log_parts: list[str] = []
        sym_log = self._run(
            "dials.symmetry",
            [str(expt), str(refl)],
            log_name="dials.symmetry.log",
        )
        log_parts.append(sym_log)
        sym_expt = self.work_dir / "symmetrized.expt"
        sym_refl = self.work_dir / "symmetrized.refl"
        if not sym_expt.is_file() or not sym_refl.is_file():
            raise RuntimeError("dials.symmetry did not write symmetrized.{expt,refl}")

        scale_args = [
            str(sym_expt),
            str(sym_refl),
            f"nproc={self.settings.nproc}",
            "unmerged_mtz=scaled_unmerged.mtz",
            "merged_mtz=scaled.mtz",
        ]
        res = d_min if d_min is not None else self.settings.d_min
        if res is not None and res > 0:
            scale_args.append(f"cut_data.d_min={res:g}")
        scale_log = self._run(
            "dials.scale",
            scale_args,
            log_name="dials.scale.log",
        )
        log_parts.append(scale_log)
        scaled_expt = self.work_dir / "scaled.expt"
        scaled_refl = self.work_dir / "scaled.refl"
        if not scaled_expt.is_file() or not scaled_refl.is_file():
            raise RuntimeError("dials.scale did not write scaled.{expt,refl}")

        merged_mtz = self.work_dir / "merged.mtz"
        unmerged = self.work_dir / "scaled_unmerged.mtz"
        # Prefer dials.merge when scaled.mtz was not written by scale.
        if not (self.work_dir / "scaled.mtz").is_file():
            merge_log = self._run(
                "dials.merge",
                [str(scaled_refl), str(scaled_expt)],
                log_name="dials.merge.log",
            )
            log_parts.append(merge_log)
        scaled_mtz = self.work_dir / "scaled.mtz"
        if scaled_mtz.is_file() and not merged_mtz.is_file():
            # Treat scale's merged_mtz as the merge product.
            merged_mtz = scaled_mtz
        elif not merged_mtz.is_file() and scaled_mtz.is_file():
            merged_mtz = scaled_mtz

        return DialsScaleResult(
            work_dir=self.work_dir,
            symmetrized_expt=sym_expt,
            symmetrized_refl=sym_refl,
            scaled_expt=scaled_expt,
            scaled_refl=scaled_refl,
            merged_mtz=merged_mtz if merged_mtz.is_file() else None,
            unmerged_mtz=unmerged if unmerged.is_file() else None,
            symmetry_html=_optional_file(self.work_dir / "dials.symmetry.html"),
            scale_html=_optional_file(self.work_dir / "dials.scale.html"),
            log="\n".join(log_parts),
        )

    def report(
        self,
        experiments: Path | None = None,
        reflections: Path | None = None,
    ) -> DialsReportResult:
        """Run ``dials.report`` → ``dials.report.html``."""
        expt = Path(experiments or (self.work_dir / "refined.expt"))
        refl = Path(reflections or (self.work_dir / "refined.refl"))
        # Prefer integrated when present.
        if (self.work_dir / "integrated.expt").is_file():
            expt = self.work_dir / "integrated.expt"
            refl = self.work_dir / "integrated.refl"
        args: list[str] = []
        if expt.is_file():
            args.append(str(expt))
        if refl.is_file():
            args.append(str(refl))
        if not args:
            raise FileNotFoundError("No experiments/reflections for dials.report")
        log = self._run("dials.report", args, log_name="dials.report.log")
        html = self.work_dir / "dials.report.html"
        return DialsReportResult(
            work_dir=self.work_dir,
            report_html=html if html.is_file() else None,
            log=log,
        )

    def spot_counts_per_image(
        self,
        experiments: Path | None = None,
        reflections: Path | None = None,
    ) -> DialsReportResult:
        """Run ``dials.spot_counts_per_image`` (JSON plot data)."""
        expt = Path(experiments or (self.work_dir / "imported.expt"))
        refl = Path(reflections or (self.work_dir / "strong.refl"))
        if not expt.is_file() or not refl.is_file():
            raise FileNotFoundError("Need imported.expt + strong.refl for spot counts")
        json_path = self.work_dir / "spot_counts.json"
        log = self._run(
            "dials.spot_counts_per_image",
            [str(expt), str(refl), f"json={json_path.name}"],
            log_name="dials.spot_counts_per_image.log",
        )
        return DialsReportResult(
            work_dir=self.work_dir,
            spot_counts_json=json_path if json_path.is_file() else None,
            log=log,
        )

    def launch_reciprocal_lattice_viewer(
        self,
        experiments: Path | None = None,
        reflections: Path | None = None,
    ) -> subprocess.Popen:
        """Start ``dials.reciprocal_lattice_viewer`` (external GUI)."""
        expt = Path(experiments or (self.work_dir / "indexed.expt"))
        refl = Path(reflections or (self.work_dir / "indexed.refl"))
        args = [str(self._exe("dials.reciprocal_lattice_viewer"))]
        if expt.is_file():
            args.append(str(expt))
        if refl.is_file():
            args.append(str(refl))
        return subprocess.Popen(args, cwd=str(self.work_dir))

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
        if image_range is not None:
            # Limit trial integration to the selected contiguous wedge.
            args.append(f"scan_range={image_range.as_scan_range()}")
        log = self._run("dials.integrate", args, log_name="dials.integrate.log")
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
            log_name="dials.export.log",
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
        bravais: bool = True,
    ) -> tuple[DialsSpotFindingResult, DialsIndexingResult, DialsIntegrationResult | None]:
        """Full interactive pipeline on an HDF5 master for *image_range*.

        When *bravais* is True, runs refine_bravais_settings and applies the
        recommended solution before the final refine / integrate.
        """
        imported = self.import_images(master_h5)
        spots = self.find_spots(imported, image_range)
        indexed = self.index(
            imported, spots.strong_refl, refine=False if bravais else refine
        )
        if bravais:
            try:
                bravais_result = self.refine_bravais_settings(
                    indexed.indexed_expt or indexed.experiments,
                    indexed.indexed_refl or indexed.reflections,
                )
                chosen = pick_recommended_bravais(bravais_result.solutions)
                if chosen is not None:
                    indexed = self.apply_bravais_solution(
                        chosen,
                        indexed_expt=indexed.indexed_expt,
                        indexed_refl=indexed.indexed_refl,
                        refine=refine,
                    )
                elif refine:
                    indexed = self.index(imported, spots.strong_refl, refine=True)
            except Exception as ex:  # noqa: BLE001
                logger.warning("Bravais step failed (%s); falling back", ex)
                if refine:
                    indexed = self.index(imported, spots.strong_refl, refine=True)
        integ = None
        if integrate:
            integ = self.integrate(
                indexed.experiments,
                indexed.reflections,
                image_range=image_range,
            )
        return spots, indexed, integ


# ---------------------------------------------------------------------------
# Session / work-dir rehydration (no DIALS re-run)
# ---------------------------------------------------------------------------


def load_spot_result_from_work_dir(
    work_dir: Path | str,
    *,
    image_range: DialsImageRange | None = None,
) -> DialsSpotFindingResult | None:
    """Rebuild :class:`DialsSpotFindingResult` from ``imported.expt`` + ``strong.refl``.

    :param work_dir: DIALS run directory.
    :param image_range: Optional wedge (defaults to ``1…1`` if unknown).
    :returns: Result or ``None`` if required files are missing.
    """
    root = Path(work_dir)
    imported = root / "imported.expt"
    strong = root / "strong.refl"
    if not imported.is_file() or not strong.is_file():
        return None
    n_spots = count_strong_spots(strong)
    centroids: Path | None = None
    for candidate in (
        root / "strong_centroids.npz",
        root / "centroids.npz",
    ):
        if candidate.is_file():
            centroids = candidate
            break
    if centroids is None:
        try:
            centroids = export_strong_centroids_npz(strong)
        except Exception as ex:  # noqa: BLE001
            logger.debug("Could not export centroids while rehydrating: %s", ex)
    rng = image_range or DialsImageRange(1, 1)
    log_text = ""
    log_path = root / "dials.find_spots.log"
    if log_path.is_file():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    return DialsSpotFindingResult(
        work_dir=root,
        imported_expt=imported,
        strong_refl=strong,
        n_spots=n_spots,
        image_range=rng,
        log=log_text,
        centroids_npz=centroids,
    )


def _pair_expt_refl(root: Path, stem: str) -> tuple[Path, Path] | None:
    expt = root / f"{stem}.expt"
    refl = root / f"{stem}.refl"
    if expt.is_file() and refl.is_file():
        return expt, refl
    return None


def work_dir_has_report_inputs(work_dir: Path | str) -> bool:
    """True when ``dials.report`` can run or ``dials.report.html`` already exists."""
    root = Path(work_dir)
    if (root / "dials.report.html").is_file():
        return True
    for stem in ("integrated", "refined", "indexed"):
        if _pair_expt_refl(root, stem) is not None:
            return True
    return False


def work_dir_has_scale_inputs(work_dir: Path | str) -> bool:
    """True when ``integrated.expt`` + ``integrated.refl`` exist for symmetry/scale."""
    return _pair_expt_refl(Path(work_dir), "integrated") is not None


def load_index_result_from_work_dir(
    work_dir: Path | str,
) -> tuple[DialsIndexingResult, DialsBravaisResult | None] | None:
    """Rebuild index (+ optional Bravais) results from work-dir experiments.

    Prefers ``refined`` → ``reindexed`` → ``indexed`` experiment pairs.

    :returns: ``(indexed, bravais_or_none)`` or ``None`` if no pair exists.
    """
    root = Path(work_dir)
    pair = None
    refined = False
    for stem, is_refined in (
        ("refined", True),
        ("reindexed", True),
        ("indexed", False),
    ):
        pair = _pair_expt_refl(root, stem)
        if pair is not None:
            refined = is_refined or stem != "indexed"
            break
    if pair is None:
        return None
    experiments, reflections = pair
    try:
        solution = parse_crystal_from_expt(experiments)
    except Exception as ex:  # noqa: BLE001
        logger.warning("Could not parse crystal from %s: %s", experiments, ex)
        solution = DialsCrystalSolution(
            a=0.0,
            b=0.0,
            c=0.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
            spacegroup="P1",
        )
    indexed_pair = _pair_expt_refl(root, "indexed")
    predictions = None
    for name in ("predictions.npz", "indexed_predictions.npz"):
        cand = root / name
        if cand.is_file():
            predictions = cand
            break
    log_text = ""
    for log_name in ("dials.refine.log", "dials.index.log"):
        log_path = root / log_name
        if log_path.is_file():
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            break
    indexed = DialsIndexingResult(
        work_dir=root,
        experiments=experiments,
        reflections=reflections,
        solution=solution,
        refined=refined,
        log=log_text,
        predictions_npz=predictions,
        indexed_expt=indexed_pair[0] if indexed_pair else None,
        indexed_refl=indexed_pair[1] if indexed_pair else None,
    )
    bravais: DialsBravaisResult | None = None
    summary = root / "bravais_summary.json"
    if summary.is_file():
        try:
            solutions = parse_bravais_summary(summary, work_dir=root)
            bravais_log = root / "dials.refine_bravais_settings.log"
            bravais = DialsBravaisResult(
                work_dir=root,
                summary_json=summary,
                solutions=solutions,
                log=(
                    bravais_log.read_text(encoding="utf-8", errors="replace")
                    if bravais_log.is_file()
                    else ""
                ),
            )
        except Exception as ex:  # noqa: BLE001
            logger.warning("Could not rehydrate Bravais summary: %s", ex)
    return indexed, bravais


def load_integration_result_from_work_dir(
    work_dir: Path | str,
    *,
    image_range: DialsImageRange | None = None,
    mtz_path: Path | str | None = None,
) -> DialsIntegrationResult | None:
    """Rebuild :class:`DialsIntegrationResult` from integrated experiments + MTZ."""
    root = Path(work_dir)
    pair = _pair_expt_refl(root, "integrated")
    if pair is None:
        return None
    experiments, reflections = pair
    mtz: Path | None = Path(mtz_path) if mtz_path else None
    if mtz is None or not mtz.is_file():
        for name in ("integrated.mtz", "dials.mtz", "scaled.mtz"):
            cand = root / name
            if cand.is_file():
                mtz = cand
                break
    if mtz is None or not mtz.is_file():
        return None
    n_ref = count_strong_spots(reflections) or None
    log_text = ""
    log_path = root / "dials.integrate.log"
    if log_path.is_file():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    return DialsIntegrationResult(
        work_dir=root,
        experiments=experiments,
        reflections=reflections,
        mtz_path=mtz,
        image_range=image_range or DialsImageRange(1, 1),
        n_reflections=n_ref,
        log=log_text,
    )


def load_scale_result_from_work_dir(
    work_dir: Path | str,
) -> DialsScaleResult | None:
    """Rebuild :class:`DialsScaleResult` when ``scaled.expt`` / ``scaled.refl`` exist."""
    root = Path(work_dir)
    scaled = _pair_expt_refl(root, "scaled")
    if scaled is None:
        return None
    scaled_expt, scaled_refl = scaled
    sym = _pair_expt_refl(root, "symmetrized")
    # Symmetrized may be missing on older runs; fall back to scaled paths.
    if sym is None:
        sym_expt, sym_refl = scaled_expt, scaled_refl
    else:
        sym_expt, sym_refl = sym
    merged = None
    for name in ("scaled.mtz", "merged.mtz"):
        cand = root / name
        if cand.is_file():
            merged = cand
            break
    unmerged = _optional_file(root / "scaled_unmerged.mtz")
    return DialsScaleResult(
        work_dir=root,
        symmetrized_expt=sym_expt,
        symmetrized_refl=sym_refl,
        scaled_expt=scaled_expt,
        scaled_refl=scaled_refl,
        merged_mtz=merged,
        unmerged_mtz=unmerged,
        symmetry_html=_optional_file(root / "dials.symmetry.html"),
        scale_html=_optional_file(root / "dials.scale.html"),
        log="",
    )


def find_pointless_artifacts(
    work_dir: Path | str,
    *,
    hklout: Path | str | None = None,
    hklin: Path | str | None = None,
) -> tuple[Path | None, Path | None, Path | None]:
    """Locate Pointless HKLIN / HKLOUT / log under *work_dir*.

    :returns: ``(hklin, hklout, log)`` — any may be ``None``.
    """
    root = Path(work_dir)
    out = Path(hklout).expanduser() if hklout else None
    if out is not None and not out.is_file():
        out = None
    if out is None:
        for pattern in ("*pointless.mtz", "pointless.mtz"):
            matches = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime)
            if matches:
                out = matches[-1]
                break
    log: Path | None = None
    logs = sorted(root.glob("*pointless.log"), key=lambda p: p.stat().st_mtime)
    if logs:
        log = logs[-1]
    inp = Path(hklin).expanduser() if hklin else None
    if inp is not None and not inp.is_file():
        inp = None
    if inp is None:
        for name in ("integrated.mtz", "dials.mtz", "scaled.mtz"):
            cand = root / name
            if cand.is_file() and (out is None or cand.resolve() != out.resolve()):
                inp = cand
                break
    return inp, out, log
