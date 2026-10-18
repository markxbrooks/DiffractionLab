"""Client for ProcessingLab ``IntegrationLab`` (C++ CONTROL driver).

``IntegrationLab`` does **not** speak ``MOSFLMSOCKET`` (that remains CCP4 /
F77 ``ipmosflm``). DiffractionLab drives it with CONTROL scripts on stdin and
parses machine-readable ``INTEGRATIONLAB_*`` lines (or
``integrationlab_index.txt``) after AUTOINDEX.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

logger = logging.getLogger(__name__)


def find_integrationlab() -> Path | None:
    """Locate ProcessingLab ``IntegrationLab`` (``.exe`` on Windows)."""
    for key in (
        "INTEGRATIONLAB",
        "PROCESSINGLAB_EXEC",
        "PROCESSINGLAB_BIN",
        "ELMO_INTEGRATIONLAB",
    ):
        value = os.environ.get(key)
        if not value:
            continue
        path = Path(value).expanduser()
        if path.is_dir():
            for name in ("IntegrationLab", "IntegrationLab.exe"):
                cand = path / name
                if cand.is_file():
                    return cand.resolve()
        elif path.is_file():
            return path.resolve()

    for name in ("IntegrationLab", "IntegrationLab.exe"):
        which = shutil.which(name)
        if which:
            return Path(which).resolve()

    home = Path.home()
    here = Path(__file__).resolve()
    candidates: list[Path] = []
    # Sibling monorepo layout: …/DiffractionLab and …/ProcessingLab
    for root in (here.parents[2], here.parents[1], Path.cwd(), home / "projects"):
        candidates.extend(
            [
                root / "ProcessingLab" / "bin" / "IntegrationLab",
                root / "ProcessingLab" / "bin" / "IntegrationLab.exe",
                root / "ProcessingLab" / "mosflm-cpp" / "build" / "IntegrationLab",
                root / "ProcessingLab" / "mosflm-cpp" / "build" / "IntegrationLab.exe",
            ]
        )
    env_root = os.environ.get("PROCESSINGLAB_HOME") or os.environ.get(
        "PROCESSINGLAB_ROOT"
    )
    if env_root:
        root = Path(env_root).expanduser()
        candidates.extend(
            [
                root / "bin" / "IntegrationLab",
                root / "bin" / "IntegrationLab.exe",
                root / "mosflm-cpp" / "build" / "IntegrationLab",
            ]
        )

    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate.resolve()
        except OSError:
            continue
    return None


def prefer_processinglab() -> bool:
    """True when DiffractionLab should prefer IntegrationLab over F77 ipmosflm."""
    flag = (os.environ.get("DIFFRACTIONLAB_BACKEND") or "").strip().lower()
    if flag in {"processinglab", "integrationlab", "pl"}:
        return True
    if flag in {"mosflm", "ipmosflm", "f77"}:
        return False
    pref = (os.environ.get("DIFFRACTIONLAB_PREFER_PROCESSINGLAB") or "").strip().lower()
    if pref in {"1", "true", "yes", "on"}:
        return True
    # Prefer ProcessingLab when it is installed and F77 is missing.
    return find_integrationlab() is not None and _find_ipmosflm_silent() is None


def _find_ipmosflm_silent() -> Path | None:
    try:
        from diffraction_lab.mosflm_client import find_ipmosflm

        return find_ipmosflm()
    except Exception:  # noqa: BLE001
        return None


@dataclass
class AutoIndexDump:
    """Parsed IntegrationLab AUTOINDEX dump."""

    cell: tuple[float, float, float, float, float, float]
    matrix: tuple[float, ...]
    system: str = ""
    character: int = 0
    n_spots: int = 0


@dataclass
class ProcessingLabRun:
    """Result of one IntegrationLab CONTROL invocation."""

    returncode: int
    stdout: str
    stderr: str
    work_dir: Path
    autoindex: AutoIndexDump | None = None
    mtz_path: Path | None = None
    log_lines: list[str] = field(default_factory=list)


def parse_autoindex_dump(text: str) -> AutoIndexDump | None:
    """Parse ``INTEGRATIONLAB_*`` lines from stdout (or a sidecar file)."""
    cell: tuple[float, ...] | None = None
    matrix: list[float] = []
    system = ""
    character = 0
    n_spots = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # Sidecar format uses CELL / MATRIX without prefix.
        if line.startswith("INTEGRATIONLAB_CELL ") or line.startswith("CELL "):
            parts = line.split()
            vals = [float(x) for x in parts[1:7]]
            if len(vals) == 6:
                cell = tuple(vals)  # type: ignore[assignment]
            continue
        if line.startswith("INTEGRATIONLAB_MATRIX ") or line.startswith("MATRIX "):
            parts = line.split()
            matrix = [float(x) for x in parts[1:10]]
            continue
        if line.startswith("INTEGRATIONLAB_SYSTEM ") or line.startswith("SYSTEM "):
            system = line.split(None, 1)[1].strip()
            continue
        if line.startswith("INTEGRATIONLAB_CHARACTER ") or line.startswith(
            "CHARACTER "
        ):
            try:
                character = int(line.split()[1])
            except (IndexError, ValueError):
                pass
            continue
        if line.startswith("INTEGRATIONLAB_NSPOTS ") or line.startswith("NSPOTS "):
            try:
                n_spots = int(line.split()[1])
            except (IndexError, ValueError):
                pass
            continue
    if cell is None or len(matrix) != 9:
        return None
    return AutoIndexDump(
        cell=(cell[0], cell[1], cell[2], cell[3], cell[4], cell[5]),
        matrix=tuple(matrix),
        system=system,
        character=character,
        n_spots=n_spots,
    )


def spacegroup_for_system(system: str) -> str:
    """Map DPS Bravais id (e.g. ``cI``) to a default space-group name."""
    mapping = {
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
        "hR": "R3",
        "cP": "P23",
        "cF": "F23",
        "cI": "I23",
    }
    return mapping.get(system.strip(), "P1")


def run_control_script(
    executable: Path,
    commands: Sequence[str],
    *,
    work_dir: Path,
    hklout: Path | None = None,
    image: Path | None = None,
    timeout_s: float = 600.0,
) -> ProcessingLabRun:
    """Run IntegrationLab with a CONTROL script written under *work_dir*."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    script = work_dir / "integrationlab_control.in"
    body = "\n".join(cmd.strip() for cmd in commands if cmd and str(cmd).strip())
    if not body.upper().rstrip().endswith("EXIT"):
        body = body.rstrip() + "\nEXIT\n"
    script.write_text(body + "\n", encoding="utf-8")

    argv: list[str] = [str(executable)]
    if hklout is not None:
        argv.extend(["HKLOUT", str(hklout)])
    if image is not None:
        argv.extend(["--image", str(image)])
    argv.append(str(script))

    logger.info("Running %s in %s", " ".join(argv), work_dir)
    proc = subprocess.run(
        argv,
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    (work_dir / "integrationlab.log").write_text(stdout + "\n" + stderr, encoding="utf-8")

    dump = parse_autoindex_dump(stdout)
    sidecar = work_dir / "integrationlab_index.txt"
    if dump is None and sidecar.is_file():
        dump = parse_autoindex_dump(sidecar.read_text(encoding="utf-8", errors="replace"))

    mtz: Path | None = None
    if hklout is not None and Path(hklout).is_file() and Path(hklout).stat().st_size > 0:
        mtz = Path(hklout)
    else:
        # IntegrationLab may write session.state().hklout default process.mtz
        for cand in (work_dir / "process.mtz", work_dir / "integrationlab.mtz"):
            if cand.is_file() and cand.stat().st_size > 0:
                mtz = cand
                break

    return ProcessingLabRun(
        returncode=int(proc.returncode),
        stdout=stdout,
        stderr=stderr,
        work_dir=work_dir,
        autoindex=dump,
        mtz_path=mtz,
        log_lines=[ln for ln in stdout.splitlines() if ln.strip()],
    )


def build_autoindex_commands(
    *,
    directory: Path,
    spotfile: Path | str,
    distance_mm: float,
    wavelength_a: float,
    beam_x_mm: float,
    beam_y_mm: float,
    pixel_mm: float,
    dmax_a: float = 320.0,
    threshold: float = 20.0,
    hkldev: float = 0.3,
    nvect: int = 30,
    nvref: int = 30,
    solution: int = 12,
    prefer_system: str | None = None,
    extra: Iterable[str] | None = None,
) -> list[str]:
    """CONTROL lines for IntegrationLab AUTOINDEX from a ``.spt`` file."""
    spot = Path(spotfile)
    spot_arg = spot.name if spot.parent.resolve() == Path(directory).resolve() else str(spot)
    auto = (
        f"AUTOINDEX DPS THRESHOLD {threshold:g} HKLDEV {hkldev:g} "
        f"NVECT {nvect} NVREF {nvref} SOLUTION {solution} "
        f'FILE "{spot_arg}"'
    )
    if prefer_system:
        auto += f" SYSTEM {prefer_system}"
    cmds = [
        f"DIRECTORY {directory}",
        f"DISTANCE {distance_mm:.4f}",
        f"WAVELENGTH {wavelength_a:.6f}",
        f"PIXEL {pixel_mm:.6f} {pixel_mm:.6f}",
        f"BEAM {beam_x_mm:.4f} {beam_y_mm:.4f}",
        f"DMAX {dmax_a:.1f}",
        auto,
        "GO",
    ]
    if extra:
        cmds.extend(list(extra))
    return cmds


def build_process_commands(
    *,
    directory: Path,
    image_path: Path | None,
    template: str | None,
    distance_mm: float,
    wavelength_a: float,
    beam_x_mm: float,
    beam_y_mm: float,
    pixel_mm: float,
    cell: Sequence[float],
    matrix: Sequence[float],
    symmetry: int | str = 1,
    mosaic_deg: float = 0.3,
    process_first: int = 1,
    process_last: int = 1,
    phi_start: float = 0.0,
    osc_deg: float = 1.0,
    separation_mm: float = 0.57,
) -> list[str]:
    """CONTROL lines for IntegrationLab PROCESS with known CELL/MATRIX."""
    if len(cell) != 6:
        raise ValueError("cell needs 6 values")
    if len(matrix) != 9:
        raise ValueError("matrix needs 9 values")
    cmds = [
        f"DIRECTORY {directory}",
        f"DISTANCE {distance_mm:.4f}",
        f"WAVELENGTH {wavelength_a:.6f}",
        f"PIXEL {pixel_mm:.6f} {pixel_mm:.6f}",
        f"BEAM {beam_x_mm:.4f} {beam_y_mm:.4f}",
        f"SEPARATION {separation_mm:.3f} {separation_mm:.3f}",
        (
            f"CELL {cell[0]:.4f} {cell[1]:.4f} {cell[2]:.4f} "
            f"{cell[3]:.3f} {cell[4]:.3f} {cell[5]:.3f}"
        ),
        f"SYMMETRY {symmetry}",
        f"MOSAIC {mosaic_deg:.3f}",
        "MATRIX",
        f"{matrix[0]:.8f} {matrix[1]:.8f} {matrix[2]:.8f}",
        f"{matrix[3]:.8f} {matrix[4]:.8f} {matrix[5]:.8f}",
        f"{matrix[6]:.8f} {matrix[7]:.8f} {matrix[8]:.8f}",
    ]
    if template:
        cmds.insert(1, f"TEMPLATE {template}")
    if image_path is not None:
        cmds.append(f"IMAGE {image_path}")
    cmds.append(
        f"PROCESS {process_first} TO {process_last} "
        f"START {phi_start:.4f} ANGLE {osc_deg:.4f}"
    )
    cmds.append("GO")
    return cmds
