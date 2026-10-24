"""Run CCP4 Dimple (refinement against a known model) and read its results.

Dimple (https://ccp4.github.io/dimple/) takes merged data and a model of
the same or a similar crystal, places the model (rigid-body refinement, or
molecular replacement when R stays above ``--mr-when-r``), refines it with
Refmac and searches the difference map for unmodelled blobs:

``dimple [options] data.mtz model.pdb output_dir``

It writes ``final.pdb`` / ``final.mtz``, ``coot.sh`` (and ``blobN-coot.py``)
to look at the result, ``blobNvM.png`` pictures with ``-f png``, and
``dimple.log``, a configparser-style record of every step that
:func:`parse_dimple_log` reads.

Dimple needs the CCP4 environment, so it is started through CCP4's
``ccp4.setup-sh`` when that sits beside it.
"""

from __future__ import annotations

import configparser
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

_DIMPLE_MISSING = (
    "CCP4 Dimple not found. Install CCP4, source ccp4.setup-sh, "
    "or set $CBIN / ensure `dimple` is on PATH."
)

#: Model files Dimple reads.
MODEL_SUFFIXES = (".pdb", ".ent", ".cif", ".mmcif")


@dataclass(frozen=True)
class DimpleOptions:
    """Dimple options (``dimple -h``); the defaults are Dimple's own.

    :param slow: ``-s`` given 0, 1 or 2 times: more refinement.
    :param mr_when_r: ``-M``: molecular replacement when rigid-body R stays above this.
    :param free_r_flags: ``-R``: an MTZ with free-R flags, ``"-"`` for the data MTZ's own,
        or ``None`` for new flags.
    :param resolution: ``--reso`` high-resolution limit (Å), or ``None`` for all data.
    :param anode: ``--anode``: peaks of the anomalous map (SHELX AnoDe).
    :param images: ``-f png``: pictures of the blobs.
    :param no_hetatm: ``--no-hetatm``: remove HETATM records from the model.
    :param libin: ``--libin``: ligand descriptions (CIF) for Refmac.
    :param icolumn: ``-I``: intensity column, or ``None`` for Dimple's default (``IMEAN``).
    :param extra: Further arguments, passed as given.
    """

    slow: int = 0
    mr_when_r: float | None = None
    free_r_flags: str | None = None
    resolution: float | None = None
    anode: bool = False
    images: bool = True
    no_hetatm: bool = False
    libin: Path | None = None
    icolumn: str | None = None
    extra: tuple[str, ...] = ()

    @property
    def argv(self) -> list[str]:
        """The options as Dimple arguments."""
        args = ["-s"] * max(0, min(int(self.slow), 2))
        if self.mr_when_r is not None:
            args += ["-M", f"{self.mr_when_r:g}"]
        if self.free_r_flags:
            args += ["-R", str(self.free_r_flags)]
        if self.resolution:
            args += ["--reso", f"{self.resolution:g}"]
        if self.anode:
            args.append("--anode")
        if self.images:
            args += ["-f", "png"]
        if self.no_hetatm:
            args.append("--no-hetatm")
        if self.libin is not None:
            args += ["--libin", str(self.libin)]
        if self.icolumn:
            args += ["-I", self.icolumn]
        return args + list(self.extra)


@dataclass(frozen=True)
class DimpleStep:
    """One step of the workflow, as Dimple's screen summary shows it."""

    name: str
    info: str
    error: str | None = None


@dataclass(frozen=True)
class RefinementStep:
    """R factors before and after one Refmac step (``rigid``, ``jelly``, ``restr``)."""

    name: str
    r_start: float | None
    r_end: float | None
    r_free_start: float | None = None
    r_free_end: float | None = None


@dataclass(frozen=True)
class Blob:
    """An unmodelled difference-density blob (orthogonal Å), with its score."""

    x: float
    y: float
    z: float
    score: float | None
    coot_script: Path | None = None
    images: tuple[Path, ...] = ()


@dataclass(frozen=True)
class DimpleResult:
    """What a Dimple run produced, read from its output directory."""

    output_dir: Path
    log: Path
    steps: tuple[DimpleStep, ...] = ()
    refinement: tuple[RefinementStep, ...] = ()
    blobs: tuple[Blob, ...] = ()
    molecular_replacement: str | None = None
    error: str | None = None
    command: str = ""

    @property
    def final_pdb(self) -> Path | None:
        """The refined model, if written."""
        return _existing(self.output_dir / "final.pdb")

    @property
    def final_mtz(self) -> Path | None:
        """Map coefficients and data after refinement, if written."""
        return _existing(self.output_dir / "final.mtz")

    @property
    def coot_script(self) -> Path | None:
        """``coot.sh``: Coot on the model, maps and first blob."""
        return _existing(self.output_dir / "coot.sh")

    @property
    def final(self) -> RefinementStep | None:
        """The last Refmac step (the final R / R-free)."""
        return self.refinement[-1] if self.refinement else None

    @property
    def succeeded(self) -> bool:
        """Whether Dimple finished with a refined model."""
        return self.error is None and self.final_pdb is not None


def _existing(path: Path) -> Path | None:
    return path if path.is_file() else None


def find_dimple() -> Path | None:
    """Locate ``dimple`` on PATH, ``$CBIN`` or a CCP4 prefix.

    :returns: Absolute path to ``dimple``, or ``None`` if not found.
    """
    which = shutil.which("dimple")
    if which:
        return Path(which).expanduser().resolve()
    prefixes: list[Path] = []
    if cbin := os.environ.get("CBIN", "").strip():
        prefixes.append(Path(cbin).expanduser())
    if ccp4 := os.environ.get("CCP4", "").strip():
        prefixes.append(Path(ccp4).expanduser() / "bin")
    prefixes += [Path("/Applications/ccp4-9/bin"), Path("/Applications/ccp4-8/bin"), Path.home() / "ccp4" / "bin"]
    for prefix in prefixes:
        candidate = prefix / "dimple"
        try:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate.resolve()
        except OSError:
            continue
    return None


def _floats(value: str | None) -> list[float]:
    try:
        parsed = json.loads(value) if value else []
    except json.JSONDecodeError:
        return []
    return [float(v) for v in parsed] if isinstance(parsed, list) else []


def _float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def parse_dimple_log(path: Path) -> DimpleResult:
    """Read a Dimple output directory's ``dimple.log``.

    :param path: ``dimple.log``, or the output directory holding it.
    :returns: Its steps, R factors, blobs and any error.
    :raises FileNotFoundError: If there is no ``dimple.log``.
    """
    path = Path(path)
    log = path / "dimple.log" if path.is_dir() else path
    out = log.parent
    parser = configparser.ConfigParser(interpolation=None, strict=False, comment_prefixes=("#",),
                                       inline_comment_prefixes=None)
    with log.open(encoding="utf-8", errors="replace") as handle:
        parser.read_file(handle)
    steps: list[DimpleStep] = []
    refinement: list[RefinementStep] = []
    blobs: list[Blob] = []
    mr: str | None = None
    error: str | None = None
    for name in parser.sections():
        if name == "workflow":
            continue
        sec = parser[name]
        step_error = sec.get("error")
        if step_error is None and sec.get("exit_status", "0") not in ("0", ""):
            step_error = f"{name} exited with status {sec.get('exit_status')}"
        steps.append(DimpleStep(name, (sec.get("info") or "").strip(), step_error))
        error = error or step_error
        if name.startswith("refmac5"):
            refinement.append(RefinementStep(
                name.split(" ", 1)[-1],
                _float(sec.get("ini_overall_r")), _float(sec.get("overall_r")),
                _float(sec.get("ini_free_r")), _float(sec.get("free_r")),
            ))
        elif name in ("phaser", "molrep"):
            mr = name
        elif name == "find-blobs":
            scores = _floats(sec.get("scores"))
            try:
                centres = json.loads(sec.get("blobs") or "[]")
            except json.JSONDecodeError:
                centres = []
            for k, xyz in enumerate(centres, start=1):
                blobs.append(Blob(
                    float(xyz[0]), float(xyz[1]), float(xyz[2]),
                    scores[k - 1] if k <= len(scores) else None,
                    _existing(out / f"blob{k}-coot.py"),
                    tuple(sorted(out.glob(f"blob{k}v*.png"))),
                ))
    if error is None and not (out / "final.pdb").is_file():
        # e.g. "Error: No phaser solution." is only in the screen summary.
        screen = out / "screen.log"
        try:
            lines = screen.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            lines = []
        error = next((l.strip() for l in reversed(lines) if re.match(r"\s*(Error|Giving up)", l)),
                     "Dimple did not write final.pdb")
    return DimpleResult(output_dir=out, log=log, steps=tuple(steps), refinement=tuple(refinement),
                        blobs=tuple(blobs), molecular_replacement=mr, error=error)


def dimple_command(exe: Path, hklin: Path, xyzin: Path, output_dir: Path, options: DimpleOptions) -> str:
    """Shell command running Dimple, inside CCP4's environment when its setup script is beside it."""
    argv = shlex.join([str(exe), *options.argv, str(hklin), str(xyzin), str(output_dir)])
    setup = exe.parent / "ccp4.setup-sh"
    return f". {shlex.quote(str(setup))} && exec {argv}" if setup.is_file() else f"exec {argv}"


def new_output_dir(parent: Path) -> Path:
    """A fresh ``dimple_YYYYMMDD_HHMMSS`` directory name under *parent* (not created)."""
    stamp = f"dimple_{datetime.now():%Y%m%d_%H%M%S}"
    candidate, n = Path(parent) / stamp, 2
    while candidate.exists():
        candidate, n = Path(parent) / f"{stamp}_{n}", n + 1
    return candidate


def run_dimple(
    hklin: Path,
    xyzin: Path,
    output_dir: Path,
    *,
    options: DimpleOptions | None = None,
    on_line: Callable[[str], None] | None = None,
    job_service: Any = None,
    session_id: str | None = None,
) -> DimpleResult:
    """Refine *xyzin* against *hklin* with Dimple into *output_dir*.

    :param hklin: Merged MTZ (e.g. Aimless or dials.merge output).
    :param xyzin: Model, PDB or mmCIF.
    :param output_dir: Dimple's output directory (created; normally new,
        see :func:`new_output_dir`).
    :param options: Dimple options (default: Dimple's, with blob pictures).
    :param on_line: Called with each line Dimple prints, as it prints it.
    :param job_service: Submit as a SchedArray job to this
        :class:`~diffraction_lab.jobs.service.JobService` (``None``: run here).
    :param session_id: Recorded on the job, to group a session's steps.
    :returns: The parsed results; ``error`` says why if Dimple gave up.
    :raises FileNotFoundError: If Dimple, the MTZ or the model is missing.
    :raises RuntimeError: If Dimple failed before writing ``dimple.log``, or the job was cancelled.
    """
    hklin = Path(hklin).expanduser().resolve()
    xyzin = Path(xyzin).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    for what, p in (("MTZ", hklin), ("Model", xyzin)):
        if not p.is_file():
            raise FileNotFoundError(f"{what} not found: {p}")
    exe = find_dimple()
    if exe is None:
        raise FileNotFoundError(_DIMPLE_MISSING)
    options = options or DimpleOptions()
    emit = on_line or (lambda _line: None)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    command = dimple_command(exe, hklin, xyzin, output_dir, options)
    logger.info("Running Dimple in %s:\n%s", output_dir.parent, command)
    if job_service is not None:
        returncode = _run_as_job(command, output_dir, job_service, session_id, emit)
    else:
        with subprocess.Popen(["/bin/sh", "-c", command], cwd=str(output_dir.parent), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
            assert proc.stdout is not None
            for raw in proc.stdout:
                emit(raw.rstrip("\n"))
            returncode = proc.wait()
    try:
        result = parse_dimple_log(output_dir)
    except FileNotFoundError:
        raise RuntimeError(f"Dimple failed (exit {returncode}) before writing dimple.log: {command}") from None
    if returncode != 0 and result.error is None:
        result = replace(result, error=f"Dimple exited with status {returncode}")
    return replace(result, command=command)


def _run_as_job(command: str, output_dir: Path, service: Any, session_id: str | None,
                emit: Callable[[str], None], poll_s: float = 0.3) -> int:
    """Submit *command* to SchedArray and pass its output on as it is written."""
    from diffraction_lab.jobs.service import TERMINAL_STATES

    log = output_dir.parent / f"{output_dir.name}.out"
    job_id = service.submit(f"{command} 2>&1", working_dir=output_dir.parent, job_name=f"dimple {output_dir.name}",
                            output_file=log, tool="dimple", step="dimple", session_id=session_id,
                            metadata={"run_dir": str(output_dir)})
    position, partial = 0, ""

    def follow() -> None:
        nonlocal position, partial
        try:
            with log.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(position)
                chunk = handle.read()
                position = handle.tell()
        except FileNotFoundError:
            return
        *complete, partial = (partial + chunk).split("\n")
        for line in complete:
            emit(line)

    while True:
        status = service.get_status(job_id) or {}
        state = str(status.get("state", "")).lower()
        follow()
        if state in TERMINAL_STATES:
            break
        time.sleep(poll_s)
    follow()
    if partial:
        emit(partial)
    if state in {"cancelled", "timeout"}:
        raise RuntimeError(f"Dimple job {job_id[:8]} was {state}")
    code = status.get("return_code")
    return int(code) if code is not None else (0 if state == "completed" else 1)


def open_in_coot(script: Path) -> subprocess.Popen:
    """Start Coot on a Dimple script (``coot.sh`` or ``blobN-coot.py``) without waiting.

    :raises FileNotFoundError: If the script or Coot is missing.
    """
    script = Path(script)
    if not script.is_file():
        raise FileNotFoundError(f"Not found: {script}")
    if script.suffix == ".sh":
        argv = ["/bin/sh", str(script)]
    else:
        exe = find_dimple()
        coot = shutil.which("coot") or (str(exe.parent / "coot") if exe and (exe.parent / "coot").is_file() else None)
        if coot is None:
            raise FileNotFoundError("Coot not found")
        argv = [coot, "--no-guano", "--script", str(script)]
    return subprocess.Popen(argv, cwd=str(script.parent), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)
