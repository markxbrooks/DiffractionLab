"""Run CCP4 Aimless (scale and merge) and parse its summary.

Mirrors :mod:`diffraction_lab.pointless_runner`: locate ``aimless`` on PATH,
``$CBIN`` or a CCP4 prefix, run it with a keyword heredoc into a
timestamped log, and parse the ``SUMMARY_BEGIN`` … ``SUMMARY_END`` block
(overall / inner / outer shell statistics, resolution estimates, space group
and cell).
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from diffraction_lab.pointless_runner import PointlessRunResult

logger = logging.getLogger(__name__)

_AIMLESS_MISSING = (
    "CCP4 Aimless not found. Install CCP4, source ccp4.setup-sh, "
    "or set $CBIN / ensure `aimless` is on PATH."
)

_SHELLS = ("overall", "inner", "outer")
_LIMIT = re.compile(r"from (.+?):\s*limit\s*=\s*([0-9.]+)\s*A")


@dataclass(frozen=True)
class ShellStatistics:
    """One summary row: overall, inner and outer shell values (``None`` = "-")."""

    overall: float | None
    inner: float | None
    outer: float | None


@dataclass(frozen=True)
class AimlessSummary:
    """Parsed Aimless summary (first dataset).

    ``rows`` maps the summary labels as printed (e.g. ``"Rmeas (all I+ & I-)"``,
    ``"Mn(I) half-set correlation CC(1/2)"``) to their shell values.
    ``resolution_estimates`` maps each criterion (e.g. ``"half-dataset
    correlation CC(1/2) >  0.30"``) to its overall limit in Å.
    """

    rows: dict[str, ShellStatistics] = field(default_factory=dict)
    resolution_estimates: dict[str, float] = field(default_factory=dict)
    space_group: str | None = None
    cell: tuple[float, float, float, float, float, float] | None = None

    def value(self, label: str, shell: str = "overall") -> float | None:
        """A row's value for ``shell`` (``overall``, ``inner`` or ``outer``)."""
        row = self.rows.get(label)
        return None if row is None else getattr(row, shell)

    @property
    def high_resolution(self) -> float | None:
        """Overall high-resolution limit (Å)."""
        return self.value("High resolution limit")

    @property
    def rmeas(self) -> float | None:
        """Overall R_meas (all I+ and I-)."""
        return self.value("Rmeas (all I+ & I-)")

    @property
    def cc_half(self) -> float | None:
        """Overall CC½."""
        return self.value("Mn(I) half-set correlation CC(1/2)")

    @property
    def mean_i_over_sigma(self) -> float | None:
        """Overall ⟨I/σ(I)⟩."""
        return self.value("Mean((I)/sd(I))")

    @property
    def completeness(self) -> float | None:
        """Overall completeness (%)."""
        return self.value("Completeness")

    @property
    def multiplicity(self) -> float | None:
        """Overall multiplicity."""
        return self.value("Multiplicity")


@dataclass(frozen=True)
class AimlessRunResult:
    """Paths and parsed summary from an Aimless run."""

    hklin: Path
    hklout: Path
    log: Path
    summary: AimlessSummary
    command: str


def find_aimless() -> Path | None:
    """Locate the ``aimless`` executable on PATH, ``$CBIN``, or CCP4 prefixes.

    :returns: Absolute path to ``aimless``, or ``None`` if not found.
    """
    which = shutil.which("aimless")
    if which:
        return Path(which).expanduser().resolve()
    prefixes: list[Path] = []
    if cbin := os.environ.get("CBIN", "").strip():
        prefixes.append(Path(cbin).expanduser())
    if ccp4 := os.environ.get("CCP4", "").strip():
        prefixes.append(Path(ccp4).expanduser() / "bin")
    prefixes += [Path("/Applications/ccp4-9/bin"), Path("/Applications/ccp4-8/bin"), Path.home() / "ccp4" / "bin"]
    for prefix in prefixes:
        candidate = prefix / "aimless"
        try:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate.resolve()
        except OSError:
            continue
    return None


def _float_or_none(token: str) -> float | None:
    try:
        return float(token)
    except ValueError:
        return None


def parse_aimless_summary(text: str) -> AimlessSummary:
    """Parse the first ``SUMMARY_BEGIN`` … ``SUMMARY_END`` block of an Aimless log.

    :param text: The log text.
    :returns: The summary (empty if the log has none, e.g. a failed run).
    """
    begin = text.find("<!--SUMMARY_BEGIN-->")
    if begin < 0:
        return AimlessSummary()
    end = text.find("<!--SUMMARY_END-->", begin)
    block = text[begin:end if end > 0 else len(text)]

    rows: dict[str, ShellStatistics] = {}
    estimates: dict[str, float] = {}
    space_group: str | None = None
    cell: tuple[float, ...] | None = None
    in_table = False
    in_overall_limits = False
    for line in block.splitlines():
        stripped = line.strip()
        if "Overall" in line and "InnerShell" in line and "OuterShell" in line:
            in_table = True
            continue
        if stripped.startswith("Space group:"):
            space_group = stripped.split(":", 1)[1].strip() or None
            continue
        if stripped.startswith("Average unit cell:"):
            vals = [_float_or_none(t) for t in stripped.split(":", 1)[1].split()]
            if len(vals) == 6 and all(v is not None for v in vals):
                cell = tuple(vals)  # type: ignore[assignment]
            continue
        if stripped.startswith("Estimates of resolution limits"):
            in_table = False
            in_overall_limits = "overall" in stripped
            continue
        if in_overall_limits:
            if (m := _LIMIT.search(stripped)) is not None:
                estimates.setdefault(m.group(1).strip(), float(m.group(2)))
            elif stripped:
                in_overall_limits = stripped.startswith("from")
            continue
        if not in_table or not stripped:
            continue
        tokens = stripped.split()
        if len(tokens) < 4:
            continue
        values = tokens[-3:]
        if not all(_float_or_none(t) is not None or t == "-" for t in values):
            # Rows with a single value (e.g. "Rmerge in top intensity bin 0.057 - -")
            # still end in three tokens; anything else is prose.
            continue
        label = " ".join(tokens[:-3])
        rows.setdefault(label, ShellStatistics(*(_float_or_none(t) for t in values)))
    return AimlessSummary(rows=rows, resolution_estimates=estimates, space_group=space_group,
                          cell=cell)  # type: ignore[arg-type]


def aimless_keywords(*, resolution_high: float | None = None, anomalous: bool = False,
                     custom: str = "") -> list[str]:
    """Keyword lines for an Aimless run.

    :param resolution_high: High-resolution cutoff (Å), or ``None`` for all data.
    :param anomalous: Keep I+ and I- separate (``ANOMALOUS ON``).
    :param custom: Extra keyword line(s).
    :returns: The heredoc lines (never empty).
    """
    lines: list[str] = []
    if resolution_high:
        lines.append(f"RESOLUTION HIGH {resolution_high:g}")
    if anomalous:
        lines.append("ANOMALOUS ON")
    lines += [ln for ln in (custom or "").strip().splitlines() if ln.strip()]
    return lines or ["! DiffractionLab Aimless (defaults)"]


def run_aimless(
    hklin: Path,
    hklout: Path,
    *,
    work_dir: Path | None = None,
    resolution_high: float | None = None,
    anomalous: bool = False,
    custom: str = "",
    job_service: Any = None,
) -> AimlessRunResult:
    """Scale and merge *hklin* (normally Pointless output) into *hklout*.

    :param hklin: Unmerged MTZ, after Pointless.
    :param hklout: Merged output MTZ.
    :param work_dir: Directory for the log (default: *hklout*'s).
    :param resolution_high: High-resolution cutoff (Å), or ``None``.
    :param anomalous: ``ANOMALOUS ON``.
    :param custom: Extra keyword line(s).
    :param job_service: Submit as a SchedArray job to this
        :class:`~diffraction_lab.jobs.service.JobService` (``None``: run here).
    :returns: Paths plus the parsed summary.
    :raises FileNotFoundError: If Aimless is missing or HKLIN does not exist.
    :raises RuntimeError: If Aimless fails or writes no summary.
    """
    hklin = Path(hklin).expanduser().resolve()
    hklout = Path(hklout).expanduser().resolve()
    if not hklin.is_file():
        raise FileNotFoundError(f"HKLIN not found: {hklin}")
    exe = find_aimless()
    if exe is None:
        raise FileNotFoundError(_AIMLESS_MISSING)
    cwd = Path(work_dir).expanduser().resolve() if work_dir else hklout.parent
    cwd.mkdir(parents=True, exist_ok=True)
    hklout.parent.mkdir(parents=True, exist_ok=True)

    log_path = cwd / f"{datetime.now():%y%m%d-%H%M%S}-aimless.log"
    heredoc = "\n".join(aimless_keywords(resolution_high=resolution_high, anomalous=anomalous,
                                         custom=custom))
    program = f"{shlex.quote(str(exe))} HKLIN {shlex.quote(str(hklin))} HKLOUT {shlex.quote(str(hklout))}"
    if job_service is not None:
        # The job's output file is the log; Aimless's directory goes on PATH.
        command = (f"export PATH={shlex.quote(str(exe.parent))}:\"$PATH\"; "
                   f"{program} << EOF\n{heredoc}\nEOF")
        logger.info("Running Aimless (SchedArray) in %s:\n%s", cwd, command)
        status = job_service.submit_and_wait(command, working_dir=cwd, job_name="aimless", output_file=log_path,
                                             error_file=log_path.with_suffix(".err"), tool="aimless",
                                             step="scale", max_retries=0, poll_interval=0.5)
        returncode = status.get("return_code")
        returncode = int(returncode) if returncode is not None else (
            0 if str(status.get("state", "")).lower() == "completed" else 1)
        stderr = ""
    else:
        command = f"{program} << EOF >> {shlex.quote(log_path.name)}\n{heredoc}\nEOF"
        env = os.environ.copy()
        if str(exe.parent) not in env.get("PATH", "").split(os.pathsep):
            env["PATH"] = f"{exe.parent}{os.pathsep}{env.get('PATH', '')}"
        logger.info("Running Aimless in %s:\n%s", cwd, command)
        completed = subprocess.run(command, shell=True, cwd=str(cwd), env=env, text=True,
                                   capture_output=True, check=False)
        returncode, stderr = completed.returncode, completed.stderr or ""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        text = ""
    summary = parse_aimless_summary(text)
    if returncode != 0 or not summary.rows:
        tail = "\n".join(text.strip().splitlines()[-5:]) or stderr.strip()
        raise RuntimeError(f"Aimless failed (exit {returncode}): {tail or command}")
    return AimlessRunResult(hklin=hklin, hklout=hklout, log=log_path, summary=summary, command=command)


@dataclass(frozen=True)
class ScaleMergeResult:
    """Pointless then Aimless on one unmerged MTZ."""

    pointless: PointlessRunResult
    aimless: AimlessRunResult


def scale_and_merge(
    hklin: Path,
    work_dir: Path,
    *,
    resolution_high: float | None = None,
    anomalous: bool = False,
) -> ScaleMergeResult:
    """Determine symmetry with Pointless, then scale and merge with Aimless.

    Writes ``pointless.mtz``, ``aimless.mtz`` and both logs into *work_dir*
    (for a ProcessingLab run, its ``ccp4/`` subdirectory).

    :param hklin: Unmerged MTZ (e.g. ProcessingLab's ``process.mtz``).
    :param work_dir: Output directory (created if needed).
    :param resolution_high: Aimless high-resolution cutoff (Å), or ``None``.
    :param anomalous: Aimless ``ANOMALOUS ON``.
    :returns: Both results.
    :raises FileNotFoundError: If Pointless, Aimless or HKLIN is missing.
    :raises RuntimeError: If either program fails.
    """
    from diffraction_lab.pointless_runner import run_pointless

    work_dir = Path(work_dir).expanduser()
    work_dir.mkdir(parents=True, exist_ok=True)
    pointless = run_pointless(Path(hklin), work_dir / "pointless.mtz", work_dir=work_dir,
                              choose_space_group=True)
    aimless = run_aimless(pointless.hklout, work_dir / "aimless.mtz", work_dir=work_dir,
                          resolution_high=resolution_high, anomalous=anomalous)
    return ScaleMergeResult(pointless=pointless, aimless=aimless)
