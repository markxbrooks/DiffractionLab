"""Run CCP4 Pointless (prefer MxLib ``PointlessCommand``, else subprocess).

Soft-imports MxLib so DiffractionLab still loads when MxLib or its optional
dependencies are absent; :func:`find_pointless` gates the Run button.
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_POINTLESS_MISSING = (
    "CCP4 Pointless not found. Install CCP4, source ccp4.setup-sh, "
    "or set $CBIN / ensure `pointless` is on PATH."
)


@dataclass(frozen=True)
class PointlessRunResult:
    """Paths and parsed fields from a Pointless run."""

    hklin: Path
    hklout: Path
    log: Path
    space_group: str | None
    reindex_operator: str | None
    confidence: float | None
    command: str


def find_pointless() -> Path | None:
    """Locate the ``pointless`` executable on PATH, ``$CBIN``, or CCP4 prefixes.

    :returns: Absolute path to ``pointless``, or ``None`` if not found.
    """
    which = shutil.which("pointless")
    if which:
        return Path(which).expanduser().resolve()

    cbin = os.environ.get("CBIN", "").strip()
    if cbin:
        candidate = Path(cbin).expanduser() / "pointless"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()

    ccp4 = os.environ.get("CCP4", "").strip()
    prefixes: list[Path] = []
    if ccp4:
        prefixes.append(Path(ccp4).expanduser() / "bin")
    prefixes.extend(
        [
            Path("/Applications/ccp4-9/bin"),
            Path("/Applications/ccp4-8/bin"),
            Path.home() / "ccp4" / "bin",
        ]
    )
    for prefix in prefixes:
        candidate = prefix / "pointless"
        try:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate.resolve()
        except OSError:
            continue
    return None


def mxlib_import_error() -> str | None:
    """Return an ImportError message if MxLib Pointless is unavailable, else ``None``."""
    try:
        from mxlib.domain.ccp4.wrappers.pointless.args import PointlessArgs  # noqa: F401
        from mxlib.domain.ccp4.wrappers.pointless.command import (  # noqa: F401
            PointlessCommand,
        )

        return None
    except ImportError as ex:
        return str(ex)


def mxlib_available() -> bool:
    """Return whether MxLib's Pointless wrapper can be imported."""
    return mxlib_import_error() is None


def parse_pointless_log(log_path: Path) -> dict[str, Any]:
    """Parse space group / reindex / confidence from a Pointless log.

    Recognises CCP4 Pointless summary lines such as::

        Best Solution:    space group P 41 21 2
           Reindex operator:                   [h,k,l]
           Space group confidence:             0.931

    :param log_path: Path to the Pointless ``.log`` file.
    :returns: Dict with ``space_group``, ``reindex_operator``, ``confidence``.
    """
    results: dict[str, Any] = {
        "space_group": None,
        "reindex_operator": None,
        "confidence": None,
    }
    try:
        with open(log_path, encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                line = raw.rstrip("\n")
                lower = line.lower()
                if "best solution" in lower and "space group" in lower:
                    # "Best Solution:    space group P 41 21 2"
                    idx = lower.index("space group") + len("space group")
                    results["space_group"] = line[idx:].strip() or None
                elif "reindex operator" in lower and ":" in line:
                    # Prefer the indented Best Solution block (has "[h,k,l]").
                    value = line.split(":", 1)[1].strip()
                    if value:
                        results["reindex_operator"] = value
                elif (
                    "space group confidence" in lower
                    and "sqrt" not in lower
                    and ":" in line
                ):
                    # "   Space group confidence:             0.931"
                    # Skip the formula line that contains "Sqrt(...)".
                    token = line.split(":")[-1].strip().split()
                    if token:
                        try:
                            results["confidence"] = float(token[0])
                        except ValueError:
                            pass
    except OSError as ex:
        logger.warning("Failed to parse Pointless log %s: %s", log_path, ex)
    return results


def _keyword_lines(
    *,
    choose_space_group: bool,
    copy_free: bool,
    custom: str,
) -> list[str]:
    """Build Pointless heredoc keyword lines.

    Note: ``COPY FREE`` alone (or as the only mode-changing keyword) can make
    Pointless treat the HKLIN space group as known and skip Laue scoring.
    Prefer ``SETTING SYMMETRY-BASED`` for determination; enable COPY FREE only
    when Free-R propagation is required after a normal run.
    """
    lines: list[str] = []
    if choose_space_group:
        lines.append("SETTING SYMMETRY-BASED")
    if copy_free:
        lines.append("COPY FREE")
    custom_text = (custom or "").strip()
    if custom_text:
        lines.extend(custom_text.splitlines())
    # Empty heredoc still runs default Lauegroup mode; ensure at least a no-op
    # comment so shell ``<< EOF`` is valid when all flags are off.
    if not lines:
        lines.append("! DiffractionLab Pointless (defaults)")
    return lines


def _run_via_subprocess(
    exe: Path,
    hklin: Path,
    hklout: Path,
    cwd: Path,
    *,
    choose_space_group: bool,
    copy_free: bool,
    custom: str,
) -> PointlessRunResult:
    """Run Pointless with a shell heredoc (no MxLib)."""
    stamp = datetime.now().strftime("%y%m%d-%H%M%S")
    log_path = cwd / f"{stamp}-pointless.log"
    keywords = _keyword_lines(
        choose_space_group=choose_space_group,
        copy_free=copy_free,
        custom=custom,
    )
    heredoc = "\n".join(keywords)
    # Match MxLib's shell form: pointless HKLIN … HKLOUT … << EOF >> log
    command_str = (
        f"{shlex.quote(str(exe))} HKLIN {shlex.quote(str(hklin))} "
        f"HKLOUT {shlex.quote(str(hklout))} << EOF >> {shlex.quote(str(log_path.name))}\n"
        f"{heredoc}\n"
        f"EOF"
    )
    env = os.environ.copy()
    bin_dir = str(exe.parent)
    path_env = env.get("PATH", "")
    if bin_dir not in path_env.split(os.pathsep):
        env["PATH"] = f"{bin_dir}{os.pathsep}{path_env}"

    logger.info("Running Pointless (subprocess) in %s:\n%s", cwd, command_str)
    completed = subprocess.run(
        command_str,
        shell=True,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0 and not hklout.is_file():
        err = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(
            f"Pointless failed (exit {completed.returncode}). "
            f"{err or command_str}"
        )
    parsed = parse_pointless_log(log_path) if log_path.is_file() else {
        "space_group": None,
        "reindex_operator": None,
        "confidence": None,
    }
    return PointlessRunResult(
        hklin=hklin,
        hklout=hklout,
        log=log_path,
        space_group=parsed.get("space_group"),
        reindex_operator=parsed.get("reindex_operator"),
        confidence=parsed.get("confidence"),
        command=command_str,
    )


def _run_via_mxlib(
    hklin: Path,
    hklout: Path,
    cwd: Path,
    *,
    choose_space_group: bool,
    copy_free: bool,
    custom: str,
) -> PointlessRunResult:
    """Run Pointless via MxLib ``PointlessCommand``."""
    from mxlib.domain.ccp4.wrappers.pointless.args import PointlessArgs
    from mxlib.domain.ccp4.wrappers.pointless.command import PointlessCommand

    args = PointlessArgs(
        hklin=str(hklin),
        hklout=str(hklout),
        choose_space_group=bool(choose_space_group),
        copy_free=bool(copy_free),
        custom=(custom or "").strip(),
    )
    cmd = PointlessCommand()
    cmd.build_command(args)
    command_str = cmd.get_command()
    log_path = (cwd / Path(cmd.log).name).resolve()

    logger.info("Running Pointless (MxLib) in %s:\n%s", cwd, command_str)
    prev = os.getcwd()
    try:
        os.chdir(cwd)
        cmd.log = str(Path(cmd.log).name)
        result = cmd.run(requires_wsl=False)
    finally:
        os.chdir(prev)

    completed = getattr(result, "result", None)
    returncode = getattr(completed, "returncode", None)
    if returncode not in (0, None) and not hklout.is_file():
        stderr = getattr(completed, "stderr", "") or ""
        raise RuntimeError(
            f"Pointless failed (exit {returncode}). {stderr.strip() or command_str}"
        )

    if not log_path.is_file():
        alt = cwd / Path(cmd.log).name
        if alt.is_file():
            log_path = alt

    parsed = parse_pointless_log(log_path)
    return PointlessRunResult(
        hklin=hklin,
        hklout=hklout,
        log=log_path,
        space_group=parsed.get("space_group"),
        reindex_operator=parsed.get("reindex_operator"),
        confidence=parsed.get("confidence"),
        command=command_str,
    )


def run_pointless(
    hklin: Path,
    hklout: Path,
    *,
    work_dir: Path | None = None,
    choose_space_group: bool = True,
    copy_free: bool = False,
    custom: str = "",
) -> PointlessRunResult:
    """Run Pointless on *hklin* and write *hklout*.

    Prefers MxLib's :class:`PointlessCommand` when importable; otherwise runs
    the same shell heredoc via :mod:`subprocess`.

    :param hklin: Input reflection file.
    :param hklout: Output MTZ path.
    :param work_dir: Working directory for the subprocess / log.
    :param choose_space_group: Emit ``SETTING SYMMETRY-BASED``.
    :param copy_free: Emit ``COPY FREE`` (default off — can skip Laue scoring).
    :param custom: Optional extra keyword line(s).
    :returns: Paths plus parsed log fields.
    :raises FileNotFoundError: If Pointless is missing or HKLIN does not exist.
    :raises RuntimeError: If Pointless exits non-zero without producing HKLOUT.
    """
    hklin = Path(hklin).expanduser().resolve()
    hklout = Path(hklout).expanduser().resolve()
    if not hklin.is_file():
        raise FileNotFoundError(f"HKLIN not found: {hklin}")

    exe = find_pointless()
    if exe is None:
        raise FileNotFoundError(_POINTLESS_MISSING)

    cwd = Path(work_dir).expanduser().resolve() if work_dir else hklout.parent
    cwd.mkdir(parents=True, exist_ok=True)
    hklout.parent.mkdir(parents=True, exist_ok=True)

    bin_dir = str(exe.parent)
    path_env = os.environ.get("PATH", "")
    if bin_dir not in path_env.split(os.pathsep):
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{path_env}"

    opts = {
        "choose_space_group": bool(choose_space_group),
        "copy_free": bool(copy_free),
        "custom": (custom or "").strip(),
    }

    if mxlib_available():
        try:
            return _run_via_mxlib(hklin, hklout, cwd, **opts)
        except Exception as ex:  # noqa: BLE001
            logger.warning("MxLib Pointless failed (%s); falling back to subprocess", ex)

    return _run_via_subprocess(exe, hklin, hklout, cwd, **opts)
