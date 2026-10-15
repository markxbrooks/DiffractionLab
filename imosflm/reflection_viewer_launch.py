"""Launch ElMo Reflection File Viewer on an MTZ produced by DiffractionLab."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def resolve_elmo_python() -> Path | None:
    """Return an interpreter that can import ``elmo``, if one can be found."""
    env = os.environ.get("ELMO_PYTHON") or os.environ.get("ELMO")
    if env:
        candidate = Path(env).expanduser()
        if candidate.is_file():
            return candidate
        venv_py = candidate / ".venv" / "bin" / "python"
        if venv_py.is_file():
            return venv_py

    here = Path(__file__).resolve()
    projects = here.parents[2]  # …/projects/DiffractionLab/imosflm → projects
    for name in ("ElMo", "elmo"):
        venv_py = projects / name / ".venv" / "bin" / "python"
        if venv_py.is_file():
            return venv_py
    return None


def launch_reflection_viewer(mtz_path: Path | str) -> subprocess.Popen:
    """Start Reflection File Viewer in a detached process and load *mtz_path*.

    :param mtz_path: Path to an MTZ (or SF-CIF) file.
    :returns: Started process handle.
    :raises FileNotFoundError: If the MTZ or a suitable ElMo interpreter is missing.
    :raises RuntimeError: If ElMo cannot be imported with the resolved interpreter.
    """
    mtz = Path(mtz_path).expanduser().resolve()
    if not mtz.is_file():
        raise FileNotFoundError(f"MTZ not found: {mtz}")

    python = resolve_elmo_python()
    if python is None:
        raise FileNotFoundError(
            "Could not find ElMo (.venv/bin/python next to DiffractionLab, "
            "or set ELMO_PYTHON / ELMO)."
        )

    probe = subprocess.run(
        [
            str(python),
            "-c",
            "from elmo.ui.widgets.reflections.window import ReflectionViewerWindow",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout or "").strip() or "unknown import error"
        raise RuntimeError(
            f"ElMo Reflection File Viewer is not importable with {python}:\n{detail}"
        )

    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from PySide6.QtWidgets import QApplication\n"
        "from elmo.ui.widgets.reflections.window import ReflectionViewerWindow\n"
        "app = QApplication(sys.argv)\n"
        "win = ReflectionViewerWindow()\n"
        f"ok = win.load_path(Path({str(mtz)!r}))\n"
        "if not ok:\n"
        "    raise SystemExit(1)\n"
        "win.show()\n"
        "raise SystemExit(app.exec())\n"
    )
    return subprocess.Popen(
        [str(python), "-c", code],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
