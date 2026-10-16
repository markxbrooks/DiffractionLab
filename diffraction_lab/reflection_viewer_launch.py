"""Launch DiffractionLab Reflection File Viewer on an MTZ."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_IMPORT = (
    "from diffraction_lab.ui.widgets.reflections.window import ReflectionViewerWindow"
)


def _repo_root() -> Path:
    """Return the DiffractionLab repository root (parent of the package)."""
    # …/DiffractionLab/diffraction_lab/reflection_viewer_launch.py → DiffractionLab
    return Path(__file__).resolve().parents[1]


def _python_candidates() -> list[Path]:
    """Interpreters that might import DiffractionLab + PySide6."""
    candidates: list[Path] = []
    env = os.environ.get("ELMO_PYTHON") or os.environ.get("ELMO")
    if env:
        candidate = Path(env).expanduser()
        if candidate.is_file():
            candidates.append(candidate)
        venv_py = candidate / ".venv" / "bin" / "python"
        if venv_py.is_file():
            candidates.append(venv_py)

    # Running interpreter (DiffractionLab itself).
    candidates.append(Path(sys.executable))

    projects = _repo_root().parent
    for name in ("ElMo", "elmo", "DiffractionLab"):
        venv_py = projects / name / ".venv" / "bin" / "python"
        if venv_py.is_file():
            candidates.append(venv_py)
    return candidates


def _env_with_repo(base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment with DiffractionLab on ``PYTHONPATH``."""
    env = dict(base or os.environ)
    root = str(_repo_root())
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root if not existing else f"{root}{os.pathsep}{existing}"
    return env


def resolve_viewer_python() -> Path | None:
    """Return an interpreter that can import the Reflection File Viewer."""
    env = _env_with_repo()
    seen: set[Path] = set()
    for python in _python_candidates():
        try:
            resolved = python.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        probe = subprocess.run(
            [str(resolved), "-c", _IMPORT],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        if probe.returncode == 0:
            return resolved
    return None


def launch_reflection_viewer(mtz_path: Path | str) -> subprocess.Popen:
    """Start Reflection File Viewer in a detached process and load *mtz_path*.

    :param mtz_path: Path to an MTZ (or SF-CIF) file.
    :returns: Started process handle.
    :raises FileNotFoundError: If the MTZ or a suitable interpreter is missing.
    :raises RuntimeError: If the viewer cannot be imported.
    """
    mtz = Path(mtz_path).expanduser().resolve()
    if not mtz.is_file():
        raise FileNotFoundError(f"MTZ not found: {mtz}")

    python = resolve_viewer_python()
    if python is None:
        raise FileNotFoundError(
            "Could not find a Python that imports "
            "diffraction_lab.ui.widgets.reflections.window "
            "(try ElMo/.venv or set ELMO_PYTHON)."
        )

    env = _env_with_repo()
    probe = subprocess.run(
        [str(python), "-c", _IMPORT],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout or "").strip() or "unknown import error"
        raise RuntimeError(
            f"Reflection File Viewer is not importable with {python}:\n{detail}"
        )

    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from PySide6.QtWidgets import QApplication\n"
        f"{_IMPORT}\n"
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
        env=env,
    )
