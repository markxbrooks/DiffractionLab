"""Job metadata helpers for DiffractionLab SchedArray submissions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional


def build_job_metadata(
    *,
    tool: str,
    step: str,
    work_dir: Path | str | None = None,
    log_file: Path | str | None = None,
    session_id: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a metadata dict stored on a SchedArray ``job_queue`` row.

    :param tool: Tool family (``dials``, ``pointless``, ``mosflm``).
    :param step: Step name (e.g. ``find_spots``, ``index``).
    :param work_dir: Processing work directory.
    :param log_file: Primary stdout log path.
    :param session_id: Optional DiffractionLab session identifier.
    :param extra: Additional key/value pairs merged into the result.
    :returns: JSON-serializable metadata dictionary.
    """
    meta: dict[str, Any] = {
        "app": "diffraction_lab",
        "tool": str(tool).lower(),
        "step": str(step),
    }
    if work_dir is not None:
        meta["work_dir"] = str(Path(work_dir))
    if log_file is not None:
        meta["log_file"] = str(Path(log_file))
    if session_id:
        meta["session_id"] = str(session_id)
    if extra:
        meta.update(dict(extra))
    return meta


def metadata_tool(job: Mapping[str, Any]) -> Optional[str]:
    """Return the ``tool`` field from a job dict's metadata, if any."""
    meta = job.get("metadata") or {}
    if isinstance(meta, str):
        return None
    tool = meta.get("tool")
    return str(tool).lower() if tool else None


def metadata_log_file(job: Mapping[str, Any]) -> Optional[Path]:
    """Resolve the log file path from job metadata or ``output_file``."""
    meta = job.get("metadata") or {}
    if isinstance(meta, dict):
        log = meta.get("log_file")
        if log:
            return Path(str(log))
    out = job.get("output_file")
    if out and not str(out).startswith("("):
        path = Path(str(out))
        # SchedArray may store captured text in output_file when no path was set;
        # only treat as a path if it looks like one.
        if path.suffix or "/" in str(out) or "\\" in str(out):
            return path
    return None
