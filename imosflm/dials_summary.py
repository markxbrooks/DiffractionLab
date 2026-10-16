"""Parse DIALS ``Summary vs resolution`` tables from integrate logs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResolutionBinSummary:
    """One resolution shell from DIALS integrate ``Summary vs resolution``."""

    experiment_id: int
    d_min: float
    n_full: int
    n_part: int
    n_over: int
    n_ice: int
    n_sum: int
    n_prf: int
    ibg: float
    i_sigi_sum: float
    i_sigi_prf: float
    cc_prf: float
    rmsd_xy: float


_SECTION = re.compile(r"^\s*Summary vs resolution\s*$", re.IGNORECASE)
_DATA_ROW = re.compile(
    r"^\|\s*"
    r"(-?\d+)\s*\|\s*"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\|\s*"
    r"(-?\d+)\s*\|\s*"
    r"(-?\d+)\s*\|\s*"
    r"(-?\d+)\s*\|\s*"
    r"(-?\d+)\s*\|\s*"
    r"(-?\d+)\s*\|\s*"
    r"(-?\d+)\s*\|\s*"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\|\s*"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\|\s*"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\|\s*"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\|\s*"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\|"
    r"\s*$"
)


def parse_summary_vs_resolution_text(text: str) -> list[ResolutionBinSummary]:
    """Parse the last ``Summary vs resolution`` table in *text*.

    :param text: Contents of a ``dials.integrate.log`` (or similar).
    :returns: Resolution bins (high→low ``d_min`` as written by DIALS), or ``[]``.
    """
    lines = text.splitlines()
    # Prefer the last section (full pipeline may emit the table twice).
    starts = [i for i, line in enumerate(lines) if _SECTION.match(line)]
    if not starts:
        return []
    start = starts[-1]
    rows: list[ResolutionBinSummary] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if stripped.startswith("+") and rows:
            break
        match = _DATA_ROW.match(line)
        if not match:
            continue
        rows.append(
            ResolutionBinSummary(
                experiment_id=int(match.group(1)),
                d_min=float(match.group(2)),
                n_full=int(match.group(3)),
                n_part=int(match.group(4)),
                n_over=int(match.group(5)),
                n_ice=int(match.group(6)),
                n_sum=int(match.group(7)),
                n_prf=int(match.group(8)),
                ibg=float(match.group(9)),
                i_sigi_sum=float(match.group(10)),
                i_sigi_prf=float(match.group(11)),
                cc_prf=float(match.group(12)),
                rmsd_xy=float(match.group(13)),
            )
        )
    return rows


def parse_summary_vs_resolution_log(
    path: Path | str,
) -> list[ResolutionBinSummary]:
    """Parse ``Summary vs resolution`` from a DIALS integrate logfile."""
    log_path = Path(path).expanduser()
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return parse_summary_vs_resolution_text(text)


def find_integrate_log(work_dir: Path | str) -> Path | None:
    """Return ``dials.integrate.log`` under *work_dir* if present."""
    candidate = Path(work_dir).expanduser() / "dials.integrate.log"
    return candidate if candidate.is_file() else None
