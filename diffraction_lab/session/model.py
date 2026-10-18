"""DiffractionLab Session document model (schema v1).

A Session is the native JSON document for Save/Open — viewer image prefs plus
Processing pipeline paths and settings. Artifact binaries stay on disk; only
path pointers are stored. On Open, Processing rehydrates Find spots through
Scale from files under ``work_dir`` when present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SESSION_FORMAT = "diffractionlab.session"
SESSION_VERSION = 1


@dataclass
class ViewerState:
    """Viewer / main-window state persisted in a Session."""

    image_path: str | None = None
    dataset_name: str | None = None
    frame_index: int = 0
    show_rings: bool = True
    show_indexing_spots: bool = True
    image_inverted: bool = False
    contrast: int = 100


@dataclass
class ProcessingState:
    """Processing dialog state persisted in a Session."""

    backend: str = "mosflm"  # "dials" | "mosflm" | "processinglab"
    master_path: str | None = None
    work_dir: str | None = None
    dials_settings: dict[str, Any] = field(default_factory=dict)
    mosflm: dict[str, Any] = field(default_factory=dict)
    integrate_first: int = 1
    integrate_last: int = 40
    pointless: dict[str, Any] = field(default_factory=dict)
    mtz_path: str | None = None
    pointless_hklin: str | None = None
    pointless_hklout: str | None = None
    last_dials_log: str | None = None
    pipeline_tab: int = 0


@dataclass
class DiffractionLabSession:
    """Top-level DiffractionLab Session document."""

    version: int = SESSION_VERSION
    format: str = SESSION_FORMAT
    created: str | None = None
    viewer: ViewerState = field(default_factory=ViewerState)
    processing: ProcessingState | None = None
