"""Serialize / deserialize :class:`DiffractionLabSession` to plain dicts."""

from __future__ import annotations

from typing import Any

from diffraction_lab.session.model import (
    SESSION_FORMAT,
    SESSION_VERSION,
    DiffractionLabSession,
    ProcessingState,
    ViewerState,
)


def viewer_to_dict(state: ViewerState) -> dict[str, Any]:
    """Convert :class:`ViewerState` to a JSON-ready dict."""
    return {
        "image_path": state.image_path,
        "dataset_name": state.dataset_name,
        "frame_index": int(state.frame_index),
        "show_rings": bool(state.show_rings),
        "show_ice_rings": bool(state.show_ice_rings),
        "show_beamstop": bool(state.show_beamstop),
        "show_indexing_spots": bool(state.show_indexing_spots),
        "image_inverted": bool(state.image_inverted),
        "contrast": int(state.contrast),
    }


def viewer_from_dict(data: dict[str, Any] | None) -> ViewerState:
    """Build :class:`ViewerState` from a dict."""
    if not data:
        return ViewerState()
    return ViewerState(
        image_path=_optional_str(data.get("image_path")),
        dataset_name=_optional_str(data.get("dataset_name")),
        frame_index=int(data.get("frame_index") or 0),
        show_rings=bool(data.get("show_rings", True)),
        show_ice_rings=bool(data.get("show_ice_rings", False)),
        show_beamstop=bool(data.get("show_beamstop", False)),
        show_indexing_spots=bool(data.get("show_indexing_spots", True)),
        image_inverted=bool(data.get("image_inverted", False)),
        contrast=int(data.get("contrast") or 100),
    )


def processing_to_dict(state: ProcessingState) -> dict[str, Any]:
    """Convert :class:`ProcessingState` to a JSON-ready dict."""
    return {
        "backend": state.backend,
        "master_path": state.master_path,
        "work_dir": state.work_dir,
        "dials_settings": dict(state.dials_settings or {}),
        "mosflm": dict(state.mosflm or {}),
        "integrate_first": int(state.integrate_first),
        "integrate_last": int(state.integrate_last),
        "pointless": dict(state.pointless or {}),
        "mtz_path": state.mtz_path,
        "pointless_hklin": state.pointless_hklin,
        "pointless_hklout": state.pointless_hklout,
        "last_dials_log": state.last_dials_log,
        "pipeline_tab": int(state.pipeline_tab),
    }


def processing_from_dict(data: dict[str, Any] | None) -> ProcessingState | None:
    """Build :class:`ProcessingState` from a dict, or ``None`` if empty."""
    if not data:
        return None
    backend = str(data.get("backend") or "mosflm").strip().lower()
    if backend not in ("dials", "mosflm"):
        backend = "mosflm"
    return ProcessingState(
        backend=backend,
        master_path=_optional_str(data.get("master_path")),
        work_dir=_optional_str(data.get("work_dir")),
        dials_settings=dict(data.get("dials_settings") or {}),
        mosflm=dict(data.get("mosflm") or {}),
        integrate_first=int(data.get("integrate_first") or 1),
        integrate_last=int(data.get("integrate_last") or 40),
        pointless=dict(data.get("pointless") or {}),
        mtz_path=_optional_str(data.get("mtz_path")),
        pointless_hklin=_optional_str(data.get("pointless_hklin")),
        pointless_hklout=_optional_str(data.get("pointless_hklout")),
        last_dials_log=_optional_str(data.get("last_dials_log")),
        pipeline_tab=int(data.get("pipeline_tab") or 0),
    )


def session_to_dict(session: DiffractionLabSession) -> dict[str, Any]:
    """Convert a Session to a JSON-ready dict."""
    payload: dict[str, Any] = {
        "format": session.format or SESSION_FORMAT,
        "version": int(session.version or SESSION_VERSION),
        "created": session.created,
        "viewer": viewer_to_dict(session.viewer),
    }
    if session.processing is not None:
        payload["processing"] = processing_to_dict(session.processing)
    else:
        payload["processing"] = None
    return payload


def session_from_dict(data: dict[str, Any]) -> DiffractionLabSession:
    """Build a Session from a JSON dict.

    :raises ValueError: If the document is not a DiffractionLab session.
    """
    if not isinstance(data, dict):
        raise ValueError("Session document must be a JSON object")
    fmt = data.get("format")
    # Accept missing format only when version looks like ours (lenient load).
    if fmt is not None and fmt != SESSION_FORMAT:
        raise ValueError(
            f"Unsupported session format {fmt!r}; expected {SESSION_FORMAT!r}"
        )
    version = int(data.get("version") or SESSION_VERSION)
    if version > SESSION_VERSION:
        raise ValueError(
            f"Session schema version {version} is newer than "
            f"supported version {SESSION_VERSION}"
        )
    return DiffractionLabSession(
        version=version,
        format=SESSION_FORMAT,
        created=_optional_str(data.get("created")),
        viewer=viewer_from_dict(data.get("viewer")),
        processing=processing_from_dict(data.get("processing")),
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
