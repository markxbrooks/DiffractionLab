"""Load / save DiffractionLab Session JSON (``.dlsession``)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from diffraction_lab.session.model import DiffractionLabSession
from diffraction_lab.session.serialize import session_from_dict, session_to_dict

SESSION_SUFFIX = ".dlsession"


def is_session_path(path: Path | str) -> bool:
    """Return True if *path* looks like a DiffractionLab session file."""
    p = Path(path)
    if p.suffix.lower() == SESSION_SUFFIX:
        return True
    if p.suffix.lower() == ".json":
        return True
    return False


def relativize_path(
    path: Path | str | None,
    *,
    base_dir: Path,
) -> str | None:
    """Return a path string relative to *base_dir* when under that tree.

    Absolute strings are kept when the path is outside *base_dir* or missing.
    """
    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return None
    try:
        absolute = Path(text).expanduser().resolve()
    except OSError:
        return text
    try:
        base = base_dir.expanduser().resolve()
        relative = absolute.relative_to(base)
        return str(relative)
    except (ValueError, OSError):
        return str(absolute)


def resolve_path(
    path: str | None,
    *,
    base_dir: Path,
) -> Path | None:
    """Resolve a stored path against *base_dir* (session file parent).

    Relative paths are joined to *base_dir*; absolute paths are returned as-is.
    """
    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).expanduser()
    try:
        return candidate.resolve()
    except OSError:
        return candidate


def relativize_session(
    session: DiffractionLabSession,
    *,
    base_dir: Path,
) -> DiffractionLabSession:
    """Return a copy of *session* with path fields relativized to *base_dir*."""
    viewer = session.viewer
    viewer_copy = type(viewer)(
        image_path=relativize_path(viewer.image_path, base_dir=base_dir),
        dataset_name=viewer.dataset_name,
        frame_index=viewer.frame_index,
        show_rings=viewer.show_rings,
        show_ice_rings=viewer.show_ice_rings,
        show_beamstop=viewer.show_beamstop,
        show_masked_pixels=viewer.show_masked_pixels,
        show_indexing_spots=viewer.show_indexing_spots,
        image_inverted=viewer.image_inverted,
        contrast=viewer.contrast,
    )
    processing = session.processing
    processing_copy = None
    if processing is not None:
        processing_copy = type(processing)(
            backend=processing.backend,
            master_path=relativize_path(processing.master_path, base_dir=base_dir),
            work_dir=relativize_path(processing.work_dir, base_dir=base_dir),
            dials_settings=dict(processing.dials_settings),
            mosflm=dict(processing.mosflm),
            integrate_first=processing.integrate_first,
            integrate_last=processing.integrate_last,
            pointless=dict(processing.pointless),
            mtz_path=relativize_path(processing.mtz_path, base_dir=base_dir),
            pointless_hklin=relativize_path(
                processing.pointless_hklin, base_dir=base_dir
            ),
            pointless_hklout=relativize_path(
                processing.pointless_hklout, base_dir=base_dir
            ),
            last_dials_log=relativize_path(
                processing.last_dials_log, base_dir=base_dir
            ),
            pipeline_tab=processing.pipeline_tab,
        )
    return DiffractionLabSession(
        version=session.version,
        format=session.format,
        created=session.created,
        viewer=viewer_copy,
        processing=processing_copy,
    )


def resolve_session_paths(
    session: DiffractionLabSession,
    *,
    base_dir: Path,
) -> DiffractionLabSession:
    """Return a copy with path fields resolved against *base_dir*."""

    def _s(path: str | None) -> str | None:
        resolved = resolve_path(path, base_dir=base_dir)
        return str(resolved) if resolved is not None else None

    viewer = session.viewer
    viewer_copy = type(viewer)(
        image_path=_s(viewer.image_path),
        dataset_name=viewer.dataset_name,
        frame_index=viewer.frame_index,
        show_rings=viewer.show_rings,
        show_ice_rings=viewer.show_ice_rings,
        show_beamstop=viewer.show_beamstop,
        show_masked_pixels=viewer.show_masked_pixels,
        show_indexing_spots=viewer.show_indexing_spots,
        image_inverted=viewer.image_inverted,
        contrast=viewer.contrast,
    )
    processing = session.processing
    processing_copy = None
    if processing is not None:
        processing_copy = type(processing)(
            backend=processing.backend,
            master_path=_s(processing.master_path),
            work_dir=_s(processing.work_dir),
            dials_settings=dict(processing.dials_settings),
            mosflm=dict(processing.mosflm),
            integrate_first=processing.integrate_first,
            integrate_last=processing.integrate_last,
            pointless=dict(processing.pointless),
            mtz_path=_s(processing.mtz_path),
            pointless_hklin=_s(processing.pointless_hklin),
            pointless_hklout=_s(processing.pointless_hklout),
            last_dials_log=_s(processing.last_dials_log),
            pipeline_tab=processing.pipeline_tab,
        )
    return DiffractionLabSession(
        version=session.version,
        format=session.format,
        created=session.created,
        viewer=viewer_copy,
        processing=processing_copy,
    )


def load_session(path: Path | str) -> DiffractionLabSession:
    """Load a Session from *path* and resolve relative paths.

    :param path: Path to a ``.dlsession`` or DiffractionLab session JSON file.
    :returns: Session with absolute path fields where possible.
    :raises FileNotFoundError: If *path* does not exist.
    :raises ValueError: If the JSON is not a valid Session document.
    """
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"Session file not found: {file_path}")
    with open(file_path, encoding="utf-8") as handle:
        data = json.load(handle)
    session = session_from_dict(data)
    return resolve_session_paths(session, base_dir=file_path.parent)


def save_session(path: Path | str, session: DiffractionLabSession) -> Path:
    """Write *session* to *path*, relativizing paths under the parent dir.

    :param path: Destination ``.dlsession`` (or ``.json``) path.
    :param session: Document to write.
    :returns: Resolved destination path.
    """
    file_path = Path(path).expanduser()
    if file_path.suffix.lower() not in (SESSION_SUFFIX, ".json"):
        file_path = file_path.with_suffix(SESSION_SUFFIX)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    base_dir = file_path.parent.resolve()

    to_write = relativize_session(session, base_dir=base_dir)
    if not to_write.created:
        to_write.created = datetime.now(timezone.utc).isoformat()
    payload = session_to_dict(to_write)
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    return file_path.resolve()
