"""DIALS :class:`~picoui.specs.cli.SpecField` catalog for the Processing SpecEditor.

Maps :class:`~imosflm.dials_engine.DialsSettings` to editable CLI/phil fields and
back. Optional floats use ``0`` in the UI to mean “auto” (``None`` in settings).
"""

from __future__ import annotations

from typing import Any

from diffraction_lab.dials_engine import DialsSettings

try:
    from picoui.specs.cli import SpecField, SpecFieldKind, SpecValueType, emit_argv

    _HAS_SPEC = True
except ImportError:  # pragma: no cover - PicoUI optional
    SpecField = None  # type: ignore[misc, assignment]
    SpecFieldKind = None  # type: ignore[misc, assignment]
    SpecValueType = None  # type: ignore[misc, assignment]
    emit_argv = None  # type: ignore[misc, assignment]
    _HAS_SPEC = False


def _phil_int(key: str):
    def _fmt(value: Any) -> list[str]:
        if value is None or value == "":
            return []
        return [f"{key}={int(value)}"]

    return _fmt


def _phil_float_optional(key: str):
    """Emit ``key=value`` only when *value* is a positive float."""

    def _fmt(value: Any) -> list[str]:
        if value is None or value == "":
            return []
        number = float(value)
        if number <= 0:
            return []
        return [f"{key}={number:g}"]

    return _fmt


def _phil_str(key: str):
    def _fmt(value: Any) -> list[str]:
        text = str(value).strip() if value is not None else ""
        if not text:
            return []
        return [f"{key}={text}"]

    return _fmt


def dials_settings_fields() -> list:
    """Return SpecFields for the DiffractionLab DIALS settings panel."""
    if not _HAS_SPEC:
        raise ImportError("picoui is required for dials_settings_fields()")

    return [
        SpecField(
            name="nproc",
            kind=SpecFieldKind.OPTION,
            label="nproc",
            help="Number of parallel processes (nproc=N)",
            token="",
            value_type=SpecValueType.INT,
            default=4,
            group="Parameters",
            formatter=_phil_int("nproc"),
        ),
        SpecField(
            name="min_spot_size",
            kind=SpecFieldKind.OPTION,
            label="min spot size",
            help="spotfinder.filter.min_spot_size",
            token="",
            value_type=SpecValueType.INT,
            default=3,
            group="Parameters",
            formatter=_phil_int("spotfinder.filter.min_spot_size"),
        ),
        SpecField(
            name="d_min",
            kind=SpecFieldKind.OPTION,
            label="d_min (Å)",
            help="prediction.d_min — 0 means auto",
            token="",
            value_type=SpecValueType.FLOAT,
            default=0.0,
            group="Parameters",
            formatter=_phil_float_optional("prediction.d_min"),
        ),
        SpecField(
            name="gain",
            kind=SpecFieldKind.OPTION,
            label="gain",
            help="spotfinder.threshold.dispersion.gain — 0 means auto",
            token="",
            value_type=SpecValueType.FLOAT,
            default=0.0,
            group="Parameters",
            formatter=_phil_float_optional("spotfinder.threshold.dispersion.gain"),
        ),
        SpecField(
            name="space_group",
            kind=SpecFieldKind.OPTION,
            label="Known space group",
            help="indexing.known_symmetry.space_group (optional)",
            token="",
            value_type=SpecValueType.STR,
            default=None,
            group="Parameters",
            formatter=_phil_str("indexing.known_symmetry.space_group"),
        ),
        SpecField(
            name="unit_cell",
            kind=SpecFieldKind.OPTION,
            label="Known unit cell",
            help="indexing.known_symmetry.unit_cell — a,b,c,α,β,γ (optional)",
            token="",
            value_type=SpecValueType.STR,
            default=None,
            group="Parameters",
            formatter=_phil_str("indexing.known_symmetry.unit_cell"),
        ),
        SpecField(
            name="scan_varying",
            kind=SpecFieldKind.FLAG,
            label="Scan-varying refine",
            help="Pass scan_varying=True to dials.refine (long wedges)",
            token="scan_varying=True",
            value_type=SpecValueType.BOOL,
            default=False,
            group="Parameters",
        ),
    ]


def dials_settings_to_values(settings: DialsSettings) -> dict[str, Any]:
    """Convert :class:`DialsSettings` to SpecEditor values (0 = auto for floats)."""
    return {
        "nproc": int(settings.nproc),
        "min_spot_size": int(settings.min_spot_size),
        "d_min": float(settings.d_min) if settings.d_min is not None else 0.0,
        "gain": float(settings.gain) if settings.gain is not None else 0.0,
        "space_group": settings.space_group or "",
        "unit_cell": settings.unit_cell or "",
        "scan_varying": bool(getattr(settings, "scan_varying", False)),
    }


def values_to_dials_settings(values: dict[str, Any]) -> DialsSettings:
    """Build :class:`DialsSettings` from SpecEditor ``values()``."""
    d_min_val = float(values.get("d_min") or 0)
    gain_val = float(values.get("gain") or 0)
    sg = str(values.get("space_group") or "").strip() or None
    cell = str(values.get("unit_cell") or "").strip() or None
    return DialsSettings(
        nproc=max(1, int(values.get("nproc") or 4)),
        min_spot_size=max(1, int(values.get("min_spot_size") or 3)),
        gain=gain_val if gain_val > 0 else None,
        d_min=d_min_val if d_min_val > 0 else None,
        space_group=sg,
        unit_cell=cell,
        scan_varying=bool(values.get("scan_varying", False)),
    )


def dials_settings_phil_argv(settings: DialsSettings) -> list[str]:
    """Phil tokens that :class:`DialsEngine` would pass for *settings* (no scan_range)."""
    if not _HAS_SPEC or emit_argv is None:
        # Fallback without PicoUI
        args = [
            f"nproc={settings.nproc}",
            f"spotfinder.filter.min_spot_size={settings.min_spot_size}",
        ]
        if settings.gain is not None:
            args.append(f"spotfinder.threshold.dispersion.gain={settings.gain:g}")
        if settings.d_min is not None:
            args.append(f"prediction.d_min={settings.d_min:g}")
        if settings.space_group:
            args.append(f"indexing.known_symmetry.space_group={settings.space_group}")
        if settings.unit_cell:
            args.append(f"indexing.known_symmetry.unit_cell={settings.unit_cell}")
        return args
    return emit_argv(dials_settings_fields(), dials_settings_to_values(settings))
