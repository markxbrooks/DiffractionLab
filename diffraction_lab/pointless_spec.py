"""Pointless :class:`~picoui.specs.cli.SpecField` catalog for the Symmetry SpecEditor.

Maps UI values to the MxLib :class:`~mxlib.domain.ccp4.wrappers.pointless.args.PointlessArgs`
fields used by :mod:`diffraction_lab.pointless_runner`.
"""

from __future__ import annotations

from typing import Any

try:
    from picoui.specs.cli import SpecField, SpecFieldKind, SpecValueType

    _HAS_SPEC = True
except ImportError:  # pragma: no cover - PicoUI optional
    SpecField = None  # type: ignore[misc, assignment]
    SpecFieldKind = None  # type: ignore[misc, assignment]
    SpecValueType = None  # type: ignore[misc, assignment]
    _HAS_SPEC = False


def pointless_settings_fields() -> list:
    """Return SpecFields for the DiffractionLab Pointless / Symmetry panel."""
    if not _HAS_SPEC:
        raise ImportError("picoui is required for pointless_settings_fields()")

    return [
        SpecField(
            name="choose_space_group",
            kind=SpecFieldKind.FLAG,
            label="Choose space group",
            help=(
                "Emit SETTING SYMMETRY-BASED so Pointless scores Laue / space groups"
            ),
            token="SETTING SYMMETRY-BASED",
            value_type=SpecValueType.BOOL,
            default=True,
            group="Pointless",
        ),
        SpecField(
            name="copy_free",
            kind=SpecFieldKind.FLAG,
            label="Copy Free-R flags",
            help=(
                "Emit COPY FREE. Leave off for first-pass symmetry: COPY FREE alone "
                "can make Pointless treat HKLIN space group as known and skip scoring."
            ),
            token="COPY FREE",
            value_type=SpecValueType.BOOL,
            default=False,
            group="Pointless",
        ),
        SpecField(
            name="custom",
            kind=SpecFieldKind.OPTION,
            label="Custom keywords",
            help="Optional extra Pointless keyword line(s)",
            token="",
            value_type=SpecValueType.STR,
            default="",
            group="Pointless",
            formatter=lambda value: (
                [str(value).strip()] if str(value or "").strip() else []
            ),
        ),
    ]


def pointless_defaults() -> dict[str, Any]:
    """Default SpecEditor values for Pointless."""
    return {
        "choose_space_group": True,
        "copy_free": False,
        "custom": "",
    }


def values_to_pointless_options(values: dict[str, Any]) -> dict[str, Any]:
    """Map SpecEditor ``values()`` to kwargs for :func:`run_pointless`."""
    return {
        "choose_space_group": bool(values.get("choose_space_group", True)),
        "copy_free": bool(values.get("copy_free", False)),
        "custom": str(values.get("custom") or "").strip(),
    }
