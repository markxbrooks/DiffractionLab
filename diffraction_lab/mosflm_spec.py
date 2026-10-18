"""Mosflm DPS :class:`~picoui.specs.cli.SpecField` catalog for Processing.

Covers the Mosflm-only numeric fields on the Setup tab (spot σ, DPS I/σI,
max penalty). Frame range widgets stay explicit so their maxima can be
updated when a dataset loads.
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


def mosflm_settings_fields() -> list:
    """Return SpecFields for Mosflm DPS spot / indexing thresholds."""
    if not _HAS_SPEC:
        raise ImportError("picoui is required for mosflm_settings_fields()")

    return [
        SpecField(
            name="spot_sigma",
            kind=SpecFieldKind.OPTION,
            label="Spot σ threshold",
            help="Spot-finding sigma threshold for Mosflm DPS",
            token="",
            value_type=SpecValueType.FLOAT,
            default=5.0,
            group="Mosflm DPS",
        ),
        SpecField(
            name="i_sig_i",
            kind=SpecFieldKind.OPTION,
            label="DPS I/σI",
            help="Minimum I/σI for spots used in DPS indexing",
            token="",
            value_type=SpecValueType.FLOAT,
            default=5.0,
            group="Mosflm DPS",
        ),
        SpecField(
            name="max_penalty",
            kind=SpecFieldKind.OPTION,
            label="Max penalty to refine",
            help="Skip lattice solutions with penalty above this value",
            token="",
            value_type=SpecValueType.FLOAT,
            default=50.0,
            group="Mosflm DPS",
        ),
    ]


def mosflm_settings_defaults() -> dict[str, Any]:
    """Default SpecEditor values for Mosflm DPS settings."""
    return {
        "spot_sigma": 5.0,
        "i_sig_i": 5.0,
        "max_penalty": 50.0,
    }


def values_to_mosflm_settings(values: dict[str, Any]) -> dict[str, float]:
    """Normalize SpecEditor ``values()`` to Mosflm numeric settings."""
    return {
        "spot_sigma": float(values.get("spot_sigma") or 5.0),
        "i_sig_i": float(values.get("i_sig_i") or 5.0),
        "max_penalty": float(values.get("max_penalty") or 50.0),
    }
