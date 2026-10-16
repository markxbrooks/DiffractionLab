"""Reflection summary panel text builders."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from molib.xtal.reflections.model import ReflectionData
from molib.xtal.reflections.quantity import QuantitySelection, compute_display_values


def format_space_group(space_group: Any) -> str:
    """Return a short HM symbol from a Gemmi/string space-group value."""
    if space_group is None:
        return "unavailable"
    text = str(space_group).strip()
    if not text:
        return "unavailable"
    # gemmi.SpaceGroup str is often: <gemmi.SpaceGroup("P 21 21 21")>
    match = re.search(r'SpaceGroup\("([^"]+)"\)', text)
    if match:
        return match.group(1)
    match = re.search(r"SpaceGroup\('([^']+)'\)", text)
    if match:
        return match.group(1)
    return text


def format_unit_cell_line(cell: dict[str, Any] | None) -> str:
    """Format unit-cell parameters for toolbar / summary display."""
    if not isinstance(cell, dict):
        return "Unit cell (file): unavailable"
    return (
        "Unit cell (file): "
        f"a={cell['a']:.3f} b={cell['b']:.3f} c={cell['c']:.3f} "
        f"α={cell['alpha']:.2f} β={cell['beta']:.2f} γ={cell['gamma']:.2f}"
    )


def format_merged_line(is_merged: bool | None) -> str:
    """Format merged/unmerged heuristic for toolbar / summary display."""
    if is_merged is True:
        return "Merged/unmerged: likely merged (heuristic)"
    if is_merged is False:
        return "Merged/unmerged: unmerged / has BATCH (heuristic)"
    return "Merged/unmerged: unknown"


def _format_res_value(value: Any) -> str:
    if value is None:
        return "None"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def format_resolution_header_line(meta: dict[str, Any]) -> str:
    """Format file-header resolution limits."""
    return (
        "Resolution (file header): "
        f"d_min={_format_res_value(meta.get('resolution_high_file'))}  "
        f"d_max={_format_res_value(meta.get('resolution_low_file'))}"
    )


def format_resolution_computed_line(meta: dict[str, Any]) -> str:
    """Format resolution limits computed from Miller indices."""
    return (
        "Resolution (computed from indices): "
        f"d_min={_format_res_value(meta.get('resolution_high_computed'))}  "
        f"d_max={_format_res_value(meta.get('resolution_low_computed'))}"
    )


def format_crystal_toolbar_lines(data: ReflectionData | None) -> dict[str, str]:
    """Return short labels for the crystallographic metadata toolbar."""
    if data is None:
        return {
            "space_group": "Space group: —",
            "unit_cell": "Unit cell (file): —",
            "merged": "Merged/unmerged: —",
            "res_header": "Resolution (file header): —",
            "res_computed": "Resolution (computed from indices): —",
        }
    meta = data.summary_dict()
    return {
        "space_group": f"Space group: {format_space_group(meta.get('space_group'))}",
        "unit_cell": format_unit_cell_line(
            meta.get("unit_cell") if isinstance(meta.get("unit_cell"), dict) else None
        ),
        "merged": format_merged_line(
            meta.get("is_merged") if isinstance(meta.get("is_merged"), bool) else None
        ),
        "res_header": format_resolution_header_line(meta),
        "res_computed": format_resolution_computed_line(meta),
    }


def format_summary_text(
    data: ReflectionData | None,
    *,
    displayed_count: int = 0,
    selection: QuantitySelection | None = None,
) -> str:
    """Return plain-text summary for the summary tab."""
    if data is None:
        return "No reflection file loaded."

    meta = data.summary_dict()
    lines = [
        f"Source file: {meta['path']}",
        f"Format: {meta['format']}  (file-reported)",
        f"Reflections: {meta['n_reflections']}  (file)",
        f"Displayed (current filters): {displayed_count}  (computed)",
        f"Space group: {format_space_group(meta.get('space_group'))}",
        format_unit_cell_line(
            meta.get("unit_cell") if isinstance(meta.get("unit_cell"), dict) else None
        ),
        format_merged_line(
            meta.get("is_merged") if isinstance(meta.get("is_merged"), bool) else None
        ),
        format_resolution_header_line(meta),
        format_resolution_computed_line(meta),
    ]

    lines.append("Columns (label / type / dataset):")
    for col in meta.get("columns") or []:
        lines.append(f"  {col['label']:20s}  {col['type']:4s}  {col.get('dataset','')}")

    if meta.get("datasets"):
        lines.append("Datasets:")
        for ds in meta["datasets"]:
            lines.append(f"  {ds}")

    if selection is not None:
        values, missing = compute_display_values(data, selection)
        ok = values[~missing]
        lines.append(f"Selected quantity: {selection.display_name}  (computed)")
        if ok.size:
            lines.append(
                f"  n={ok.size}  min={np.min(ok):.4g}  max={np.max(ok):.4g}  "
                f"mean={np.mean(ok):.4g}  missing={int(missing.sum())}"
            )
        else:
            lines.append("  No finite values for selected quantity.")

    return "\n".join(lines)


def format_selection_details(
    data: ReflectionData | None,
    observation_id: int | None,
    *,
    selection: QuantitySelection | None = None,
) -> str:
    """Return details for one selected observation."""
    if data is None or observation_id is None:
        return "No reflection selected."
    i = int(observation_id)
    if i < 0 or i >= data.n_reflections:
        return "Invalid selection."
    lines = [
        f"Observation id: {i}",
        f"H K L: {int(data.h[i])} {int(data.k[i])} {int(data.l[i])}",
    ]
    if data.d_spacing is not None and np.isfinite(data.d_spacing[i]):
        lines.append(f"Resolution: {float(data.d_spacing[i]):.3f} Å  (computed)")
    else:
        lines.append("Resolution: unavailable")
    if selection is not None:
        values, missing = compute_display_values(data, selection)
        if missing[i]:
            lines.append(f"{selection.display_name}: missing")
        else:
            lines.append(f"{selection.display_name}: {float(values[i]):.6g}")
        if selection.sigma_label:
            if data.get_missing(selection.sigma_label)[i]:
                lines.append(f"{selection.sigma_label}: missing")
            else:
                lines.append(
                    f"{selection.sigma_label}: "
                    f"{float(data.get_values(selection.sigma_label)[i]):.6g}"
                )
    lines.append("All columns:")
    for label in data.column_labels():
        if data.get_missing(label)[i]:
            lines.append(f"  {label}: missing")
        else:
            lines.append(f"  {label}: {float(data.get_values(label)[i]):.6g}")
    return "\n".join(lines)
