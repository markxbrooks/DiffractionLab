"""Wire ReflectionData into plot / table / summary without owning widgets long-term."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from molib.xtal.reflections.model import ReflectionData
from molib.xtal.reflections.quantity import (
    DisplayQuantity,
    QuantitySelection,
    compute_display_values,
    resolve_quantity,
)
from molib.xtal.reflections.resolution import ResolutionFilter, resolution_contour_uv
from molib.xtal.reflections.symmetry_axes import zone_symmetry_axes
from molib.xtal.reflections.zone import (
    ZoneSpec,
    axis_labels,
    project_to_zone_plane,
    zone_membership,
)
from diffraction_lab.ui.widgets.reflections.plot import PlotPointCloud, ResolutionContour


@dataclass
class ViewerFilterState:
    """Current zone / quantity / resolution UI state."""

    zone: ZoneSpec
    quantity: DisplayQuantity = DisplayQuantity.F
    column_label: str | None = None
    d_min: float | None = None
    d_max: float | None = None
    contour_ds: tuple[float, ...] = (2.0, 3.0, 4.0, 6.0, 10.0)


@dataclass
class PreparedView:
    """Filtered arrays ready for plot and table."""

    row_indices: np.ndarray
    cloud: PlotPointCloud
    quantity: QuantitySelection | None
    displayed_count: int


def prepare_view(data: ReflectionData, state: ViewerFilterState) -> PreparedView:
    """Apply zone + resolution filters and build plot cloud."""
    miller = data.miller
    zone_mask = zone_membership(miller, state.zone)
    res_mask = ResolutionFilter(d_min=state.d_min, d_max=state.d_max).mask(
        data.d_spacing
    )
    if res_mask.shape[0] != data.n_reflections:
        res_mask = np.ones(data.n_reflections, dtype=bool)
    mask = zone_mask & res_mask
    row_indices = np.nonzero(mask)[0].astype(np.int32)

    quantity = resolve_quantity(
        data, state.quantity, column_label=state.column_label
    )
    if quantity is None:
        # Fall back to first available column
        if data.columns:
            quantity = resolve_quantity(
                data,
                DisplayQuantity.COLUMN,
                column_label=data.columns[0].label,
            )

    sym_axes = zone_symmetry_axes(
        data.space_group, state.zone, unit_cell=data.unit_cell
    )

    if row_indices.size == 0:
        cloud = PlotPointCloud(
            symmetry_axes=tuple(sym_axes),
            axis_u_label=axis_labels(state.zone)[0],
            axis_v_label=axis_labels(state.zone)[1],
        )
        return PreparedView(
            row_indices=row_indices,
            cloud=cloud,
            quantity=quantity,
            displayed_count=0,
        )

    uv = project_to_zone_plane(
        miller[row_indices], state.zone, unit_cell=data.unit_cell
    )
    if quantity is not None:
        values, missing = compute_display_values(data, quantity)
        weights = values[row_indices]
        miss = missing[row_indices]
    else:
        weights = np.ones(row_indices.shape[0], dtype=np.float64)
        miss = np.zeros(row_indices.shape[0], dtype=bool)

    contours: list[ResolutionContour] = []
    for d in state.contour_ds:
        poly = resolution_contour_uv(d, state.zone, data.unit_cell)
        if poly is not None:
            contours.append(ResolutionContour(d_angstrom=float(d), points=poly))

    u_lab, v_lab = axis_labels(state.zone)
    cloud = PlotPointCloud(
        u=uv[:, 0].astype(np.float64),
        v=uv[:, 1].astype(np.float64),
        weights=np.asarray(weights, dtype=np.float64),
        missing=np.asarray(miss, dtype=bool),
        observation_ids=row_indices.copy(),
        contours=contours,
        symmetry_axes=tuple(sym_axes),
        axis_u_label=u_lab,
        axis_v_label=v_lab,
    )
    return PreparedView(
        row_indices=row_indices,
        cloud=cloud,
        quantity=quantity,
        displayed_count=int(row_indices.shape[0]),
    )
