"""QPainter reciprocal-space zone plot."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QWidget

from molib.xtal.reflections.symmetry_axes import SymmetryAxis
from picoui.widget.zoomable import ViewTransform, ZoomableCanvas
from picoui.widget.zoomable import fit_transform as fit_world_box


@dataclass(frozen=True, slots=True)
class ResolutionContour:
    """One constant-resolution ring in plot coordinates."""

    d_angstrom: float
    points: np.ndarray  # (N, 2) polyline in (u, v)


@dataclass
class PlotPointCloud:
    """CPU-side plot data for one zone view."""

    u: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    v: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    missing: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    observation_ids: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )
    contours: list[ResolutionContour] = field(default_factory=list)
    symmetry_axes: Sequence[SymmetryAxis] = field(default_factory=tuple)
    axis_u_label: str = "u"
    axis_v_label: str = "v"


def _mosflm_prediction_color(t: float) -> QColor:
    """Map normalised weight ``t`` ∈ [0, 1] to Mosflm prediction colours.

    Matches ``ImageDisplay`` in iMosflm ``imagedisplay.tcl``:

    - fulls → blue ``#0000ff``
    - partials → yellow ``#ffff00``
    - wide → green ``#00ff00``
    - overlaps → red ``#ff0000``

    Intermediate ``t`` values blend between neighbouring bands.
    """
    t = max(0.0, min(1.0, float(t)))
    stops = (
        (0.0, (0, 0, 255)),  # blue — fulls
        (0.33, (255, 255, 0)),  # yellow — partials
        (0.66, (0, 255, 0)),  # green — wide
        (1.0, (255, 0, 0)),  # red — overlaps
    )
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t <= t1 or i == len(stops) - 2:
            if t1 <= t0:
                r, g, b = c1
            else:
                u = (t - t0) / (t1 - t0)
                u = max(0.0, min(1.0, u))
                r = int(c0[0] + (c1[0] - c0[0]) * u)
                g = int(c0[1] + (c1[1] - c0[1]) * u)
                b = int(c0[2] + (c1[2] - c0[2]) * u)
            return QColor(r, g, b, 220)
    return QColor(255, 0, 0, 220)


def fit_transform(
    u: np.ndarray,
    v: np.ndarray,
    width: int,
    height: int,
    *,
    padding: float = 0.08,
) -> ViewTransform:
    """Compute a transform that fits *u*,*v* into the widget."""
    width = max(int(width), 1)
    height = max(int(height), 1)
    if u.size == 0:
        return ViewTransform(width=width, height=height, flip_y=True)
    return fit_world_box(
        float(np.min(u)),
        float(np.max(u)),
        float(np.min(v)),
        float(np.max(v)),
        width,
        height,
        padding=padding,
        flip_y=True,
    )


def pick_nearest(
    u: np.ndarray,
    v: np.ndarray,
    observation_ids: np.ndarray,
    world_u: float,
    world_v: float,
    *,
    max_dist: float,
) -> int | None:
    """Return observation id of nearest point within *max_dist* world units."""
    if u.size == 0:
        return None
    du = u - world_u
    dv = v - world_v
    dist2 = du * du + dv * dv
    i = int(np.argmin(dist2))
    if dist2[i] > max_dist * max_dist:
        return None
    return int(observation_ids[i])


class ReciprocalSpacePlot(ZoomableCanvas):
    """Interactive 2D reciprocal-space plot for one crystallographic zone.

    Inherits pan/zoom look-and-feel from :class:`picoui.widget.zoomable.ZoomableCanvas`
    (same interaction as DiffractionLab's diffraction canvas).
    """

    reflection_selected = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, flip_y=True)
        self.setMinimumSize(320, 320)
        self._cloud = PlotPointCloud()
        self._selected_id: int | None = None
        self._show_axes = True
        self._show_contours = True
        self._show_symmetry_axes = True
        self._marker_scale = 1.0
        self._contrast = 1.0

    def set_point_cloud(self, cloud: PlotPointCloud) -> None:
        """Replace plotted points and reset the view to fit."""
        self._cloud = cloud
        self.reset_view()

    def set_selected_observation(self, observation_id: int | None) -> None:
        """Highlight a reflection by observation id."""
        self._selected_id = (
            None if observation_id is None else int(observation_id)
        )
        self.update()

    def set_show_axes(self, visible: bool) -> None:
        self._show_axes = bool(visible)
        self.update()

    def set_show_contours(self, visible: bool) -> None:
        self._show_contours = bool(visible)
        self.update()

    def set_show_symmetry_axes(self, visible: bool) -> None:
        """Show or hide crystallographic symmetry axes for the current zone."""
        self._show_symmetry_axes = bool(visible)
        self.update()

    def set_marker_scale(self, scale: float) -> None:
        self._marker_scale = max(0.2, float(scale))
        self.update()

    def set_contrast(self, contrast: float) -> None:
        self._contrast = max(0.1, float(contrast))
        self.update()

    def fit_content_transform(self) -> ViewTransform | None:
        return fit_transform(
            self._cloud.u,
            self._cloud.v,
            self.width(),
            self.height(),
        )

    def paint_contents(self, painter: QPainter, transform: ViewTransform) -> None:
        tr = transform
        if self._show_axes:
            self._paint_axes(painter, tr)

        if self._show_symmetry_axes:
            self._paint_symmetry_axes(painter, tr)

        if self._show_contours:
            self._paint_resolution_contours(painter, tr)

        cloud = self._cloud
        if cloud.u.size == 0:
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "No reflections"
            )
            return

        weights = np.asarray(cloud.weights, dtype=np.float64)
        finite = np.isfinite(weights) & ~cloud.missing
        if np.any(finite):
            w = np.abs(weights[finite])
            wmin = float(np.percentile(w, 5))
            wmax = float(np.percentile(w, 95))
            if wmax <= wmin:
                wmax = wmin + 1.0
        else:
            wmin, wmax = 0.0, 1.0

        for i in range(cloud.u.size):
            sx, sy = tr.world_to_screen(float(cloud.u[i]), float(cloud.v[i]))
            obs = int(cloud.observation_ids[i])
            selected = self._selected_id is not None and obs == self._selected_id
            if cloud.missing[i] or not np.isfinite(weights[i]):
                # Mosflm-style "absent": subdued grey (not a typed prediction).
                color = QColor(120, 120, 120, 160)
                r = 2.0 * self._marker_scale
            else:
                t = (abs(float(weights[i])) - wmin) / (wmax - wmin)
                t = max(0.0, min(1.0, t)) ** (1.0 / self._contrast)
                # Mosflm imagedisplay prediction colours by strength bands:
                # fulls blue → partials yellow → wide green → overlaps red.
                color = _mosflm_prediction_color(t)
                r = (2.0 + 5.0 * t) * self._marker_scale
            if selected:
                # Mosflm highlight_colour is gold.
                painter.setPen(QPen(QColor(255, 215, 0), 2))
                painter.setBrush(color)
                painter.drawEllipse(QPointF(sx, sy), r + 2.0, r + 2.0)
            else:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(color)
                painter.drawEllipse(QPointF(sx, sy), r, r)

    def on_left_click(self, world_u: float, world_v: float, event: QMouseEvent) -> None:
        del event
        max_dist = 8.0 / max(self._transform.scale, 1e-6)
        obs = pick_nearest(
            self._cloud.u,
            self._cloud.v,
            self._cloud.observation_ids,
            world_u,
            world_v,
            max_dist=max_dist,
        )
        self._selected_id = obs
        self.update()
        if obs is not None:
            self.reflection_selected.emit(obs)

    def _paint_axes(self, painter: QPainter, tr: ViewTransform) -> None:
        painter.setPen(QPen(QColor(180, 180, 180), 1))
        ox, oy = tr.world_to_screen(0.0, 0.0)
        painter.drawLine(QPointF(0, oy), QPointF(tr.width, oy))
        painter.drawLine(QPointF(ox, 0), QPointF(ox, tr.height))
        painter.drawText(QPointF(tr.width - 40, oy - 6), self._cloud.axis_u_label)
        painter.drawText(QPointF(ox + 6, 16), self._cloud.axis_v_label)

    @staticmethod
    def _format_resolution_label(d_angstrom: float) -> str:
        d = float(d_angstrom)
        if abs(d - round(d)) < 1e-6:
            return f"{int(round(d))} Å"
        text = f"{d:.2f}".rstrip("0").rstrip(".")
        return f"{text} Å"

    def _paint_resolution_contours(
        self, painter: QPainter, tr: ViewTransform
    ) -> None:
        pen = QPen(QColor(90, 140, 200, 160))
        pen.setWidth(1)
        label_color = QColor(140, 190, 255, 230)
        font = QFont(painter.font())
        font.setPointSize(max(9, font.pointSize()))
        painter.setFont(font)

        for contour in self._cloud.contours:
            pts_arr = getattr(contour, "points", contour)
            if pts_arr is None or len(pts_arr) < 2:
                continue
            pts = [
                QPointF(*tr.world_to_screen(float(p[0]), float(p[1])))
                for p in pts_arr
            ]
            painter.setPen(pen)
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i + 1])

            d_val = getattr(contour, "d_angstrom", None)
            if d_val is None:
                continue
            arr = np.asarray(pts_arr, dtype=np.float64)
            scores = arr[:, 0] + arr[:, 1]
            order = np.argsort(-scores)
            anchor = None
            for idx in order[: max(8, len(order) // 8)]:
                sx, sy = tr.world_to_screen(float(arr[idx, 0]), float(arr[idx, 1]))
                if 8 <= sx <= tr.width - 8 and 8 <= sy <= tr.height - 8:
                    anchor = (sx, sy)
                    break
            if anchor is None:
                i_best = int(order[0])
                anchor = tr.world_to_screen(
                    float(arr[i_best, 0]), float(arr[i_best, 1])
                )
            label = self._format_resolution_label(float(d_val))
            painter.setPen(label_color)
            painter.drawText(QPointF(anchor[0] + 4, anchor[1] - 3), label)

    def _paint_symmetry_axes(self, painter: QPainter, tr: ViewTransform) -> None:
        axes = list(self._cloud.symmetry_axes or ())
        if not axes:
            return
        ox, oy = tr.world_to_screen(0.0, 0.0)
        font = QFont(painter.font())
        font.setBold(True)
        font.setPointSize(max(10, font.pointSize()))
        painter.setFont(font)

        corners = [
            tr.screen_to_world(0, 0),
            tr.screen_to_world(tr.width, 0),
            tr.screen_to_world(0, tr.height),
            tr.screen_to_world(tr.width, tr.height),
        ]
        span = max(abs(c[0] - tr.center_u) for c in corners) + max(
            abs(c[1] - tr.center_v) for c in corners
        )
        span = max(span, 1.0)

        color = QColor(255, 170, 70, 220)
        pen = QPen(color, 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)

        normal_labels: list[str] = []
        for ax in axes:
            if ax.orientation == "normal":
                normal_labels.append(ax.label)
                continue
            p0 = tr.world_to_screen(-ax.u * span, -ax.v * span)
            p1 = tr.world_to_screen(ax.u * span, ax.v * span)
            painter.drawLine(QPointF(*p0), QPointF(*p1))
            lx, ly = tr.world_to_screen(ax.u * span * 0.72, ax.v * span * 0.72)
            painter.setPen(QColor(255, 200, 120))
            painter.drawText(QPointF(lx + 4, ly - 4), ax.label)
            painter.setPen(pen)

        if normal_labels:
            seen: list[str] = []
            for lab in normal_labels:
                if lab not in seen:
                    seen.append(lab)
            text = ", ".join(seen)
            r = 10.0 + 2.0 * max(len(seen) - 1, 0)
            painter.setPen(QPen(color, 2))
            painter.setBrush(QColor(40, 40, 44, 200))
            painter.drawEllipse(QPointF(ox, oy), r, r)
            painter.setPen(QColor(255, 200, 120))
            painter.drawText(
                QRectF(ox - 40, oy - r - 18, 80, 16),
                Qt.AlignmentFlag.AlignCenter,
                text,
            )
