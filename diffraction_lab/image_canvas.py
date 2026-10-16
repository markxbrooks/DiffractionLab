"""Zoomable diffraction-image canvas for DiffractionLab (PicoUI ZoomableCanvas)."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QCursor,
    QFont,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QWidget

try:
    from picoui.widget.zoomable import ViewTransform, ZoomableCanvas, fit_transform

    _HAS_ZOOMABLE = True
except ImportError:  # pragma: no cover
    ViewTransform = None  # type: ignore[misc, assignment]
    ZoomableCanvas = QWidget  # type: ignore[misc, assignment]
    fit_transform = None  # type: ignore[misc, assignment]
    _HAS_ZOOMABLE = False

logger = logging.getLogger(__name__)

# Spot overlay entries: (x_px, y_px, predicted)
# Mosflm Tcl colours (src/spots.tcl / imagedisplay.tcl):
#   found / observed  → red
#   predicted / indexed (fulls) → blue
#
# ``predicted`` True  = indexed/predicted marker (blue)
# ``predicted`` False = found/observed spot (red)
SpotOverlay = list[tuple[float, float, bool]]

# Default d-spacings (Å) for detector resolution rings.
DEFAULT_RING_RESOLUTIONS_A = (1.0, 2.0, 3.0)


@dataclass(frozen=True)
class ResolutionRing:
    """One constant-resolution circle on the detector (image-pixel coords)."""

    d_angstrom: float
    center_x: float
    center_y: float
    radius_px: float


def radius_pixels_for_resolution(
    resolution_a: float,
    wavelength_a: float,
    distance_mm: float,
    pixel_size_mm: float,
) -> float | None:
    """Return ring radius in detector pixels, or ``None`` if not physical."""
    try:
        d = float(resolution_a)
        wavelength = float(wavelength_a)
        distance = float(distance_mm)
        pixel = float(pixel_size_mm)
    except (TypeError, ValueError):
        return None
    if d <= 0.0 or wavelength <= 0.0 or distance <= 0.0 or pixel <= 0.0:
        return None
    arg = wavelength / (2.0 * d)
    if abs(arg) > 1.0:
        return None
    two_theta = 2.0 * math.asin(arg)
    radius_mm = distance * math.tan(two_theta)
    return radius_mm / pixel


def build_resolution_rings(
    *,
    beam_x: float,
    beam_y: float,
    wavelength_a: float,
    distance_mm: float,
    pixel_size_mm: float,
    resolutions_a: tuple[float, ...] = DEFAULT_RING_RESOLUTIONS_A,
) -> list[ResolutionRing]:
    """Build drawable resolution rings in image-pixel coordinates."""
    rings: list[ResolutionRing] = []
    for d in resolutions_a:
        radius = radius_pixels_for_resolution(
            d, wavelength_a, distance_mm, pixel_size_mm
        )
        if radius is None or radius <= 0.0:
            continue
        rings.append(
            ResolutionRing(
                d_angstrom=float(d),
                center_x=float(beam_x),
                center_y=float(beam_y),
                radius_px=float(radius),
            )
        )
    return rings


class DiffractionImageCanvas(ZoomableCanvas if _HAS_ZOOMABLE else QWidget):
    """Diffraction image view with Reflection-Viewer zoom/pan interaction.

    World coordinates are detector pixels (origin top-left, *y* down).
    Resolution rings and indexing spots are painted in screen space from
    world geometry (same pattern as Reflection Viewer contours).
    """

    cursor_moved = Signal(float, float)  # image x, y in pixels

    #: Max distance (image pixels) to snap status readout to a nearby spot.
    NEAREST_SPOT_MAX_PX = 40.0

    def __init__(self, parent: QWidget | None = None) -> None:
        if _HAS_ZOOMABLE:
            super().__init__(parent, flip_y=False)
        else:
            QWidget.__init__(self, parent)
            self.setMinimumSize(200, 200)
            self.setMouseTracking(True)
        self.setToolTip(
            "Crosshair shows cursor resolution · Drag to pan (hand) · "
            "Scroll to zoom · View → Reset Zoom to fit"
        )
        self._pixmap = QPixmap()
        self._spot_overlay: SpotOverlay | None = None
        self._show_spots = True
        self._resolution_rings: list[ResolutionRing] = []
        self._show_rings = True
        self._show_crosshair = True
        self._cursor_world: tuple[float, float] | None = None
        self._nearest_spot: tuple[float, float, float] | None = None  # x, y, dist
        self._placeholder = "No image loaded"
        self._fit_on_next_pixmap = True
        self._set_idle_cursor()

    def set_placeholder(self, text: str) -> None:
        """Text shown when no pixmap is loaded."""
        self._placeholder = str(text)
        self.update()

    def set_image_pixmap(self, pixmap: QPixmap, *, reset_view: bool | None = None) -> None:
        """Replace the displayed image.

        If *reset_view* is ``None``, the view resets only when the image size
        changes or no view has been fitted yet.
        """
        old_size = (self._pixmap.width(), self._pixmap.height())
        self._pixmap = QPixmap(pixmap) if pixmap is not None else QPixmap()
        new_size = (self._pixmap.width(), self._pixmap.height())
        should_reset = reset_view
        if should_reset is None:
            should_reset = self._fit_on_next_pixmap or old_size != new_size
        if should_reset and _HAS_ZOOMABLE:
            self.reset_view()
            self._fit_on_next_pixmap = False
        else:
            self.update()

    def set_spot_overlay(
        self,
        spots: SpotOverlay | None,
        *,
        visible: bool | None = None,
    ) -> None:
        """Set indexing spot markers in image-pixel coordinates."""
        self._spot_overlay = spots
        if visible is not None:
            self._show_spots = bool(visible)
        self.update()

    def set_spots_visible(self, visible: bool) -> None:
        """Show or hide indexing spot markers."""
        self._show_spots = bool(visible)
        self.update()

    def set_resolution_rings(
        self,
        rings: list[ResolutionRing] | None,
        *,
        visible: bool | None = None,
    ) -> None:
        """Set detector resolution rings (image-pixel geometry)."""
        self._resolution_rings = list(rings or [])
        if visible is not None:
            self._show_rings = bool(visible)
        self.update()

    def set_rings_visible(self, visible: bool) -> None:
        """Show or hide resolution rings without recomputing geometry."""
        self._show_rings = bool(visible)
        self.update()

    def set_crosshair_visible(self, visible: bool) -> None:
        """Show or hide the cursor crosshair overlay."""
        self._show_crosshair = bool(visible)
        self.update()

    def cursor_world(self) -> tuple[float, float] | None:
        """Return the last cursor position in image pixels, if any."""
        return self._cursor_world

    def nearest_spot(self) -> tuple[float, float, float] | None:
        """Return ``(x, y, distance_px)`` for the nearest overlay spot, if any."""
        return self._nearest_spot

    def _set_idle_cursor(self) -> None:
        """Crosshair cursor when idle; closed hand while panning (base class)."""
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))

    def has_image(self) -> bool:
        """True when a non-null pixmap is loaded."""
        return not self._pixmap.isNull()

    def image_size(self) -> tuple[int, int]:
        """Return ``(width, height)`` of the source pixmap in pixels."""
        return self._pixmap.width(), self._pixmap.height()

    def fit_content_transform(self) -> ViewTransform | None:
        if not _HAS_ZOOMABLE or self._pixmap.isNull():
            return None
        w = max(self._pixmap.width(), 1)
        h = max(self._pixmap.height(), 1)
        return fit_transform(
            0.0,
            float(w),
            0.0,
            float(h),
            self.width(),
            self.height(),
            padding=0.02,
            flip_y=False,
        )

    def paint_contents(self, painter: QPainter, transform: ViewTransform) -> None:
        if self._pixmap.isNull():
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                self._placeholder,
            )
            return

        src_w = self._pixmap.width()
        src_h = self._pixmap.height()
        x0, y0 = transform.world_to_screen(0.0, 0.0)
        x1, y1 = transform.world_to_screen(float(src_w), float(src_h))
        dest_w = max(1.0, abs(x1 - x0))
        dest_h = max(1.0, abs(y1 - y0))

        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawPixmap(
            QRectF(min(x0, x1), min(y0, y1), dest_w, dest_h),
            self._pixmap,
            QRectF(0.0, 0.0, float(src_w), float(src_h)),
        )

        if self._show_rings and self._resolution_rings:
            self._paint_resolution_rings(painter, transform)

        if self._show_spots and self._spot_overlay:
            half = max(4.0, 6.0 * (transform.scale / max(self.width() / src_w, 1e-6)))
            half = min(half, 14.0)
            # Mosflm: found spots red; predicted/indexed fulls blue.
            pen_found = QPen(QColor(255, 0, 0), 2)  # #ff0000
            pen_predicted = QPen(QColor(0, 0, 255), 2)  # #0000ff
            drawn = 0
            for x_px, y_px, predicted in self._spot_overlay:
                painter.setPen(pen_predicted if predicted else pen_found)
                sx, sy = transform.world_to_screen(float(x_px), float(y_px))
                painter.drawLine(int(sx - half), int(sy), int(sx + half), int(sy))
                painter.drawLine(int(sx), int(sy - half), int(sx), int(sy + half))
                drawn += 1
            if drawn:
                logger.debug("Drew %d indexing spots on zoomable canvas", drawn)

        if self._show_crosshair and self._cursor_world is not None:
            self._paint_crosshair(painter, transform)

    def _paint_crosshair(
        self, painter: QPainter, transform: ViewTransform
    ) -> None:
        """Draw full-widget crosshair at the cursor, highlighting nearest spot."""
        cx, cy = transform.world_to_screen(*self._cursor_world)
        pen = QPen(QColor(0, 220, 255, 200), 1, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(QPointF(0.0, cy), QPointF(float(self.width()), cy))
        painter.drawLine(QPointF(cx, 0.0), QPointF(cx, float(self.height())))
        # Small solid cross at the exact cursor.
        painter.setPen(QPen(QColor(0, 255, 255, 230), 1))
        painter.drawLine(QPointF(cx - 8, cy), QPointF(cx + 8, cy))
        painter.drawLine(QPointF(cx, cy - 8), QPointF(cx, cy + 8))

        if self._nearest_spot is not None:
            sx, sy, _dist = self._nearest_spot
            nx, ny = transform.world_to_screen(sx, sy)
            painter.setPen(QPen(QColor(255, 220, 40, 230), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(nx, ny), 10.0, 10.0)
            painter.drawLine(QPointF(cx, cy), QPointF(nx, ny))

    def nearest_spot_at(
        self, x_px: float, y_px: float
    ) -> tuple[float, float, float] | None:
        """Return nearest overlay spot within :attr:`NEAREST_SPOT_MAX_PX`."""
        if not self._spot_overlay:
            return None
        best: tuple[float, float, float] | None = None
        max_d2 = self.NEAREST_SPOT_MAX_PX * self.NEAREST_SPOT_MAX_PX
        for sx, sy, _predicted in self._spot_overlay:
            dx = float(sx) - x_px
            dy = float(sy) - y_px
            d2 = dx * dx + dy * dy
            if d2 > max_d2:
                continue
            if best is None or d2 < best[2] * best[2]:
                best = (float(sx), float(sy), math.sqrt(d2))
        return best

    def _paint_resolution_rings(
        self, painter: QPainter, transform: ViewTransform
    ) -> None:
        """Draw resolution rings in screen space (crisp at any zoom)."""
        pen = QPen(QColor(255, 64, 64, 220), 2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        font = QFont(painter.font())
        font.setPointSize(max(10, font.pointSize()))
        font.setBold(True)
        painter.setFont(font)
        label_color = QColor(255, 140, 140, 240)

        for ring in self._resolution_rings:
            cx, cy = transform.world_to_screen(ring.center_x, ring.center_y)
            # Map a point on the ring to screen to get radius in screen pixels.
            rx, ry = transform.world_to_screen(
                ring.center_x + ring.radius_px, ring.center_y
            )
            radius = abs(rx - cx)
            if radius < 2.0:
                continue
            painter.setPen(pen)
            painter.drawEllipse(QPointF(cx, cy), radius, radius)
            label = self._format_resolution_label(ring.d_angstrom)
            painter.setPen(label_color)
            painter.drawText(QPointF(cx + radius + 4.0, cy - 3.0), label)

    @staticmethod
    def _format_resolution_label(d_angstrom: float) -> str:
        d = float(d_angstrom)
        if abs(d - round(d)) < 1e-6:
            return f"{int(round(d))} Å"
        text = f"{d:.2f}".rstrip("0").rstrip(".")
        return f"{text} Å"

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._cursor_world = None
        self._nearest_spot = None
        self.update()
        if _HAS_ZOOMABLE:
            super().leaveEvent(event)
        else:
            QWidget.leaveEvent(self, event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if _HAS_ZOOMABLE:
            super().mouseMoveEvent(event)
            if self._pixmap.isNull():
                return
            wu, wv = self._transform.screen_to_world(
                event.position().x(), event.position().y()
            )
            self._cursor_world = (float(wu), float(wv))
            self._nearest_spot = self.nearest_spot_at(wu, wv)
            self.cursor_moved.emit(float(wu), float(wv))
            if not self.is_panning():
                self.update()
        else:
            super().mouseMoveEvent(event)
