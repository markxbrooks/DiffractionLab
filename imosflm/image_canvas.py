"""Zoomable diffraction-image canvas for qtMosflm (PicoUI ZoomableCanvas)."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPen, QPixmap
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

# Spot overlay entries: (x_px, y_px, matched)
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

    def __init__(self, parent: QWidget | None = None) -> None:
        if _HAS_ZOOMABLE:
            super().__init__(parent, flip_y=False)
        else:
            QWidget.__init__(self, parent)
            self.setMinimumSize(200, 200)
            self.setMouseTracking(True)
        self.setToolTip(
            "Scroll to zoom · Middle-drag or Shift+Left-drag to pan · "
            "View → Reset Zoom to fit"
        )
        self._pixmap = QPixmap()
        self._spot_overlay: SpotOverlay | None = None
        self._show_spots = True
        self._resolution_rings: list[ResolutionRing] = []
        self._show_rings = True
        self._placeholder = "No image loaded"
        self._fit_on_next_pixmap = True

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
            pen = QPen(QColor(0, 220, 70), 2)
            painter.setPen(pen)
            drawn = 0
            for x_px, y_px, matched in self._spot_overlay:
                if not matched:
                    continue
                sx, sy = transform.world_to_screen(float(x_px), float(y_px))
                painter.drawLine(int(sx - half), int(sy), int(sx + half), int(sy))
                painter.drawLine(int(sx), int(sy - half), int(sx), int(sy + half))
                drawn += 1
            if drawn:
                logger.debug("Drew %d indexing spots on zoomable canvas", drawn)

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

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if _HAS_ZOOMABLE:
            super().mouseMoveEvent(event)
            if self._drag_origin is not None:
                return
            if self._pixmap.isNull():
                return
            wu, wv = self._transform.screen_to_world(
                event.position().x(), event.position().y()
            )
            self.cursor_moved.emit(float(wu), float(wv))
        else:
            super().mouseMoveEvent(event)
