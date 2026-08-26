"""Annotation canvas widget for image display and annotation editing."""
from __future__ import annotations

import logging
import math

from PyQt5.QtWidgets import QWidget, QMenu, QAction, QActionGroup, QInputDialog
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF, QTimer, QByteArray
from PyQt5.QtGui import (
    QPainter,
    QPainterPath,
    QPen,
    QBrush,
    QColor,
    QPixmap,
    QImage,
    QImageReader,
    QFont,
    QWheelEvent,
    QMouseEvent,
    QPaintEvent,
    QResizeEvent,
    QPolygonF,
)
from PyQt5.QtSvg import QSvgRenderer

from src.core.annotation import (
    Annotation,
    Keypoint,
    POSE_BOUNDING_BOX_RGB,
    annotation_display_label,
    keypoint_display_rgb,
    keypoint_skeleton_colored_segments,
)
from src.core.resources import LOADING_SVG
from src.ui.icons import icon
from src.ui.theme import PALETTE

logger = logging.getLogger(__name__)

# Visual constants
HANDLE_SIZE = 6
KEYPOINT_RADIUS = 5
POLYGON_EDGE_HIT_RADIUS = 10
POLYGON_INSERT_HANDLE_RADIUS = 4
OBB_ROTATION_HANDLE_OFFSET = 24
OBB_ROTATION_HANDLE_RADIUS = 6
OBB_ROTATION_HIT_RADIUS = 11
CROSSHAIR_GUIDE_WIDTH = 2
CROSSHAIR_CENTER_WIDTH = 3
LABEL_FONT_SIZE = 11
LABEL_PADDING = 3
MIN_SCALE = 0.1
MAX_SCALE = 20.0
ZOOM_FACTOR = 1.15


class AnnotationCanvas(QWidget):
    """Canvas widget for displaying images and editing annotations.

    Signals:
        annotation_created(Annotation): New annotation drawn by user.
        annotation_modified(str): Annotation with given ID was moved/resized.
        annotation_selected(str): Annotation with given ID was selected (or None).
        annotation_deleted(str): Annotation with given ID should be deleted.
        class_requested(float, float): Request class picker at pixel position (after drawing).
        annotations_changed(): Any change to annotations occurred.
    """

    annotation_created = pyqtSignal(object)   # Annotation
    annotation_modified = pyqtSignal(str)     # annotation id
    annotation_selected = pyqtSignal(object)  # annotation id or None
    annotation_deleted = pyqtSignal(str)      # annotation id
    annotation_copied = pyqtSignal(str)       # annotation id (for copy via right-click)
    class_requested = pyqtSignal(float, float)
    class_change_requested = pyqtSignal(str, float, float)  # ann_id, px, py
    annotations_changed = pyqtSignal()
    keypoint_attach_requested = pyqtSignal(str, float, float)  # ann_id, px, py
    keypoint_selected = pyqtSignal(str, int)  # ann_id, kp_index

    tool_mode_requested = pyqtSignal(str)

    zoom_changed = pyqtSignal(float)  # current scale factor

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(200, 200)

        # Image state
        self._image: QPixmap | None = None
        self._image_w: int = 0
        self._image_h: int = 0

        # View transform
        self._scale: float = 1.0
        self._offset_x: float = 0.0
        self._offset_y: float = 0.0
        self._view_initialized: bool = False

        # Tool mode: "select", "draw_bbox", "draw_obb", "draw_polygon", "draw_keypoint"
        self.tool_mode: str = "select"

        # Annotations
        self._annotations: list[Annotation] = []
        self._selected_id: str | None = None
        self._selected_kp_idx: int | None = None
        self._class_colors: dict[str, str] = {}

        # Drawing state
        self._drawing: bool = False
        self._draw_start: tuple[float, float] | None = None  # normalized
        self._draw_current: tuple[float, float] | None = None  # normalized
        self._mouse_pos: tuple[float, float] | None = None  # widget pixels
        self._polygon_points: list[tuple[float, float]] = []
        self._polygon_point_limit: int | None = None
        self._obb_editing_enabled = False

        # Dragging state (move/resize)
        self._dragging: bool = False
        self._drag_type: str = ""  # "move", "resize_tl", "resize_br", etc., "move_kp"
        self._drag_ann_id: str | None = None
        self._drag_kp_idx: int = -1
        self._drag_start_norm: tuple[float, float] | None = None
        self._drag_ann_snapshot: dict | None = None

        # Polygon edge insertion state. When a selected polygon edge is hovered,
        # a small handle is shown at the nearest point. Pressing and dragging
        # that handle inserts a new polygon vertex and moves it immediately.
        self._hover_poly_edge_index: int | None = None
        self._hover_poly_edge_point: tuple[float, float] | None = None

        # Panning state
        self._panning: bool = False
        self._pan_start: tuple[float, float] | None = None

        # Conflict pairs: {ann_id: paired_ann_id} (bidirectional)
        self._conflict_pairs: dict[str, str] = {}
        self._loading = False
        self._loading_angle = 0
        self._loading_renderer = (
            QSvgRenderer(QByteArray(LOADING_SVG.read_bytes()))
            if LOADING_SVG.exists()
            else None
        )
        self._loading_timer = QTimer(self)
        self._loading_timer.setInterval(80)
        self._loading_timer.timeout.connect(self._advance_loading)
        self._set_default_cursor()

    def _set_default_cursor(self) -> None:
        self.setCursor(Qt.ArrowCursor)

    def _set_click_cursor(self) -> None:
        self.setCursor(Qt.PointingHandCursor)




    def norm_to_pixel(self, nx: float, ny: float) -> tuple[float, float]:
        """Convert normalized [0,1] image coords to widget pixel coords."""
        px = nx * self._image_w * self._scale + self._offset_x
        py = ny * self._image_h * self._scale + self._offset_y
        return px, py

    def pixel_to_norm(self, px: float, py: float) -> tuple[float, float]:
        """Convert widget pixel coords to normalized [0,1] image coords."""
        if self._image_w == 0 or self._image_h == 0 or self._scale == 0:
            return 0.0, 0.0
        nx = (px - self._offset_x) / (self._image_w * self._scale)
        ny = (py - self._offset_y) / (self._image_h * self._scale)
        return nx, ny

    def _clamp_norm(self, nx: float, ny: float) -> tuple[float, float]:
        """Clamp normalized coords to [0, 1]."""
        return max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))

    def _minimum_bbox_size_norm(self) -> tuple[float, float]:
        """Return the smallest valid bbox size in normalized image units."""
        min_w = 1.0 / self._image_w if self._image_w > 0 else 0.0
        min_h = 1.0 / self._image_h if self._image_h > 0 else 0.0
        return min_w, min_h



    def load_image(self, path: str) -> None:
        """Load and display an image, fit to window."""
        self.set_loading(True)
        try:
            reader = QImageReader(path)
            reader.setAutoTransform(True)
            qimage = reader.read()
            if qimage.isNull():
                logger.warning("Failed to load image: %s", path)
                return
            self.set_pixmap(QPixmap.fromImage(qimage))
        finally:
            self.set_loading(False)

    def set_loading(self, loading: bool) -> None:
        """Show or hide the canvas loading spinner."""
        self._loading = loading
        if loading:
            self._loading_timer.start()
        else:
            self._loading_timer.stop()
        self.update()

    def set_pixmap(self, pixmap: QPixmap) -> None:
        """Set a pre-loaded pixmap as the display image."""
        keep_scale = self._view_initialized and self._image is not None
        old_scale = self._scale
        self._image = pixmap
        self._image_w = pixmap.width()
        self._image_h = pixmap.height()
        if keep_scale:
            self._scale = max(MIN_SCALE, min(old_scale, MAX_SCALE))
            self._center_image()
            self.zoom_changed.emit(self._scale)
        else:
            self._fit_to_window()
            self._view_initialized = True
            self.zoom_changed.emit(self._scale)
        self.update()

    def set_annotations(self, annotations: list[Annotation]) -> None:
        """Set the annotations to display."""
        # Annotation replacement commonly accompanies an image switch or disk
        # reload. Never let a drag started on the previous image continue and
        # mutate the newly installed annotation list.
        self.cancel_interaction()
        self._annotations = list(annotations)
        self._selected_id = None
        self._clear_polygon_edge_hover()
        self.update()

    def set_class_colors(self, colors: dict[str, str]) -> None:
        """Set class name -> hex color mapping."""
        self._class_colors = colors
        self.update()

    def _advance_loading(self) -> None:
        self._loading_angle = (self._loading_angle + 30) % 360
        self.update()

    def select_annotation(self, ann_id: str | None) -> None:
        """Select an annotation by ID, or deselect with None."""
        self._selected_id = ann_id
        self._selected_kp_idx = None
        self._clear_polygon_edge_hover()
        self.annotation_selected.emit(ann_id)
        self.update()

    def set_tool_mode(self, mode: str) -> None:
        """Set tool mode: select, bbox, OBB, polygon, or keypoint drawing."""
        self.tool_mode = mode
        self._drawing = False
        self._draw_start = None
        self._draw_current = None
        self._mouse_pos = None
        self._clear_polygon_edge_hover()
        if mode != "draw_polygon":
            self._polygon_points = []
        if mode == "select":
            self._set_default_cursor()
        elif mode in ("draw_bbox", "draw_obb", "draw_polygon", "draw_keypoint"):
            self._set_click_cursor()

    def set_polygon_point_limit(self, limit: int | None) -> None:
        """Limit new polygons to a fixed number of vertices (four for OBB)."""
        self._polygon_point_limit = limit if limit and limit >= 3 else None
        self.cancel_polygon()

    def set_obb_editing_enabled(self, enabled: bool) -> None:
        """Enable OBB-specific context-menu and rotation-handle behavior."""
        self._obb_editing_enabled = enabled
        self.update()

    def request_tool_mode(self, mode: str) -> None:
        """Switch the canvas tool mode and notify owner widgets."""
        self.set_tool_mode(mode)
        self.tool_mode_requested.emit(mode)

    def set_locked(self, locked: bool) -> None:
        """Set lock state (no-op, kept for API compatibility)."""
        pass

    @property
    def annotations(self) -> list[Annotation]:
        """Return the current annotations list (mutable reference)."""
        return self._annotations

    @annotations.setter
    def annotations(self, value: list[Annotation]) -> None:
        """Replace annotations list."""
        self._annotations = value
        self.update()

    @property
    def is_locked(self) -> bool:
        """Return whether the canvas is in locked (view-only) mode."""
        return False

    def add_annotation(self, ann: Annotation) -> None:
        """Append an annotation to the canvas and repaint."""
        self._annotations.append(ann)
        self.update()

    def add_annotations(self, anns: list[Annotation]) -> None:
        """Append multiple annotations and repaint once."""
        self._annotations.extend(anns)
        self.update()

    def remove_annotation(self, ann_id: str) -> None:
        """Remove annotation by ID and clear selection."""
        self._annotations = [a for a in self._annotations if a.id != ann_id]
        self._selected_id = None
        self.update()

    def clear_draw_state(self) -> None:
        """Clear in-progress drawing state."""
        self._draw_start = None
        self._draw_current = None
        self._mouse_pos = None
        self._polygon_points = []
        self._clear_polygon_edge_hover()
        self.update()

    def cancel_interaction(self) -> None:
        """Cancel transient mouse interaction without changing annotations."""
        self._drawing = False
        self._draw_start = None
        self._draw_current = None
        self._mouse_pos = None
        self._polygon_points = []
        self._dragging = False
        self._drag_type = ""
        self._drag_ann_id = None
        self._drag_kp_idx = -1
        self._drag_start_norm = None
        self._drag_ann_snapshot = None
        self._panning = False
        self._pan_start = None
        self._clear_polygon_edge_hover()
        if self.tool_mode == "select":
            self._set_default_cursor()
        else:
            self._set_click_cursor()

    def clear(self) -> None:
        """Clear image and annotations."""
        self._image = None
        self._image_w = 0
        self._image_h = 0
        self._annotations = []
        self._selected_id = None
        self._drawing = False
        self._draw_start = None
        self._draw_current = None
        self._mouse_pos = None
        self._polygon_points = []
        self._clear_polygon_edge_hover()
        self._conflict_pairs.clear()
        self._set_default_cursor()
        self.update()



    def set_conflict_pairs(self, pairs: list[tuple[str, str]]) -> None:
        """Set conflict pairs. Each pair is (existing_id, pred_id)."""
        for eid, pid in pairs:
            self._conflict_pairs[eid] = pid
            self._conflict_pairs[pid] = eid
        self.update()

    def resolve_conflict(self, keep_id: str) -> None:
        """Keep one annotation from a conflict pair and remove the other."""
        remove_id = self._conflict_pairs.get(keep_id)
        if not remove_id:
            return
        # Clean up mapping (both directions)
        self._conflict_pairs.pop(keep_id, None)
        self._conflict_pairs.pop(remove_id, None)
        # Remove the losing annotation
        self._annotations = [a for a in self._annotations if a.id != remove_id]
        if self._selected_id == remove_id:
            self._selected_id = None
        self.annotation_deleted.emit(remove_id)
        self.annotations_changed.emit()
        self.update()

    def clear_conflicts(self) -> None:
        """Clear all conflict state."""
        self._conflict_pairs.clear()
        self.update()

    def get_selected_annotation(self) -> Annotation | None:
        """Return the currently selected annotation."""
        if self._selected_id is None:
            return None
        for ann in self._annotations:
            if ann.id == self._selected_id:
                return ann
        return None

    def hit_test(self, px: float, py: float) -> str | None:
        """Find annotation at pixel position. Returns annotation ID or None."""
        nx, ny = self.pixel_to_norm(px, py)

        # Check keypoints first (smaller targets, higher priority)
        kp_radius_norm_x = KEYPOINT_RADIUS / (self._image_w * self._scale) if self._image_w * self._scale > 0 else 0
        kp_radius_norm_y = KEYPOINT_RADIUS / (self._image_h * self._scale) if self._image_h * self._scale > 0 else 0

        for ann in reversed(self._annotations):
            for kp in ann.keypoints:
                if abs(kp.x - nx) < kp_radius_norm_x * 2 and abs(kp.y - ny) < kp_radius_norm_y * 2:
                    return ann.id

        # Segment annotations carry a bbox for export/conflict logic, but canvas
        # selection should follow the visible mask region, not the bbox fill.
        for ann in reversed(self._annotations):
            if ann.polygon and self._hit_test_polygon_region(ann, px, py):
                return ann.id

        # Check detection bboxes. Polygon annotations are intentionally skipped:
        # their bbox is metadata and should not make the whole rectangle clickable.
        for ann in reversed(self._annotations):
            if ann.bbox and not ann.polygon:
                cx, cy, w, h = ann.bbox
                x1, y1 = cx - w / 2, cy - h / 2
                x2, y2 = cx + w / 2, cy + h / 2
                if x1 <= nx <= x2 and y1 <= ny <= y2:
                    return ann.id

        return None

    def _fit_to_window(self) -> None:
        """Scale and offset so image fits in widget."""
        if self._image_w == 0 or self._image_h == 0:
            return
        ww, wh = self.width(), self.height()
        if ww <= 0 or wh <= 0:
            return
        sx = ww / self._image_w
        sy = wh / self._image_h
        self._scale = min(sx, sy)
        # Center the image
        self._offset_x = (ww - self._image_w * self._scale) / 2
        self._offset_y = (wh - self._image_h * self._scale) / 2

    def _center_image(self) -> None:
        """Center the current image without changing the zoom scale."""
        if self._image_w == 0 or self._image_h == 0:
            return
        self._offset_x = (self.width() - self._image_w * self._scale) / 2
        self._offset_y = (self.height() - self._image_h * self._scale) / 2



    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor(PALETTE["canvas"]))

        if self._image is None:
            if self._loading:
                self._paint_loading(painter)
            else:
                painter.setPen(QColor(PALETTE["text_subtle"]))
                painter.drawText(self.rect(), Qt.AlignCenter, "无图片")
            painter.end()
            return

        # Draw image
        dest = QRectF(
            self._offset_x, self._offset_y,
            self._image_w * self._scale, self._image_h * self._scale,
        )
        painter.drawPixmap(dest.toRect(), self._image)
        self._paint_crosshair(painter, dest)

        # Viewport bounds for culling
        vp_left = 0.0
        vp_top = 0.0
        vp_right = float(self.width())
        vp_bottom = float(self.height())

        # LOD: skip labels at very small zoom
        draw_labels = self._scale >= 0.3

        # Draw annotations with viewport culling
        for ann in self._annotations:
            # Cull: skip if annotation is entirely outside viewport
            if ann.bbox and not self._ann_in_viewport(ann, vp_left, vp_top, vp_right, vp_bottom):
                continue
            is_selected = ann.id == self._selected_id
            color = QColor(self._class_colors.get(ann.class_name, PALETTE["primary"]))
            self._paint_annotation(painter, ann, color, is_selected, draw_labels)

        # Draw in-progress bbox
        if self.tool_mode == "draw_polygon" and self._polygon_points:
            self._paint_polygon_preview(painter)
        elif self._drawing and self._draw_start and self._draw_current:
            self._paint_drawing_preview(painter)

        # Zoom level indicator + lock badge
        if self._image is not None:
            font = QFont()
            font.setPixelSize(11)
            painter.setFont(font)
            painter.setPen(QColor(PALETTE["text_subtle"]))
            zoom_pct = int(self._scale * 100)
            status_text = f"{zoom_pct}%"
            painter.drawText(8, self.height() - 8, status_text)

        if self._loading:
            self._paint_loading(painter)

        painter.end()

    def _paint_loading(self, painter: QPainter) -> None:
        size = 34
        x = (self.width() - size) / 2
        y = (self.height() - size) / 2
        if self._loading_renderer is None:
            painter.setPen(QColor(PALETTE["primary"]))
            painter.drawText(self.rect(), Qt.AlignCenter, "Loading")
            return
        painter.save()
        painter.translate(x + size / 2, y + size / 2)
        painter.rotate(self._loading_angle)
        painter.translate(-size / 2, -size / 2)
        self._loading_renderer.render(painter, QRectF(0, 0, size, size))
        painter.restore()

    def _paint_crosshair(self, painter: QPainter, image_rect: QRectF) -> None:
        if self._mouse_pos is None:
            return
        if self._panning or self._dragging:
            return
        px, py = self._mouse_pos
        if not image_rect.contains(px, py):
            return

        painter.save()
        painter.setClipRect(image_rect)
        primary = QColor(PALETTE["primary"])
        faint = QColor(primary)
        faint.setAlpha(95)
        painter.setPen(QPen(faint, CROSSHAIR_GUIDE_WIDTH, Qt.DashLine))
        painter.drawLine(int(image_rect.left()), int(py), int(image_rect.right()), int(py))
        painter.drawLine(int(px), int(image_rect.top()), int(px), int(image_rect.bottom()))
        painter.setPen(QPen(primary, CROSSHAIR_CENTER_WIDTH))
        gap = 7
        painter.drawLine(int(px - gap), int(py), int(px + gap), int(py))
        painter.drawLine(int(px), int(py - gap), int(px), int(py + gap))
        painter.restore()

    def _ann_in_viewport(
        self, ann: Annotation, vp_left: float, vp_top: float, vp_right: float, vp_bottom: float
    ) -> bool:
        """Check if annotation bbox overlaps the viewport."""
        bounds = self._annotation_bounds(ann)
        if bounds is None:
            return True
        x1, y1 = self.norm_to_pixel(bounds[0], bounds[1])
        x2, y2 = self.norm_to_pixel(bounds[2], bounds[3])
        # Annotation is outside if entirely to the left, right, above, or below viewport
        if x2 < vp_left or x1 > vp_right or y2 < vp_top or y1 > vp_bottom:
            return False
        return True

    def _paint_annotation(
        self, painter: QPainter, ann: Annotation, color: QColor, selected: bool,
        draw_labels: bool = True,
    ) -> None:
        """Paint a single annotation (bbox + keypoints + label)."""
        in_conflict = ann.id in self._conflict_pairs
        if ann.polygon:
            points = [QPointF(*self.norm_to_pixel(x, y)) for x, y in ann.polygon]
            polygon = QPolygonF(points)
            pen = QPen(color, 2)
            if not ann.confirmed:
                pen.setStyle(Qt.DashLine)
            if selected:
                pen.setWidth(pen.width() + 1)
            fill = QColor(color)
            fill.setAlpha(45 if selected else 28)
            painter.setPen(pen)
            painter.setBrush(QBrush(fill))
            painter.drawPolygon(polygon)

            if draw_labels or selected:
                label_x, label_y = points[0].x(), points[0].y()
                self._paint_label(painter, label_x, label_y, ann, color, in_conflict)

            if selected:
                if self._is_obb_annotation(ann):
                    self._paint_obb_resize_handles(painter, ann.polygon)
                    self._paint_obb_rotation_handle(painter, ann)
                else:
                    self._paint_polygon_handles(painter, ann.polygon)

        if ann.bbox and not ann.polygon:
            cx, cy, w, h = ann.bbox
            x1, y1 = self.norm_to_pixel(cx - w / 2, cy - h / 2)
            x2, y2 = self.norm_to_pixel(cx + w / 2, cy + h / 2)
            bbox_color = QColor(*POSE_BOUNDING_BOX_RGB) if ann.keypoints else color

            if in_conflict and not ann.confirmed:
                # Conflict prediction: teal dashed, thicker
                conflict_color = bbox_color if ann.keypoints else QColor(PALETTE["teal"])
                pen = QPen(conflict_color, 3, Qt.DashLine)
            else:
                pen = QPen(bbox_color, 2)
                if not ann.confirmed:
                    pen.setStyle(Qt.DashLine)
            if selected:
                pen.setWidth(pen.width() + 1)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))

            # Label background (skip at low zoom for performance)
            if draw_labels or selected:
                self._paint_label(painter, x1, y1, ann, color, in_conflict)

            # Control handles when selected
            if selected:
                self._paint_handles(painter, x1, y1, x2, y2)

        # Pose skeleton is painted first so keypoint circles stay visible on top.
        if ann.keypoints:
            painter.setBrush(Qt.NoBrush)
            fallback_rgb = (color.red(), color.green(), color.blue())
            for start, end, rgb in keypoint_skeleton_colored_segments(
                ann.keypoints, fallback_rgb
            ):
                skeleton_pen = QPen(QColor(*rgb), 2 + (1 if selected else 0))
                skeleton_pen.setCapStyle(Qt.RoundCap)
                painter.setPen(skeleton_pen)
                painter.drawLine(
                    QPointF(*self.norm_to_pixel(start.x, start.y)),
                    QPointF(*self.norm_to_pixel(end.x, end.y)),
                )

        # Keypoints
        for i, kp in enumerate(ann.keypoints):
            px, py = self.norm_to_pixel(kp.x, kp.y)
            is_kp_selected = selected and self._selected_kp_idx == i
            r = KEYPOINT_RADIUS + (3 if is_kp_selected else (2 if selected else 0))
            point_rgb = keypoint_display_rgb(
                ann.keypoints, i, (color.red(), color.green(), color.blue())
            )
            point_color = QColor(*point_rgb)

            if kp.visible == 0:
                painter.setPen(QPen(QColor(PALETTE["text_subtle"]), 1))
                painter.setBrush(Qt.NoBrush)
            elif kp.visible == 1:
                painter.setPen(QPen(point_color, 1))
                painter.setBrush(Qt.NoBrush)
            else:
                painter.setPen(QPen(point_color, 1))
                painter.setBrush(QBrush(point_color))

            if is_kp_selected:
                painter.setPen(QPen(QColor(PALETTE["text"]), 2))

            painter.drawEllipse(QPointF(px, py), r, r)

            # Label for keypoint (show when selected or when individual kp is selected)
            if (selected and draw_labels) or is_kp_selected:
                painter.setPen(QColor(PALETTE["text"]))
                font = QFont()
                font.setPixelSize(10)
                painter.setFont(font)
                vis_text = ["inv", "occ", "vis"][kp.visible]
                label_text = f"{kp.label} ({vis_text})" if is_kp_selected else kp.label
                painter.drawText(int(px + r + 2), int(py - 2), label_text)

    def _paint_handles(self, painter: QPainter, x1: float, y1: float, x2: float, y2: float) -> None:
        """Paint resize handles on selected bbox corners."""
        painter.setPen(QPen(QColor(PALETTE["text"]), 1))
        painter.setBrush(QBrush(QColor(PALETTE["primary"])))
        hs = HANDLE_SIZE
        for hx, hy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            painter.drawRect(QRectF(hx - hs, hy - hs, hs * 2, hs * 2))

    def _paint_polygon_handles(self, painter: QPainter, polygon: list[tuple[float, float]]) -> None:
        """Paint existing polygon vertices and the hovered edge-insert handle."""
        painter.setPen(QPen(QColor(PALETTE["text"]), 1))
        painter.setBrush(QBrush(QColor(PALETTE["primary"])))
        hs = HANDLE_SIZE
        for nx, ny in polygon:
            px, py = self.norm_to_pixel(nx, ny)
            painter.drawEllipse(QPointF(px, py), hs, hs)

        # The smaller green handle marks where a new vertex will be inserted.
        if (
            self._hover_poly_edge_index is not None
            and self._hover_poly_edge_point is not None
            and not self._dragging
        ):
            px, py = self.norm_to_pixel(*self._hover_poly_edge_point)
            radius = POLYGON_INSERT_HANDLE_RADIUS
            painter.setPen(QPen(QColor(PALETTE["text"]), 1))
            painter.setBrush(QBrush(QColor(PALETTE["success"])))
            painter.drawEllipse(QPointF(px, py), radius, radius)

            # A tiny plus sign makes the insertion action discoverable.
            painter.setPen(QPen(QColor(PALETTE["ink"]), 1))
            arm = max(2, radius - 1)
            painter.drawLine(int(px - arm), int(py), int(px + arm), int(py))
            painter.drawLine(int(px), int(py - arm), int(px), int(py + arm))

    def _paint_obb_resize_handles(
        self,
        painter: QPainter,
        polygon: list[tuple[float, float]],
    ) -> None:
        """Paint OBB corners as bbox-style square resize handles."""
        painter.setPen(QPen(QColor(PALETTE["text"]), 1))
        painter.setBrush(QBrush(QColor(PALETTE["primary"])))
        hs = HANDLE_SIZE
        for nx, ny in polygon:
            px, py = self.norm_to_pixel(nx, ny)
            painter.drawRect(QRectF(px - hs, py - hs, hs * 2, hs * 2))

    def _paint_obb_rotation_handle(self, painter: QPainter, ann: Annotation) -> None:
        positions = self._obb_rotation_handle_positions(ann)
        if positions is None:
            return
        edge_point, handle_point = positions
        painter.setPen(QPen(QColor(PALETTE["primary"]), 2))
        painter.setBrush(QBrush(QColor(PALETTE["panel"])))
        painter.drawLine(QPointF(*edge_point), QPointF(*handle_point))
        painter.drawEllipse(
            QPointF(*handle_point),
            OBB_ROTATION_HANDLE_RADIUS,
            OBB_ROTATION_HANDLE_RADIUS,
        )

    def _paint_label(
        self, painter: QPainter, x: float, y: float, ann: Annotation, color: QColor, in_conflict: bool
    ) -> None:
        label_text = annotation_display_label(
            ann,
            (self._image_w, self._image_h),
        )
        if in_conflict:
            label_text += " \u21c4"
        elif not ann.confirmed:
            label_text += " \u26a1"
        font = QFont()
        font.setPixelSize(LABEL_FONT_SIZE)
        painter.setFont(font)
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(label_text) + LABEL_PADDING * 2
        th = fm.height() + LABEL_PADDING * 2
        label_rect = QRectF(x, y - th, tw, th)
        if label_rect.top() < 0:
            label_rect.moveTop(y)
        painter.setPen(color)
        painter.drawText(label_rect, Qt.AlignCenter, label_text)

    def _paint_polygon_preview(self, painter: QPainter) -> None:
        points = [QPointF(*self.norm_to_pixel(x, y)) for x, y in self._polygon_points]
        if self._draw_current is not None:
            points.append(QPointF(*self.norm_to_pixel(*self._draw_current)))
        if not points:
            return
        painter.setPen(QPen(QColor(PALETTE["primary"]), 2, Qt.DashLine))
        fill = QColor(PALETTE["primary"])
        fill.setAlpha(24)
        painter.setBrush(QBrush(fill))
        if len(points) >= 3:
            painter.drawPolygon(QPolygonF(points))
        elif len(points) >= 2:
            painter.drawPolyline(QPolygonF(points))
        for point in points:
            painter.setBrush(QBrush(QColor(PALETTE["primary"])))
            painter.drawEllipse(point, HANDLE_SIZE, HANDLE_SIZE)

    def _paint_drawing_preview(self, painter: QPainter) -> None:
        """Paint the bbox being drawn with size HUD."""
        sx, sy = self.norm_to_pixel(*self._draw_start)
        ex, ey = self.norm_to_pixel(*self._draw_current)
        painter.setPen(QPen(QColor(PALETTE["primary"]), 2, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        x = min(sx, ex)
        y = min(sy, ey)
        w = abs(ex - sx)
        h = abs(ey - sy)
        painter.drawRect(QRectF(x, y, w, h))

        # Size HUD (pixel dimensions)
        if self._image_w > 0 and self._image_h > 0:
            ns = self._draw_start
            nc = self._draw_current
            pw = int(abs(nc[0] - ns[0]) * self._image_w)
            ph = int(abs(nc[1] - ns[1]) * self._image_h)
            size_text = f"{pw} x {ph}"
            font = QFont()
            font.setPixelSize(11)
            painter.setFont(font)
            bg = QColor(PALETTE["panel_raised"])
            bg.setAlpha(200)
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(size_text) + 8
            th = fm.height() + 4
            label_x = x + w / 2 - tw / 2
            label_y = y + h + 4
            painter.fillRect(QRectF(label_x, label_y, tw, th), bg)
            painter.setPen(QColor(PALETTE["text"]))
            painter.drawText(QRectF(label_x, label_y, tw, th), Qt.AlignCenter, size_text)



    def mousePressEvent(self, event: QMouseEvent) -> None:
        px, py = event.x(), event.y()
        self._mouse_pos = (px, py)

        if self.tool_mode == "draw_polygon" and event.button() == Qt.RightButton:
            if len(self._polygon_points) >= 3:
                self.finish_polygon()
            else:
                self.cancel_polygon()
            return

        # Middle button pans everywhere. Ctrl+Left remains a fallback in any
        # tool mode; plain Left pans blank canvas space in select mode below.
        ctrl_left = (
            event.button() == Qt.LeftButton
            and event.modifiers() & Qt.ControlModifier
        )
        if event.button() == Qt.MiddleButton or ctrl_left:
            self._panning = True
            self._pan_start = (px, py)
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() != Qt.LeftButton:
            return

        if self.tool_mode == "draw_obb":
            handle = self._hit_test_handle(px, py)
            if handle == "rotate_obb":
                self._dragging = True
                self._drag_type = handle
                self._drag_ann_id = self._selected_id
                self._drag_start_norm = self.pixel_to_norm(px, py)
                ann = self.get_selected_annotation()
                if ann:
                    self._drag_ann_snapshot = ann.to_dict()
                return

        if self.tool_mode in ("draw_bbox", "draw_obb"):
            nx, ny = self._clamp_norm(*self.pixel_to_norm(px, py))
            self._drawing = True
            self._draw_start = (nx, ny)
            self._draw_current = (nx, ny)

        elif self.tool_mode == "draw_polygon":
            nx, ny = self._clamp_norm(*self.pixel_to_norm(px, py))
            if len(self._polygon_points) >= 3 and self._near_polygon_first_point(px, py):
                self.finish_polygon()
                return
            if not self._polygon_points or self._distance_norm(self._polygon_points[-1], (nx, ny)) > 0.001:
                self._polygon_points.append((nx, ny))
            self._draw_current = (nx, ny)
            if (
                self._polygon_point_limit is not None
                and len(self._polygon_points) >= self._polygon_point_limit
            ):
                self.finish_polygon()
                return
            self.update()

        elif self.tool_mode == "draw_keypoint":
            nx, ny = self._clamp_norm(*self.pixel_to_norm(px, py))
            self._draw_start = (nx, ny)
            # Don't emit class_requested here; do it on mouse release
            # to avoid popup appearing while mouse button is still pressed

        elif self.tool_mode == "select":
            # Check if clicking a handle first (for selected bbox)
            handle = self._hit_test_handle(px, py)
            if handle:
                self._dragging = True
                self._drag_type = handle
                self._drag_ann_id = self._selected_id
                self._drag_start_norm = self.pixel_to_norm(px, py)
                ann = self.get_selected_annotation()
                if ann:
                    self._drag_ann_snapshot = ann.to_dict()
                return

            # Check if clicking a keypoint to drag
            kp_hit = self._hit_test_keypoint(px, py)
            if kp_hit:
                ann_id, kp_idx = kp_hit
                self._set_click_cursor()
                self._dragging = True
                self._drag_type = "move_kp"
                self._drag_ann_id = ann_id
                self._drag_kp_idx = kp_idx
                self._drag_start_norm = self.pixel_to_norm(px, py)
                return

            # A selected polygon edge can be dragged to insert a new vertex.
            # Existing vertex/keypoint handles take priority, so this check comes
            # after them and before the whole-annotation move hit test.
            edge_hit = self._hit_test_polygon_edge(px, py)
            if edge_hit:
                edge_index, insert_point = edge_hit
                ann = self.get_selected_annotation()
                if ann and ann.polygon:
                    self._drag_ann_snapshot = ann.to_dict()
                    insert_index = edge_index + 1
                    ann.polygon.insert(insert_index, insert_point)
                    self._sync_polygon_bbox(ann)
                    if not ann.confirmed:
                        ann.confirmed = True

                    self._dragging = True
                    self._drag_type = f"poly_vertex_{insert_index}"
                    self._drag_ann_id = ann.id
                    self._drag_start_norm = insert_point
                    self._clear_polygon_edge_hover()
                    self._set_click_cursor()
                    self.annotations_changed.emit()
                    self.update()
                    return

            # Hit test annotations
            hit_id = self.hit_test(px, py)
            if hit_id:
                self.select_annotation(hit_id)
                self._set_click_cursor()
                self._dragging = True
                self._drag_type = "move"
                self._drag_ann_id = hit_id
                self._drag_start_norm = self.pixel_to_norm(px, py)
                ann = self.get_selected_annotation()
                if ann:
                    self._drag_ann_snapshot = ann.to_dict()
            else:
                self.select_annotation(None)
                self._panning = True
                self._pan_start = (px, py)
                self.setCursor(Qt.ClosedHandCursor)
                return

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        px, py = event.x(), event.y()
        self._mouse_pos = (px, py)

        if self._panning and self._pan_start:
            dx = px - self._pan_start[0]
            dy = py - self._pan_start[1]
            self._offset_x += dx
            self._offset_y += dy
            self._pan_start = (px, py)
            self.update()
            return

        if self._drawing and self._draw_start:
            nx, ny = self._clamp_norm(*self.pixel_to_norm(px, py))
            self._draw_current = (nx, ny)
            self.update()
            return

        if self.tool_mode == "draw_polygon" and self._polygon_points:
            nx, ny = self._clamp_norm(*self.pixel_to_norm(px, py))
            self._draw_current = (nx, ny)
            self.update()
            return

        if self._dragging and self._drag_ann_id:
            if self._drag_type == "rotate_obb":
                self.setCursor(Qt.CrossCursor)
            elif self._drag_type.startswith("resize_obb_"):
                vertex_idx = int(self._drag_type.rsplit("_", 1)[1])
                self.setCursor(Qt.SizeFDiagCursor if vertex_idx % 2 == 0 else Qt.SizeBDiagCursor)
            elif (
                self._drag_type in ("move", "move_kp")
                or self._drag_type.startswith("poly_vertex_")
            ):
                self._set_click_cursor()
            nx, ny = self.pixel_to_norm(px, py)
            self._handle_drag(nx, ny)
            self.update()
            return

        # Cursor feedback in select mode
        if self.tool_mode == "select" and self._image is not None:
            if event.modifiers() & Qt.ControlModifier:
                self.setCursor(Qt.OpenHandCursor)
                self.update()
                return
            handle = self._hit_test_handle(px, py)
            if handle:
                self._clear_polygon_edge_hover()
                if handle == "rotate_obb":
                    self.setCursor(Qt.CrossCursor)
                elif handle.startswith("resize_obb_"):
                    vertex_idx = int(handle.rsplit("_", 1)[1])
                    self.setCursor(Qt.SizeFDiagCursor if vertex_idx % 2 == 0 else Qt.SizeBDiagCursor)
                elif handle.startswith("poly_vertex_"):
                    self._set_click_cursor()
                elif "tl" in handle or "br" in handle:
                    self.setCursor(Qt.SizeFDiagCursor)
                else:
                    self.setCursor(Qt.SizeBDiagCursor)
            elif self._hit_test_keypoint(px, py):
                self._clear_polygon_edge_hover()
                self._set_click_cursor()
            else:
                edge_hit = self._hit_test_polygon_edge(px, py)
                if edge_hit:
                    edge_index, edge_point = edge_hit
                    self._set_polygon_edge_hover(edge_index, edge_point)
                    self._set_click_cursor()
                elif self.hit_test(px, py):
                    self._clear_polygon_edge_hover()
                    self._set_click_cursor()
                else:
                    self._clear_polygon_edge_hover()
                    self.setCursor(Qt.OpenHandCursor)
        elif self._image is not None:
            # Draw modes: Ctrl signals pan availability, otherwise image clicks.
            if event.modifiers() & Qt.ControlModifier:
                self.setCursor(Qt.OpenHandCursor)
            else:
                self._set_click_cursor()
        if self._image is not None:
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        px, py = event.x(), event.y()

        if self._panning and event.button() in (Qt.MiddleButton, Qt.LeftButton):
            self._panning = False
            self._pan_start = None
            if self.tool_mode == "select":
                self._set_default_cursor()
            else:
                self._set_click_cursor()
            return

        if event.button() != Qt.LeftButton:
            return

        if (
            self._drawing
            and self._draw_start
            and self.tool_mode in ("draw_bbox", "draw_obb")
        ):
            nx, ny = self._clamp_norm(*self.pixel_to_norm(px, py))
            self._draw_current = (nx, ny)
            sx, sy = self._draw_start
            w = abs(nx - sx)
            h = abs(ny - sy)
            if w > 0.01 and h > 0.01:
                self.class_requested.emit(px, py)
            else:
                self._draw_start = None
                self._draw_current = None
            self._drawing = False
            self.update()
            return

        if self.tool_mode == "draw_keypoint" and self._draw_start:
            # Check if inside an existing bbox; attach to it.
            hit_id = self.hit_test(px, py)
            if hit_id:
                ann = next((a for a in self._annotations if a.id == hit_id), None)
                if ann and ann.bbox:
                    self.keypoint_attach_requested.emit(hit_id, px, py)
                    return
            # Outside any bbox; create a standalone keypoint.
            self.class_requested.emit(px, py)
            return

        if self._dragging:
            if (
                self._drag_type in (
                    "move", "resize_tl", "resize_tr", "resize_bl", "resize_br",
                    "move_kp", "rotate_obb",
                )
                or self._drag_type.startswith(("poly_vertex_", "resize_obb_"))
            ):
                self.annotation_modified.emit(self._drag_ann_id)
            self._dragging = False
            self._drag_type = ""
            self._drag_ann_id = None
            self._drag_kp_idx = -1
            self._drag_start_norm = None
            self._drag_ann_snapshot = None
            self._clear_polygon_edge_hover()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if self.tool_mode == "draw_polygon" and event.button() == Qt.LeftButton:
            self.finish_polygon()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def leaveEvent(self, event) -> None:
        """Hide transient hover guides when the mouse leaves."""
        had_mouse_pos = self._mouse_pos is not None
        self._mouse_pos = None
        if not self._dragging:
            self._clear_polygon_edge_hover()
        if had_mouse_pos:
            self.update()
        super().leaveEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._image is None:
            return
        factor = ZOOM_FACTOR if event.angleDelta().y() > 0 else 1.0 / ZOOM_FACTOR
        self._apply_zoom(event.x(), event.y(), factor)
        event.accept()

    def _apply_zoom(self, center_px: float, center_py: float, factor: float) -> None:
        """Apply zoom by factor around a pixel center point."""
        old_nx, old_ny = self.pixel_to_norm(center_px, center_py)
        self._scale = max(MIN_SCALE, min(self._scale * factor, MAX_SCALE))
        new_px = old_nx * self._image_w * self._scale + self._offset_x
        new_py = old_ny * self._image_h * self._scale + self._offset_y
        self._offset_x += center_px - new_px
        self._offset_y += center_py - new_py
        self.zoom_changed.emit(self._scale)
        self.update()

    def zoom_in(self) -> None:
        """Zoom in by one step."""
        if self._image is None:
            return
        self._apply_zoom(self.width() / 2, self.height() / 2, ZOOM_FACTOR)

    def zoom_out(self) -> None:
        """Zoom out by one step."""
        if self._image is None:
            return
        self._apply_zoom(self.width() / 2, self.height() / 2, 1.0 / ZOOM_FACTOR)

    def zoom_fit(self) -> None:
        """Reset zoom to fit image in window."""
        if self._image is None:
            return
        self._fit_to_window()
        self._view_initialized = True
        self.zoom_changed.emit(self._scale)
        self.update()

    def contextMenuEvent(self, event) -> None:
        """Show right-click context menu."""
        px, py = event.x(), event.y()

        if self.tool_mode == "draw_polygon" and self._polygon_points:
            return

        if not self._is_image_pixel(px, py):
            return


        kp_hit = self._hit_test_keypoint(px, py)

        hit_id = self.hit_test(px, py)
        ann = None
        if hit_id:
            self.select_annotation(hit_id)
            ann = self.get_selected_annotation()

        menu = QMenu(self)

        self._add_tool_mode_actions(menu)
        if not ann:
            menu.exec_(event.globalPos())
            return

        menu.addSeparator()

        # Keypoint-specific actions (when right-clicking directly on a keypoint)
        if kp_hit and kp_hit[0] == hit_id:
            kp_idx = kp_hit[1]
            kp = ann.keypoints[kp_idx] if kp_idx < len(ann.keypoints) else None
            if kp:
                vis_names = ["不可见", "遮挡", "可见"]
                vis_label = vis_names[kp.visible] if kp.visible < 3 else "?"
                kp_header = menu.addAction(f"关键点 {kp.label} ({vis_label})")
                kp_header.setEnabled(False)

                rename_kp = menu.addAction("重命名关键点")
                rename_kp.triggered.connect(
                    lambda _, aid=ann.id, ki=kp_idx: self._request_rename_keypoint(aid, ki))

                cycle_vis = menu.addAction("切换可见性")
                cycle_vis.triggered.connect(
                    lambda _, aid=ann.id, ki=kp_idx: self.cycle_keypoint_visibility(aid, ki))

                del_kp = menu.addAction("删除关键点")
                del_kp.triggered.connect(
                    lambda _, aid=ann.id, ki=kp_idx: self.remove_keypoint(aid, ki))

                menu.addSeparator()

        # Conflict resolution options
        paired_id = self._conflict_pairs.get(ann.id)
        if paired_id:
            paired_ann = next((a for a in self._annotations if a.id == paired_id), None)
            if paired_ann:
                # Determine which is existing (confirmed) and which is prediction
                if ann.confirmed:
                    existing_ann, pred_ann = ann, paired_ann
                else:
                    existing_ann, pred_ann = paired_ann, ann
                keep_existing = menu.addAction(
                    f"保留确认框 (conf={existing_ann.confidence:.2f})")
                keep_existing.triggered.connect(
                    lambda: self.resolve_conflict(existing_ann.id))
                keep_pred = menu.addAction(
                    f"保留预测框 (conf={pred_ann.confidence:.2f})")
                keep_pred.triggered.connect(
                    lambda: self.resolve_conflict(pred_ann.id))
                menu.addSeparator()

        # Modify class
        change_cls = menu.addAction("修改类别")
        change_cls.triggered.connect(lambda: self.class_change_requested.emit(ann.id, px, py))

        if ann.confirmed:
            unconfirm = menu.addAction("取消确认")
            unconfirm.triggered.connect(lambda: self._toggle_confirm(ann, False))
        else:
            confirm = menu.addAction("确认")
            confirm.triggered.connect(lambda: self._toggle_confirm(ann, True))

        menu.addSeparator()

        copy_ann = menu.addAction("复制标注 (Ctrl+C)")
        copy_ann.triggered.connect(lambda: self.annotation_copied.emit(ann.id))

        delete = menu.addAction("删除")
        delete.triggered.connect(lambda: self.annotation_deleted.emit(ann.id))

        menu.exec_(event.globalPos())

    def _add_tool_mode_actions(self, menu: QMenu) -> None:
        """Add tool switching actions to a canvas context menu."""
        tool_group = QActionGroup(menu)
        tool_group.setExclusive(True)

        tools = [
            ("select", "移动", "cursor"),
            ("draw_bbox", "矩形框", "bbox"),
            ("draw_polygon", "多边形", "polygon"),
        ]
        if self._obb_editing_enabled:
            tools.append(("draw_obb", "旋转框", "obb"))
        tools.append(("draw_keypoint", "关键点", "keypoint"))
        for mode, text, icon_name in tools:
            action = menu.addAction(icon(icon_name), text)
            action.setCheckable(True)
            action.setChecked(self.tool_mode == mode)
            action.triggered.connect(
                lambda _checked=False, m=mode: self.request_tool_mode(m)
            )
            tool_group.addAction(action)

    def _is_image_pixel(self, px: float, py: float) -> bool:
        """Return whether widget pixel coordinates are inside the displayed image."""
        if self._image is None:
            return False

        nx, ny = self.pixel_to_norm(px, py)
        return 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0

    def resizeEvent(self, event: QResizeEvent) -> None:
        if self._image and not self._draw_start:
            self._center_image()
        super().resizeEvent(event)

    def keyPressEvent(self, event) -> None:
        if self.tool_mode == "draw_polygon":
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self.finish_polygon()
                return
            if event.key() == Qt.Key_Escape:
                self.cancel_polygon()
                return
        event.ignore()



    def _handle_drag(self, nx: float, ny: float) -> None:
        """Handle ongoing drag operation."""
        if not self._drag_ann_id or not self._drag_start_norm:
            return

        ann = None
        for a in self._annotations:
            if a.id == self._drag_ann_id:
                ann = a
                break
        if ann is None:
            return

        dx = nx - self._drag_start_norm[0]
        dy = ny - self._drag_start_norm[1]

        if self._drag_type == "move" and self._drag_ann_snapshot:
            if ann.bbox:
                orig_bbox = self._drag_ann_snapshot["bbox"]
                new_cx = orig_bbox[0] + dx
                new_cy = orig_bbox[1] + dy
                w, h = orig_bbox[2], orig_bbox[3]
                new_cx = max(w / 2, min(1.0 - w / 2, new_cx))
                new_cy = max(h / 2, min(1.0 - h / 2, new_cy))
                ann.bbox = (new_cx, new_cy, w, h)
            if ann.polygon:
                orig_polygon = self._drag_ann_snapshot.get("polygon", [])
                if orig_polygon:
                    min_x = min(p[0] for p in orig_polygon)
                    max_x = max(p[0] for p in orig_polygon)
                    min_y = min(p[1] for p in orig_polygon)
                    max_y = max(p[1] for p in orig_polygon)
                    pdx = max(-min_x, min(1.0 - max_x, dx))
                    pdy = max(-min_y, min(1.0 - max_y, dy))
                    ann.polygon = [(x + pdx, y + pdy) for x, y in orig_polygon]
            # Move keypoints by same offset
            if "keypoints" in self._drag_ann_snapshot:
                for i, kp_dict in enumerate(self._drag_ann_snapshot["keypoints"]):
                    if i < len(ann.keypoints):
                        ann.keypoints[i].x = max(0, min(1, kp_dict["x"] + dx))
                        ann.keypoints[i].y = max(0, min(1, kp_dict["y"] + dy))
            if not ann.confirmed:
                ann.confirmed = True

        elif self._drag_type == "rotate_obb" and self._drag_ann_snapshot:
            original = self._drag_ann_snapshot.get("polygon", [])
            if len(original) != 4:
                return
            center_x = sum(point[0] for point in original) / 4
            center_y = sum(point[1] for point in original) / 4
            image_w = max(1, self._image_w)
            image_h = max(1, self._image_h)
            start_x = (self._drag_start_norm[0] - center_x) * image_w
            start_y = (self._drag_start_norm[1] - center_y) * image_h
            current_x = (nx - center_x) * image_w
            current_y = (ny - center_y) * image_h
            if math.hypot(start_x, start_y) < 1e-9 or math.hypot(current_x, current_y) < 1e-9:
                return
            angle = math.atan2(current_y, current_x) - math.atan2(start_y, start_x)
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            rotated = []
            for point_x, point_y in original:
                offset_x = (point_x - center_x) * image_w
                offset_y = (point_y - center_y) * image_h
                rotated.append((
                    center_x + (offset_x * cos_a - offset_y * sin_a) / image_w,
                    center_y + (offset_x * sin_a + offset_y * cos_a) / image_h,
                ))
            min_x = min(point[0] for point in rotated)
            max_x = max(point[0] for point in rotated)
            min_y = min(point[1] for point in rotated)
            max_y = max(point[1] for point in rotated)
            shift_x = -min_x if min_x < 0 else (1 - max_x if max_x > 1 else 0)
            shift_y = -min_y if min_y < 0 else (1 - max_y if max_y > 1 else 0)
            ann.polygon = [(x + shift_x, y + shift_y) for x, y in rotated]
            self._sync_polygon_bbox(ann)
            if not ann.confirmed:
                ann.confirmed = True

        elif self._drag_type.startswith("resize_obb_") and self._drag_ann_snapshot:
            try:
                vertex_idx = int(self._drag_type.rsplit("_", 1)[1])
            except ValueError:
                return
            self._resize_obb_corner(ann, vertex_idx, nx, ny)
            if not ann.confirmed:
                ann.confirmed = True

        elif self._drag_type.startswith("poly_vertex_"):
            try:
                vertex_idx = int(self._drag_type.rsplit("_", 1)[1])
            except ValueError:
                return
            if 0 <= vertex_idx < len(ann.polygon):
                ann.polygon[vertex_idx] = self._clamp_norm(nx, ny)
                self._sync_polygon_bbox(ann)
                if not ann.confirmed:
                    ann.confirmed = True

        elif self._drag_type == "move_kp":
            if 0 <= self._drag_kp_idx < len(ann.keypoints):
                ann.keypoints[self._drag_kp_idx].x = max(0.0, min(1.0, nx))
                ann.keypoints[self._drag_kp_idx].y = max(0.0, min(1.0, ny))
                if not ann.confirmed:
                    ann.confirmed = True

        elif self._drag_type.startswith("resize_") and ann.bbox and self._drag_ann_snapshot:
            orig_bbox = self._drag_ann_snapshot["bbox"]
            ocx, ocy, ow, oh = orig_bbox
            ox1, oy1 = ocx - ow / 2, ocy - oh / 2
            ox2, oy2 = ocx + ow / 2, ocy + oh / 2
            min_w, min_h = self._minimum_bbox_size_norm()

            if "tl" in self._drag_type:
                ox1 = max(0.0, min(ox2 - min_w, ox1 + dx))
                oy1 = max(0.0, min(oy2 - min_h, oy1 + dy))
            elif "tr" in self._drag_type:
                ox2 = max(ox1 + min_w, min(1.0, ox2 + dx))
                oy1 = max(0.0, min(oy2 - min_h, oy1 + dy))
            elif "bl" in self._drag_type:
                ox1 = max(0.0, min(ox2 - min_w, ox1 + dx))
                oy2 = max(oy1 + min_h, min(1.0, oy2 + dy))
            elif "br" in self._drag_type:
                ox2 = max(ox1 + min_w, min(1.0, ox2 + dx))
                oy2 = max(oy1 + min_h, min(1.0, oy2 + dy))

            ann.bbox = ((ox1 + ox2) / 2, (oy1 + oy2) / 2, ox2 - ox1, oy2 - oy1)
            if not ann.confirmed:
                ann.confirmed = True

    def _resize_obb_corner(
        self,
        ann: Annotation,
        vertex_idx: int,
        nx: float,
        ny: float,
    ) -> None:
        """Resize an OBB from one corner while preserving a true rectangle.

        The opposite corner stays fixed. The two adjacent corners follow the
        dragged corner along the box's local axes, matching bbox corner-resize
        interaction without allowing an arbitrary four-sided polygon.
        """
        if not self._drag_ann_snapshot or not 0 <= vertex_idx < 4:
            return
        original = self._drag_ann_snapshot.get("polygon", [])
        if len(original) != 4:
            return

        image_w = max(1, self._image_w)
        image_h = max(1, self._image_h)

        def to_image_point(point) -> tuple[float, float]:
            return point[0] * image_w, point[1] * image_h

        points = [to_image_point(point) for point in original]
        opposite_idx = (vertex_idx + 2) % 4
        next_idx = (vertex_idx + 1) % 4
        previous_idx = (vertex_idx - 1) % 4
        anchor_x, anchor_y = points[opposite_idx]

        next_vector = (
            points[next_idx][0] - anchor_x,
            points[next_idx][1] - anchor_y,
        )
        previous_vector = (
            points[previous_idx][0] - anchor_x,
            points[previous_idx][1] - anchor_y,
        )
        next_length = math.hypot(*next_vector)
        if next_length < 1e-9:
            return
        next_axis = (
            next_vector[0] / next_length,
            next_vector[1] / next_length,
        )

        # Gram-Schmidt also repairs slightly malformed legacy OBBs on resize.
        projection = (
            previous_vector[0] * next_axis[0]
            + previous_vector[1] * next_axis[1]
        )
        orthogonal_previous = (
            previous_vector[0] - projection * next_axis[0],
            previous_vector[1] - projection * next_axis[1],
        )
        previous_length = math.hypot(*orthogonal_previous)
        if previous_length < 1e-9:
            return
        previous_axis = (
            orthogonal_previous[0] / previous_length,
            orthogonal_previous[1] / previous_length,
        )
        if (
            previous_axis[0] * previous_vector[0]
            + previous_axis[1] * previous_vector[1]
        ) < 0:
            previous_axis = (-previous_axis[0], -previous_axis[1])

        clamped_x, clamped_y = self._clamp_norm(nx, ny)
        delta = (
            clamped_x * image_w - anchor_x,
            clamped_y * image_h - anchor_y,
        )
        next_size = max(
            1.0,
            delta[0] * next_axis[0] + delta[1] * next_axis[1],
        )
        previous_size = max(
            1.0,
            delta[0] * previous_axis[0] + delta[1] * previous_axis[1],
        )

        def ray_limit(axis: tuple[float, float]) -> float:
            limits: list[float] = []
            if axis[0] > 1e-12:
                limits.append((image_w - anchor_x) / axis[0])
            elif axis[0] < -1e-12:
                limits.append(-anchor_x / axis[0])
            if axis[1] > 1e-12:
                limits.append((image_h - anchor_y) / axis[1])
            elif axis[1] < -1e-12:
                limits.append(-anchor_y / axis[1])
            return max(0.0, min(limits)) if limits else float("inf")

        next_size = min(next_size, ray_limit(next_axis))
        previous_size = min(previous_size, ray_limit(previous_axis))

        def resized_points(size_a: float, size_b: float) -> dict[int, tuple[float, float]]:
            next_point = (
                anchor_x + next_axis[0] * size_a,
                anchor_y + next_axis[1] * size_a,
            )
            previous_point = (
                anchor_x + previous_axis[0] * size_b,
                anchor_y + previous_axis[1] * size_b,
            )
            dragged_point = (
                next_point[0] + previous_point[0] - anchor_x,
                next_point[1] + previous_point[1] - anchor_y,
            )
            return {
                opposite_idx: (anchor_x, anchor_y),
                next_idx: next_point,
                previous_idx: previous_point,
                vertex_idx: dragged_point,
            }

        def points_fit(candidate: dict[int, tuple[float, float]]) -> bool:
            return all(
                -1e-9 <= x <= image_w + 1e-9
                and -1e-9 <= y <= image_h + 1e-9
                for x, y in candidate.values()
            )

        candidate = resized_points(next_size, previous_size)
        if not points_fit(candidate):
            # A dragged corner can be inside while an adjacent corner crosses
            # the image edge. Move toward the one-pixel minimum rectangle until
            # every corner is valid; the feasible image region is convex.
            minimum = resized_points(1.0, 1.0)
            if not points_fit(minimum):
                return
            low, high = 0.0, 1.0
            for _ in range(32):
                mid = (low + high) / 2.0
                size_a = 1.0 + (next_size - 1.0) * mid
                size_b = 1.0 + (previous_size - 1.0) * mid
                if points_fit(resized_points(size_a, size_b)):
                    low = mid
                else:
                    high = mid
            next_size = 1.0 + (next_size - 1.0) * low
            previous_size = 1.0 + (previous_size - 1.0) * low
            candidate = resized_points(next_size, previous_size)

        ann.polygon = [
            (candidate[index][0] / image_w, candidate[index][1] / image_h)
            for index in range(4)
        ]
        self._sync_polygon_bbox(ann)

    def _set_polygon_edge_hover(
        self,
        edge_index: int,
        point: tuple[float, float],
    ) -> None:
        """Store the currently hovered polygon edge insertion point."""
        changed = (
            self._hover_poly_edge_index != edge_index
            or self._hover_poly_edge_point != point
        )
        self._hover_poly_edge_index = edge_index
        self._hover_poly_edge_point = point
        if changed:
            self.update()

    def _clear_polygon_edge_hover(self) -> None:
        """Clear the edge insertion preview handle."""
        if (
            self._hover_poly_edge_index is not None
            or self._hover_poly_edge_point is not None
        ):
            self._hover_poly_edge_index = None
            self._hover_poly_edge_point = None
            self.update()

    def _hit_test_polygon_edge(
        self,
        px: float,
        py: float,
    ) -> tuple[int, tuple[float, float]] | None:
        """Return the nearest edge and projection point for a selected polygon.

        The distance test is performed in widget pixels, so the interaction
        tolerance remains stable at every zoom level. Existing vertex handles
        are checked before this method by the mouse handlers.
        """
        ann = self.get_selected_annotation()
        if ann is None or len(ann.polygon) < 3:
            return None
        if (
            (self._polygon_point_limit is not None and len(ann.polygon) >= self._polygon_point_limit)
            or self._is_obb_annotation(ann)
        ):
            return None

        edge_hit = self._nearest_polygon_edge(ann.polygon, px, py)
        if edge_hit is None:
            return None
        best_edge, best_distance_sq, best_pixel_point = edge_hit

        if (
            best_edge < 0
            or best_pixel_point is None
            or best_distance_sq > POLYGON_EDGE_HIT_RADIUS ** 2
        ):
            return None

        point = self._clamp_norm(
            *self.pixel_to_norm(*best_pixel_point)
        )
        return best_edge, point

    def _hit_test_polygon_region(
        self,
        ann: Annotation,
        px: float,
        py: float,
    ) -> bool:
        """Return whether a point is inside or close to a polygon's visible mask."""
        if len(ann.polygon) < 3:
            return False
        path = QPainterPath()
        first_x, first_y = self.norm_to_pixel(*ann.polygon[0])
        path.moveTo(first_x, first_y)
        for point in ann.polygon[1:]:
            path.lineTo(*self.norm_to_pixel(*point))
        path.closeSubpath()
        path.setFillRule(Qt.OddEvenFill)
        if path.contains(QPointF(px, py)):
            return True

        edge_hit = self._nearest_polygon_edge(ann.polygon, px, py)
        if edge_hit is None:
            return False
        _, distance_sq, _ = edge_hit
        return distance_sq <= POLYGON_EDGE_HIT_RADIUS ** 2

    def _nearest_polygon_edge(
        self,
        polygon: list[tuple[float, float]],
        px: float,
        py: float,
    ) -> tuple[int, float, tuple[float, float]] | None:
        """Return nearest polygon edge index, squared pixel distance, and point."""
        best_edge = -1
        best_distance_sq = float("inf")
        best_pixel_point: tuple[float, float] | None = None
        count = len(polygon)

        for edge_index in range(count):
            ax, ay = self.norm_to_pixel(*polygon[edge_index])
            bx, by = self.norm_to_pixel(*polygon[(edge_index + 1) % count])
            vx = bx - ax
            vy = by - ay
            length_sq = vx * vx + vy * vy
            if length_sq <= 1e-12:
                continue

            t = ((px - ax) * vx + (py - ay) * vy) / length_sq
            t = max(0.0, min(1.0, t))
            qx = ax + t * vx
            qy = ay + t * vy
            distance_sq = (px - qx) ** 2 + (py - qy) ** 2

            if distance_sq < best_distance_sq:
                best_edge = edge_index
                best_distance_sq = distance_sq
                best_pixel_point = (qx, qy)

        if best_edge < 0 or best_pixel_point is None:
            return None
        return best_edge, best_distance_sq, best_pixel_point

    @staticmethod
    def _sync_polygon_bbox(ann: Annotation) -> None:
        """Keep an existing polygon annotation bbox aligned with its vertices.

        Manually created polygon-only annotations keep ``bbox=None``. For model
        predictions that already carry a bbox, moving or inserting a polygon
        vertex updates that bbox to the polygon's new bounds.
        """
        if ann.bbox is None or not ann.polygon:
            return
        xs = [point[0] for point in ann.polygon]
        ys = [point[1] for point in ann.polygon]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        ann.bbox = (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
            x2 - x1,
            y2 - y1,
        )

    def _hit_test_handle(self, px: float, py: float) -> str | None:
        """Check if pixel pos hits a resize handle on the selected bbox."""
        if not self._selected_id:
            return None
        ann = self.get_selected_annotation()
        if not ann:
            return None

        is_obb = self._is_obb_annotation(ann)
        if is_obb:
            positions = self._obb_rotation_handle_positions(ann)
            if positions is not None:
                _edge_point, handle_point = positions
                if math.hypot(px - handle_point[0], py - handle_point[1]) <= OBB_ROTATION_HIT_RADIUS:
                    return "rotate_obb"

        if ann.polygon:
            for idx, (nx, ny) in enumerate(ann.polygon):
                hpx, hpy = self.norm_to_pixel(nx, ny)
                if abs(px - hpx) <= HANDLE_SIZE + 4 and abs(py - hpy) <= HANDLE_SIZE + 4:
                    return f"resize_obb_{idx}" if is_obb else f"poly_vertex_{idx}"
            if is_obb:
                return None

        if not ann.bbox:
            return None

        cx, cy, w, h = ann.bbox
        corners = {
            "resize_tl": (cx - w / 2, cy - h / 2),
            "resize_tr": (cx + w / 2, cy - h / 2),
            "resize_bl": (cx - w / 2, cy + h / 2),
            "resize_br": (cx + w / 2, cy + h / 2),
        }
        for handle_name, (nx, ny) in corners.items():
            hpx, hpy = self.norm_to_pixel(nx, ny)
            if abs(px - hpx) <= HANDLE_SIZE + 2 and abs(py - hpy) <= HANDLE_SIZE + 2:
                return handle_name
        return None

    def _is_obb_annotation(self, ann: Annotation) -> bool:
        """Return whether an annotation should use rigid OBB editing tools."""
        return (
            self._obb_editing_enabled
            and ann.bbox is not None
            and len(ann.polygon) == 4
        )

    def _obb_rotation_handle_positions(
        self,
        ann: Annotation,
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """Return the outward top-edge midpoint and rotation-handle position."""
        if len(ann.polygon) != 4:
            return None
        points = [self.norm_to_pixel(x, y) for x, y in ann.polygon]
        center_x = sum(point[0] for point in points) / 4
        center_y = sum(point[1] for point in points) / 4
        edge_midpoints = [
            (
                (points[index][0] + points[(index + 1) % 4][0]) / 2,
                (points[index][1] + points[(index + 1) % 4][1]) / 2,
            )
            for index in range(4)
        ]
        edge_x, edge_y = min(edge_midpoints, key=lambda point: point[1])
        vector_x = edge_x - center_x
        vector_y = edge_y - center_y
        length = math.hypot(vector_x, vector_y)
        if length < 1e-6:
            return None
        handle = (
            edge_x + vector_x / length * OBB_ROTATION_HANDLE_OFFSET,
            edge_y + vector_y / length * OBB_ROTATION_HANDLE_OFFSET,
        )
        return (edge_x, edge_y), handle

    def _hit_test_keypoint(self, px: float, py: float) -> tuple[str, int] | None:
        """Check if pixel pos hits a keypoint. Returns (ann_id, kp_index) or None."""
        for ann in reversed(self._annotations):
            for i, kp in enumerate(ann.keypoints):
                kpx, kpy = self.norm_to_pixel(kp.x, kp.y)
                if abs(px - kpx) <= KEYPOINT_RADIUS + 4 and abs(py - kpy) <= KEYPOINT_RADIUS + 4:
                    return ann.id, i
        return None

    def _near_polygon_first_point(self, px: float, py: float) -> bool:
        if not self._polygon_points:
            return False
        first_px, first_py = self.norm_to_pixel(*self._polygon_points[0])
        return abs(px - first_px) <= HANDLE_SIZE + 8 and abs(py - first_py) <= HANDLE_SIZE + 8

    @staticmethod
    def _distance_norm(a: tuple[float, float], b: tuple[float, float]) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    @staticmethod
    def _point_in_polygon(nx: float, ny: float, polygon: list[tuple[float, float]]) -> bool:
        inside = False
        count = len(polygon)
        if count < 3:
            return False
        j = count - 1
        for i in range(count):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            intersects = ((yi > ny) != (yj > ny)) and (
                nx < (xj - xi) * (ny - yi) / ((yj - yi) or 1e-12) + xi
            )
            if intersects:
                inside = not inside
            j = i
        return inside

    @staticmethod
    def _annotation_bounds(ann: Annotation) -> tuple[float, float, float, float] | None:
        xs: list[float] = []
        ys: list[float] = []
        if ann.bbox:
            cx, cy, w, h = ann.bbox
            xs.extend([cx - w / 2, cx + w / 2])
            ys.extend([cy - h / 2, cy + h / 2])
        if ann.polygon:
            xs.extend([p[0] for p in ann.polygon])
            ys.extend([p[1] for p in ann.polygon])
        if not xs or not ys:
            return None
        return min(xs), min(ys), max(xs), max(ys)

    def _toggle_confirm(self, ann: Annotation, confirmed: bool) -> None:
        ann.confirmed = confirmed
        self.annotations_changed.emit()
        self.update()

    def _request_rename_keypoint(self, ann_id: str, kp_idx: int) -> None:
        """Show input dialog to rename a keypoint."""
        ann = next((a for a in self._annotations if a.id == ann_id), None)
        if not ann or kp_idx >= len(ann.keypoints):
            return
        old_label = ann.keypoints[kp_idx].label
        new_label, ok = QInputDialog.getText(
            self, "重命名关键点", "标签:", text=old_label)
        if ok and new_label.strip():
            self.rename_keypoint(ann_id, kp_idx, new_label.strip())

    def finish_polygon(self) -> None:
        """Request class selection for the in-progress polygon."""
        if len(self._polygon_points) < 3:
            return
        if self._draw_current is not None and self._distance_norm(self._polygon_points[-1], self._draw_current) > 0.001:
            self._polygon_points.append(self._draw_current)
        if len(self._polygon_points) >= 2 and self._distance_norm(self._polygon_points[0], self._polygon_points[-1]) < 0.001:
            self._polygon_points.pop()
        if len(self._polygon_points) < 3:
            return
        px, py = self.norm_to_pixel(*self._polygon_points[-1])
        self.class_requested.emit(px, py)

    def cancel_polygon(self) -> None:
        self._polygon_points = []
        self._draw_current = None
        self.update()



    def create_bbox_from_draw(self, class_name: str, class_id: int) -> Annotation | None:
        """Create a bbox annotation from the last draw operation."""
        if not self._draw_start or not self._draw_current:
            return None
        sx, sy = self._draw_start
        ex, ey = self._draw_current
        w = abs(ex - sx)
        h = abs(ey - sy)
        min_w, min_h = self._minimum_bbox_size_norm()
        if w < min_w or h < min_h:
            return None
        cx = (sx + ex) / 2
        cy = (sy + ey) / 2
        ann = Annotation(
            class_name=class_name,
            class_id=class_id,
            bbox=(cx, cy, w, h),
            confirmed=True,
            source="manual",
        )
        self._annotations.append(ann)
        self.select_annotation(ann.id)
        self.annotation_created.emit(ann)
        self.annotations_changed.emit()
        self._draw_start = None
        self._draw_current = None
        self.update()
        return ann

    def create_obb_from_draw(self, class_name: str, class_id: int) -> Annotation | None:
        """Create an axis-aligned OBB that can subsequently be rotated."""
        if not self._draw_start or not self._draw_current:
            return None
        sx, sy = self._draw_start
        ex, ey = self._draw_current
        x1, x2 = sorted((sx, ex))
        y1, y2 = sorted((sy, ey))
        w = x2 - x1
        h = y2 - y1
        min_w, min_h = self._minimum_bbox_size_norm()
        if w < min_w or h < min_h:
            return None

        polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        ann = Annotation(
            class_name=class_name,
            class_id=class_id,
            bbox=((x1 + x2) / 2, (y1 + y2) / 2, w, h),
            polygon=polygon,
            confirmed=True,
            source="manual",
        )
        ann.clamp()
        self._annotations.append(ann)
        self.select_annotation(ann.id)
        self.annotation_created.emit(ann)
        self.annotations_changed.emit()
        self._draw_start = None
        self._draw_current = None
        self.update()
        return ann

    def create_polygon_from_draw(self, class_name: str, class_id: int) -> Annotation | None:
        """Create a polygon annotation from the in-progress point list."""
        polygon = list(self._polygon_points)
        if len(polygon) >= 2 and self._distance_norm(polygon[0], polygon[-1]) < 0.001:
            polygon.pop()
        if len(polygon) < 3:
            self.cancel_polygon()
            return None
        if self._polygon_point_limit is not None and len(polygon) != self._polygon_point_limit:
            self.cancel_polygon()
            return None
        if self._polygon_point_limit == 4:
            center_x = sum(point[0] for point in polygon) / 4
            center_y = sum(point[1] for point in polygon) / 4
            polygon.sort(key=lambda point: math.atan2(point[1] - center_y, point[0] - center_x))
        bbox = None
        if self._polygon_point_limit == 4:
            xs = [point[0] for point in polygon]
            ys = [point[1] for point in polygon]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            bbox = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
        ann = Annotation(
            class_name=class_name,
            class_id=class_id,
            bbox=bbox,
            polygon=polygon,
            confirmed=True,
            source="manual",
        )
        ann.clamp()
        self._annotations.append(ann)
        self.select_annotation(ann.id)
        self.annotation_created.emit(ann)
        self.annotations_changed.emit()
        self._polygon_points = []
        self._draw_current = None
        self.update()
        return ann

    def create_keypoint_at(
        self, class_name: str, class_id: int, label: str = "point"
    ) -> Annotation | None:
        """Create a keypoint annotation at the stored draw position."""
        if not self._draw_start:
            return None
        nx, ny = self._draw_start
        kp = Keypoint(x=nx, y=ny, visible=2, label=label)
        ann = Annotation(
            class_name=class_name,
            class_id=class_id,
            keypoints=[kp],
            confirmed=True,
            source="manual",
        )
        self._annotations.append(ann)
        self.select_annotation(ann.id)
        self.annotation_created.emit(ann)
        self.annotations_changed.emit()
        self._draw_start = None
        self.update()
        return ann

    def select_keypoint(self, ann_id: str, kp_idx: int) -> None:
        """Select a specific keypoint within an annotation."""
        self._selected_id = ann_id
        self._selected_kp_idx = kp_idx
        self.annotation_selected.emit(ann_id)
        self.keypoint_selected.emit(ann_id, kp_idx)
        self.update()

    def add_keypoint_to_annotation(self, ann_id: str, kp: Keypoint) -> None:
        """Append a keypoint to an existing annotation."""
        for ann in self._annotations:
            if ann.id == ann_id:
                ann.keypoints.append(kp)
                if not ann.confirmed:
                    ann.confirmed = True
                self.annotation_modified.emit(ann_id)
                self.annotations_changed.emit()
                self.update()
                return

    def remove_keypoint(self, ann_id: str, kp_idx: int) -> None:
        """Remove a single keypoint from an annotation.

        If the annotation has no bbox and this is the last keypoint, remove the annotation.
        """
        for ann in self._annotations:
            if ann.id == ann_id:
                if 0 <= kp_idx < len(ann.keypoints):
                    ann.keypoints.pop(kp_idx)
                    if not ann.bbox and not ann.keypoints:
                        self._annotations = [a for a in self._annotations if a.id != ann_id]
                        self.annotation_deleted.emit(ann_id)
                    else:
                        self.annotation_modified.emit(ann_id)
                    if self._selected_kp_idx is not None and self._selected_kp_idx >= len(ann.keypoints):
                        self._selected_kp_idx = None
                    self.annotations_changed.emit()
                    self.update()
                return

    def rename_keypoint(self, ann_id: str, kp_idx: int, new_label: str) -> None:
        """Rename a keypoint's label."""
        for ann in self._annotations:
            if ann.id == ann_id:
                if 0 <= kp_idx < len(ann.keypoints):
                    ann.keypoints[kp_idx].label = new_label
                    self.annotation_modified.emit(ann_id)
                    self.annotations_changed.emit()
                    self.update()
                return

    def cycle_keypoint_visibility(self, ann_id: str, kp_idx: int) -> None:
        """Cycle keypoint visibility: 0 -> 1 -> 2 -> 0."""
        for ann in self._annotations:
            if ann.id == ann_id:
                if 0 <= kp_idx < len(ann.keypoints):
                    ann.keypoints[kp_idx].visible = (ann.keypoints[kp_idx].visible + 1) % 3
                    self.annotation_modified.emit(ann_id)
                    self.annotations_changed.emit()
                    self.update()
                return
