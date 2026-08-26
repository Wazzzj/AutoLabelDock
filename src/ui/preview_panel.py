"""Read-only project preview grid with annotation overlays."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PyQt5.QtCore import Qt, QPointF, QRect, QRectF, QSize, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStyle,
    QStyledItemDelegate,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from src.core.annotation import (
    ImageAnnotation,
    POSE_BOUNDING_BOX_RGB,
    annotation_center,
    annotation_display_label,
    annotation_pixel_geometry,
    keypoint_display_rgb,
    keypoint_skeleton_colored_segments,
)
from src.core.annotation_classes import merged_project_annotation_classes
from src.core.label_io import load_annotation
from src.core.project import ProjectManager
from src.core.tags import TagFilter
from src.ui.icons import icon
from src.ui.tag_widget import TagFilterBar
from src.ui.theme import PALETTE, set_button_role
from src.ui.views.thumbnail_loader import ThumbnailLoader
from src.utils.image import get_image_size, load_pixmap


_PATH_ROLE = Qt.UserRole
_ANNOTATION_ROLE = Qt.UserRole + 1
_SUMMARY_ROLE = Qt.UserRole + 2
_ROI_ROLE = Qt.UserRole + 3
_LABEL_BAR_H = 22
_CARD_PAD = 8
_ZOOM_FACTOR = 1.15
_ROI_CLOSE_RADIUS_PX = 12
_ALL_DATA_FOLDERS_TEXT = "所有版本"


@dataclass(frozen=True)
class PreviewSummary:
    text: str
    status: str
    color: str


@dataclass(frozen=True)
class NumericRange:
    """Optional inclusive numeric range used by preview filters."""

    minimum: float | None = None
    maximum: float | None = None

    def is_active(self) -> bool:
        return self.minimum is not None or self.maximum is not None

    def contains(self, value: float) -> bool:
        if self.minimum is not None and value < self.minimum:
            return False
        if self.maximum is not None and value > self.maximum:
            return False
        return True


@dataclass(frozen=True)
class PreviewRoi:
    """Session-only normalized ROI used by preview analysis, never training."""

    shape: str
    points: tuple[tuple[float, float], ...]


def _point_in_polygon(
    point: tuple[float, float],
    polygon: tuple[tuple[float, float], ...],
) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        if (y1 > y) != (y2 > y):
            intersection_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x <= intersection_x:
                inside = not inside
        previous = current
    return inside


def _roi_contains_point(roi: PreviewRoi, point: tuple[float, float]) -> bool:
    if roi.shape in {"rectangle", "ellipse"} and len(roi.points) == 2:
        (x1, y1), (x2, y2) = roi.points
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        if roi.shape == "rectangle":
            return left <= point[0] <= right and top <= point[1] <= bottom
        radius_x = (right - left) / 2.0
        radius_y = (bottom - top) / 2.0
        if radius_x <= 0 or radius_y <= 0:
            return False
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        return (
            ((point[0] - center_x) / radius_x) ** 2
            + ((point[1] - center_y) / radius_y) ** 2
            <= 1.0
        )
    if roi.shape == "polygon":
        return _point_in_polygon(point, roi.points)
    return False


def _annotation_inside_rois(annotation, rois: list[PreviewRoi]) -> bool:
    """Use bbox center or polygon centroid for ROI membership."""
    if not rois:
        return True
    center = annotation_center(annotation)
    if center is None:
        return False
    return any(_roi_contains_point(roi, center) for roi in rois)


@dataclass(frozen=True)
class PreviewAdvancedFilter:
    """Numeric acceptance constraints for one detection-box class."""

    width: NumericRange = field(default_factory=NumericRange)
    height: NumericRange = field(default_factory=NumericRange)
    area: NumericRange = field(default_factory=NumericRange)
    confidence: NumericRange = field(default_factory=NumericRange)
    center_x: NumericRange = field(default_factory=NumericRange)
    center_y: NumericRange = field(default_factory=NumericRange)

    def annotation_ranges(self) -> tuple[NumericRange, ...]:
        return (
            self.width,
            self.height,
            self.area,
            self.confidence,
            self.center_x,
            self.center_y,
        )

    def has_annotation_constraints(self) -> bool:
        return any(value.is_active() for value in self.annotation_ranges())

    def active_count(self) -> int:
        return sum(value.is_active() for value in self.annotation_ranges())


def _annotation_matches_advanced_filter(
    annotation,
    image_size: tuple[int, int],
    advanced_filter: PreviewAdvancedFilter,
) -> bool:
    if (
        advanced_filter.confidence.is_active()
        and not advanced_filter.confidence.contains(annotation.confidence)
    ):
        return False

    geometry_ranges = (
        advanced_filter.width,
        advanced_filter.height,
        advanced_filter.area,
        advanced_filter.center_x,
        advanced_filter.center_y,
    )
    if not any(value.is_active() for value in geometry_ranges):
        return True
    geometry = annotation_pixel_geometry(annotation, image_size)
    if geometry is None:
        return False
    return all(
        not numeric_range.is_active() or numeric_range.contains(value)
        for numeric_range, value in zip(geometry_ranges, geometry)
    )


def _annotation_control_result(
    annotation,
    image_size: tuple[int, int],
    control_rules: dict[str, PreviewAdvancedFilter],
    enabled: bool = True,
    rois: list[PreviewRoi] | None = None,
) -> bool | None:
    """Return True/False for a controlled detection box, otherwise None."""
    if not enabled:
        return None
    if annotation_pixel_geometry(annotation, image_size) is None:
        return None
    if rois and not _annotation_inside_rois(annotation, rois):
        return None
    rule = control_rules.get(annotation.class_name)
    if rule is None or not rule.has_annotation_constraints():
        # A global ROI is itself a control rule: every detection inside it
        # passes the ROI gate even when its class has no additional limits.
        return True if rois else None
    return _annotation_matches_advanced_filter(annotation, image_size, rule)


def _image_control_result(
    annotation: ImageAnnotation,
    control_rules: dict[str, PreviewAdvancedFilter],
    enabled: bool = True,
    rois: list[PreviewRoi] | None = None,
) -> bool | None:
    """Return NG when any controlled detection box fails, OK when all pass."""
    results = [
        result
        for item in annotation.annotations
        if (
            result := _annotation_control_result(
                item,
                annotation.image_size,
                control_rules,
                enabled,
                rois,
            )
        ) is not None
    ]
    if not results:
        # With a global ROI enabled, no in-ROI detection means the inspected
        # region contains no defect and therefore passes this control gate.
        return True if enabled and rois else None
    return all(results)


def _annotation_control_pen_style(
    annotation,
    image_size: tuple[int, int],
    control_rules: dict[str, PreviewAdvancedFilter],
    enabled: bool,
    rois: list[PreviewRoi] | None = None,
):
    """Return solid for passing boxes and dashed for failing boxes."""
    result = _annotation_control_result(
        annotation,
        image_size,
        control_rules,
        enabled,
        rois,
    )
    if rois and not _annotation_inside_rois(annotation, rois):
        return Qt.DashLine
    if result is False or (result is None and not annotation.confirmed):
        return Qt.DashLine
    return Qt.SolidLine


def _image_matches_annotation_filters(
    annotation: ImageAnnotation,
    class_filter: str | None,
) -> bool:
    """Match the ordinary class filter without treating control as filtering."""
    image_class_match = bool(
        class_filter is not None and class_filter in annotation.image_tags
    )
    candidates = [item for item in annotation.annotations if item.class_name == class_filter]
    if class_filter is not None:
        return image_class_match or bool(candidates)
    return True


def _summary_for_annotation(
    ia: ImageAnnotation,
    class_colors: dict[str, str],
    default_color: str = "#6c7086",
) -> PreviewSummary:
    """Return compact status text for a preview tile."""
    if ia.image_tags:
        cls = ia.image_tags[0]
        status = "confirmed" if ia.image_tags_confirmed else "pending"
        suffix = "已确认" if ia.image_tags_confirmed else "待确认"
        return PreviewSummary(
            text=f"{cls} · {suffix}",
            status=status,
            color=class_colors.get(cls, default_color),
        )

    if not ia.annotations:
        return PreviewSummary("未标注", "unlabeled", default_color)

    confirmed = sum(1 for ann in ia.annotations if ann.confirmed)
    total = len(ia.annotations)
    status = "confirmed" if confirmed == total else "pending"
    suffix = "已确认" if confirmed == total else f"待确认 {total - confirmed}"
    return PreviewSummary(
        text=f"{total} 个标注 · {suffix}",
        status=status,
        color=PALETTE["success"] if status == "confirmed" else PALETTE["warning"],
    )


def _norm_xy(rect: QRect, nx: float, ny: float) -> tuple[float, float]:
    return rect.x() + nx * rect.width(), rect.y() + ny * rect.height()


def _norm_point(rect: QRect, nx: float, ny: float) -> QPointF:
    x, y = _norm_xy(rect, nx, ny)
    return QPointF(x, y)


def _draw_annotation_overlays(
    painter: QPainter,
    image_rect: QRect,
    annotation: ImageAnnotation,
    class_colors: dict[str, str],
    stroke_scale: float = 1.0,
    control_rules: dict[str, PreviewAdvancedFilter] | None = None,
    control_enabled: bool = False,
    rois: list[PreviewRoi] | None = None,
) -> None:
    """Draw saved detect/pose annotations over an image rect."""
    painter.setRenderHint(QPainter.Antialiasing)
    line_width = max(2, int(2 * stroke_scale))
    point_radius = max(3, int(4 * stroke_scale))
    for ann in annotation.annotations:
        color = QColor(class_colors.get(ann.class_name, PALETTE["primary"]))
        label_anchor: tuple[float, float] | None = None
        if ann.polygon:
            polygon = QPolygonF([
                _norm_point(image_rect, x, y)
                for x, y in ann.polygon
            ])
            label_anchor = (
                min(point.x() for point in polygon),
                min(point.y() for point in polygon),
            )
            pen = QPen(color, line_width)
            pen.setStyle(_annotation_control_pen_style(
                ann, annotation.image_size, control_rules or {}, control_enabled, rois
            ))
            painter.setPen(pen)
            # Preview overlays keep the detection area fully transparent so
            # image details remain unobstructed; only the outline is rendered.
            painter.setBrush(Qt.NoBrush)
            painter.drawPolygon(polygon)

        if ann.bbox and not ann.polygon:
            cx, cy, w, h = ann.bbox
            x1, y1 = _norm_xy(image_rect, cx - w / 2, cy - h / 2)
            x2, y2 = _norm_xy(image_rect, cx + w / 2, cy + h / 2)
            bbox_color = QColor(*POSE_BOUNDING_BOX_RGB) if ann.keypoints else color
            pen = QPen(bbox_color, line_width)
            pen.setStyle(_annotation_control_pen_style(
                ann, annotation.image_size, control_rules or {}, control_enabled, rois
            ))
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
            label_anchor = (x1, y1)

        if ann.keypoints:
            painter.setBrush(Qt.NoBrush)
            fallback_rgb = (color.red(), color.green(), color.blue())
            for start, end, rgb in keypoint_skeleton_colored_segments(
                ann.keypoints, fallback_rgb
            ):
                skeleton_pen = QPen(QColor(*rgb), line_width)
                skeleton_pen.setCapStyle(Qt.RoundCap)
                painter.setPen(skeleton_pen)
                painter.drawLine(
                    _norm_point(image_rect, start.x, start.y),
                    _norm_point(image_rect, end.x, end.y),
                )

        for index, kp in enumerate(ann.keypoints):
            x, y = _norm_xy(image_rect, kp.x, kp.y)
            point_rgb = keypoint_display_rgb(
                ann.keypoints, index, (color.red(), color.green(), color.blue())
            )
            point_color = QColor(*point_rgb)
            if kp.visible == 0:
                painter.setPen(QPen(QColor(PALETTE["text_subtle"]), line_width))
                painter.setBrush(Qt.NoBrush)
            elif kp.visible == 1:
                painter.setPen(QPen(point_color, line_width))
                painter.setBrush(Qt.NoBrush)
            else:
                painter.setPen(QPen(point_color, line_width))
                painter.setBrush(point_color)
            painter.drawEllipse(
                int(x - point_radius),
                int(y - point_radius),
                point_radius * 2,
                point_radius * 2,
            )

        if label_anchor is None and ann.keypoints:
            label_anchor = _norm_xy(
                image_rect,
                ann.keypoints[0].x,
                ann.keypoints[0].y,
            )
        if label_anchor is not None:
            _draw_annotation_label(
                painter,
                image_rect,
                label_anchor,
                annotation_display_label(
                    ann,
                    annotation.image_size,
                    include_pixels=image_rect.width() >= 400,
                ),
                color,
                stroke_scale,
            )

    control_result = _image_control_result(
        annotation,
        control_rules or {},
        control_enabled,
        rois,
    )
    if control_result is not None:
        _draw_control_badge(painter, image_rect, control_result, stroke_scale)

    _draw_preview_rois(painter, image_rect, rois or [], stroke_scale)


def _draw_preview_rois(
    painter: QPainter,
    image_rect: QRect,
    rois: list[PreviewRoi],
    stroke_scale: float = 1.0,
) -> None:
    """Draw session-only ROI outlines without filling the underlying image."""
    if not rois:
        return
    color = QColor(PALETTE["warning"])
    painter.setPen(QPen(color, max(2, round(2 * stroke_scale)), Qt.DashDotLine))
    painter.setBrush(Qt.NoBrush)
    for roi in rois:
        if roi.shape in {"rectangle", "ellipse"} and len(roi.points) == 2:
            start = _norm_point(image_rect, *roi.points[0])
            end = _norm_point(image_rect, *roi.points[1])
            rect = QRectF(start, end).normalized()
            if roi.shape == "rectangle":
                painter.drawRect(rect)
            else:
                painter.drawEllipse(rect)
        elif roi.shape == "polygon" and len(roi.points) >= 3:
            painter.drawPolygon(QPolygonF([
                _norm_point(image_rect, *point) for point in roi.points
            ]))


def _draw_control_badge(
    painter: QPainter,
    image_rect: QRect,
    passed: bool,
    stroke_scale: float = 1.0,
) -> None:
    """Draw the aggregate detection-control result at the image top-left."""
    text = "OK" if passed else "NG"
    color = QColor(PALETTE["success"] if passed else PALETTE["danger"])
    font = QFont()
    font.setPixelSize(max(13, min(28, round(15 * stroke_scale ** 0.2))))
    font.setBold(True)
    painter.setFont(font)
    metrics = painter.fontMetrics()
    width = metrics.horizontalAdvance(text) + 14
    height = metrics.height() + 6
    rect = QRect(image_rect.left() + 4, image_rect.top() + 4, width, height)
    background = QColor(color)
    background.setAlpha(225)
    painter.fillRect(rect, background)
    painter.setPen(QColor(PALETTE["ink"]))
    painter.drawText(rect, Qt.AlignCenter, text)

def _draw_annotation_label(
    painter: QPainter,
    image_rect: QRect,
    anchor: tuple[float, float],
    text: str,
    color: QColor,
    stroke_scale: float,
) -> None:
    """Draw class/area text with a fully transparent background."""
    if not text:
        return
    font = QFont()
    font.setPixelSize(max(9, min(16, round(10 * stroke_scale ** 0.25))))
    font.setBold(True)
    painter.setFont(font)
    metrics = painter.fontMetrics()
    padding_x = 5
    padding_y = 2
    max_text_width = max(1, image_rect.width() - padding_x * 2)
    display_text = metrics.elidedText(text, Qt.ElideRight, max_text_width)
    label_width = min(
        image_rect.width(),
        metrics.horizontalAdvance(display_text) + padding_x * 2,
    )
    label_height = metrics.height() + padding_y * 2
    anchor_x, anchor_y = anchor
    label_x = max(
        image_rect.left(),
        min(int(anchor_x), image_rect.right() - label_width + 1),
    )
    label_y = int(anchor_y) - label_height
    if label_y < image_rect.top():
        label_y = int(anchor_y)
    label_y = min(label_y, image_rect.bottom() - label_height + 1)
    label_rect = QRect(label_x, label_y, label_width, label_height)
    painter.setPen(color)
    painter.drawText(
        label_rect.adjusted(padding_x, 0, -padding_x, 0),
        Qt.AlignLeft | Qt.AlignVCenter,
        display_text,
    )


def render_preview_pixmap(
    pixmap: QPixmap | None,
    annotation: ImageAnnotation,
    class_colors: dict[str, str],
    control_rules: dict[str, PreviewAdvancedFilter] | None = None,
    control_enabled: bool = False,
    rois: list[PreviewRoi] | None = None,
) -> QPixmap | None:
    """Render the full-resolution image with the same overlays as preview."""
    if pixmap is None or pixmap.isNull():
        return None
    result = QPixmap(pixmap.size())
    result.fill(Qt.transparent)
    painter = QPainter(result)
    image_rect = QRect(0, 0, pixmap.width(), pixmap.height())
    painter.drawPixmap(image_rect, pixmap)
    stroke_scale = max(
        1.0,
        min(pixmap.width() / 1000.0, pixmap.height() / 750.0),
    )
    _draw_annotation_overlays(
        painter,
        image_rect,
        annotation,
        class_colors,
        stroke_scale=stroke_scale,
        control_rules=control_rules or {},
        control_enabled=control_enabled,
        rois=rois or [],
    )
    painter.end()
    return result


def save_preview_image(
    output_path: Path | str,
    pixmap: QPixmap | None,
    annotation: ImageAnnotation,
    class_colors: dict[str, str],
    control_rules: dict[str, PreviewAdvancedFilter] | None = None,
    control_enabled: bool = False,
    rois: list[PreviewRoi] | None = None,
) -> bool:
    """Save one full-resolution annotated preview as PNG."""
    rendered = render_preview_pixmap(
        pixmap,
        annotation,
        class_colors,
        control_rules,
        control_enabled,
        rois,
    )
    if rendered is None:
        return False
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return rendered.save(str(output_path), "PNG")


def _path_is_within(path: Path, parent: Path) -> bool:
    """Return whether path resolves to parent itself or one of its children."""
    try:
        path.resolve().relative_to(parent.resolve())
    except (OSError, ValueError):
        return False
    return True


class PreviewGrid(QListWidget):
    """Icon grid that stores image path, annotation model and thumbnail."""

    edit_requested = pyqtSignal(object)  # Path

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items_by_path: dict[str, QListWidgetItem] = {}
        self.setObjectName("previewGrid")
        self.setViewMode(QListWidget.IconMode)
        self.setResizeMode(QListWidget.Adjust)
        self.setMovement(QListWidget.Static)
        self.setWrapping(True)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setUniformItemSizes(True)
        self.setSpacing(10)
        self.setIconSize(QSize(176, 132))
        self.setItemDelegate(PreviewDelegate(self, self))

    def clear(self) -> None:
        self._items_by_path.clear()
        super().clear()

    def add_preview_item(
        self,
        path: Path,
        annotation: ImageAnnotation,
        summary: PreviewSummary,
        rois: list[PreviewRoi] | None = None,
    ) -> QListWidgetItem:
        item = QListWidgetItem(self)
        key = str(path)
        item.setData(_PATH_ROLE, key)
        item.setData(_ANNOTATION_ROLE, annotation)
        item.setData(_SUMMARY_ROLE, summary)
        item.setData(_ROI_ROLE, list(rois or []))
        item.setToolTip(key)
        item.setText("")
        self._items_by_path[key] = item
        return item

    def update_thumbnail(self, path: Path, pixmap: QPixmap) -> None:
        item = self._items_by_path.get(str(path))
        if item is None:
            return
        item.setData(Qt.DecorationRole, pixmap)
        self.update(self.indexFromItem(item))

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        item = self.itemAt(event.pos())
        if item is None:
            return
        self.setCurrentItem(item)
        path = Path(item.data(_PATH_ROLE))
        menu = QMenu(self)
        edit_action = menu.addAction("编辑")
        chosen = menu.exec_(event.globalPos())
        if chosen == edit_action:
            self.edit_requested.emit(path)


class PreviewDelegate(QStyledItemDelegate):
    """Draw thumbnail plus read-only annotation overlay."""

    def __init__(self, grid: PreviewGrid, parent=None):
        super().__init__(parent)
        self._grid = grid
        self._class_colors: dict[str, str] = {}
        self._control_rules: dict[str, PreviewAdvancedFilter] = {}
        self._control_enabled = False

    def set_class_colors(self, colors: dict[str, str]) -> None:
        self._class_colors = dict(colors)

    def set_control_rules(
        self,
        rules: dict[str, PreviewAdvancedFilter],
        enabled: bool,
    ) -> None:
        self._control_rules = dict(rules)
        self._control_enabled = enabled
        self._grid.viewport().update()

    def sizeHint(self, option, index):  # noqa: N802
        icon_size = self._grid.iconSize()
        return QSize(
            icon_size.width() + _CARD_PAD * 2,
            icon_size.height() + _LABEL_BAR_H + _CARD_PAD * 2,
        )

    def paint(self, painter: QPainter, option, index):  # noqa: N802
        painter.save()
        rect = option.rect.adjusted(2, 2, -2, -2)

        if option.state & QStyle.State_Selected:
            painter.fillRect(rect, QColor(PALETTE["primary_soft"]))
        elif option.state & QStyle.State_MouseOver:
            painter.fillRect(rect, QColor(PALETTE["panel_alt"]))
        else:
            painter.fillRect(rect, QColor(PALETTE["panel"]))

        image_rect = QRect(
            rect.x() + _CARD_PAD,
            rect.y() + _CARD_PAD,
            rect.width() - _CARD_PAD * 2,
            rect.height() - _CARD_PAD * 2 - _LABEL_BAR_H,
        )
        drawn_rect = self._draw_thumbnail(painter, image_rect, index)
        annotation: ImageAnnotation | None = index.data(_ANNOTATION_ROLE)
        if annotation is not None and drawn_rect.isValid():
            _draw_annotation_overlays(
                painter,
                drawn_rect,
                annotation,
                self._class_colors,
                control_rules=self._control_rules,
                control_enabled=self._control_enabled,
                rois=index.data(_ROI_ROLE) or [],
            )
        self._draw_label(painter, rect, index)
        painter.restore()

    def _draw_thumbnail(self, painter: QPainter, image_rect: QRect, index) -> QRect:
        pixmap = index.data(Qt.DecorationRole)
        if not isinstance(pixmap, QPixmap) or pixmap.isNull():
            painter.fillRect(image_rect, QColor(PALETTE["bg_deep"]))
            painter.setPen(QColor(PALETTE["text_subtle"]))
            painter.drawText(image_rect, Qt.AlignCenter, "加载中")
            return QRect()

        scaled = pixmap.scaled(
            image_rect.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        x = image_rect.x() + (image_rect.width() - scaled.width()) // 2
        y = image_rect.y() + (image_rect.height() - scaled.height()) // 2
        target = QRect(x, y, scaled.width(), scaled.height())
        painter.drawPixmap(target, scaled)
        return target

    def _draw_label(self, painter: QPainter, rect: QRect, index) -> None:
        summary: PreviewSummary | None = index.data(_SUMMARY_ROLE)
        if summary is None:
            summary = PreviewSummary("未标注", "unlabeled", "#6c7086")

        label_rect = QRect(
            rect.x() + _CARD_PAD,
            rect.bottom() - _LABEL_BAR_H - _CARD_PAD + 1,
            rect.width() - _CARD_PAD * 2,
            _LABEL_BAR_H,
        )
        bg = QColor(summary.color)
        bg.setAlpha(210)
        painter.fillRect(label_rect, bg)
        painter.setPen(QColor(PALETTE["ink"]))
        font = QFont()
        font.setPixelSize(11)
        painter.setFont(font)
        painter.drawText(
            label_rect.adjusted(6, 0, -6, 0),
            Qt.AlignVCenter | Qt.AlignLeft,
            summary.text,
        )

class DetailPreviewCanvas(QWidget):
    """Large read-only image view used by the preview dialog."""

    edit_requested = pyqtSignal()
    rois_changed = pyqtSignal(object)

    def __init__(
        self,
        pixmap: QPixmap | None,
        annotation: ImageAnnotation,
        class_colors: dict[str, str],
        control_rules: dict[str, PreviewAdvancedFilter] | None = None,
        control_enabled: bool = False,
        rois: list[PreviewRoi] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._pixmap = pixmap
        self._annotation = annotation
        self._class_colors = dict(class_colors)
        self._control_rules = dict(control_rules or {})
        self._control_enabled = control_enabled
        self._rois = list(rois or [])
        self._roi_tool = ""
        self._roi_drag_start: tuple[float, float] | None = None
        self._roi_drag_current: tuple[float, float] | None = None
        self._roi_polygon_points: list[tuple[float, float]] = []
        self._roi_move_start: tuple[float, float] | None = None
        self._roi_move_snapshot: PreviewRoi | None = None
        self._scale = 1.0
        self.setMinimumSize(240, 180)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    def set_roi_tool(self, tool: str) -> None:
        self._roi_tool = tool if tool in {"rectangle", "ellipse", "polygon"} else ""
        self._roi_drag_start = None
        self._roi_drag_current = None
        self._roi_polygon_points = []
        self._roi_move_start = None
        self._roi_move_snapshot = None
        self.setCursor(Qt.CrossCursor if self._roi_tool else Qt.ArrowCursor)
        self.update()

    def set_rois(self, rois: list[PreviewRoi]) -> None:
        self._rois = list(rois)
        self.update()

    def clear_rois(self) -> None:
        if not self._rois:
            return
        self._rois = []
        self.rois_changed.emit(list(self._rois))
        self.update()

    def _event_norm(self, event) -> tuple[float, float]:
        width = max(1, self.sizeHint().width())
        height = max(1, self.sizeHint().height())
        return (
            max(0.0, min(1.0, event.x() / width)),
            max(0.0, min(1.0, event.y() / height)),
        )

    def _constrain_roi_end(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> tuple[float, float]:
        """Constrain the ellipse tool to a true circle in displayed pixels."""
        if self._roi_tool != "ellipse":
            return end
        width = max(1, self.sizeHint().width())
        height = max(1, self.sizeHint().height())
        dx = (end[0] - start[0]) * width
        dy = (end[1] - start[1]) * height
        side = min(abs(dx), abs(dy))
        if side <= 0:
            return start
        constrained = (
            start[0] + (side if dx >= 0 else -side) / width,
            start[1] + (side if dy >= 0 else -side) / height,
        )
        return (
            max(0.0, min(1.0, constrained[0])),
            max(0.0, min(1.0, constrained[1])),
        )

    def _near_first_polygon_point(self, point: tuple[float, float]) -> bool:
        if not self._roi_polygon_points:
            return False
        first = self._roi_polygon_points[0]
        width = max(1, self.sizeHint().width())
        height = max(1, self.sizeHint().height())
        return (
            ((point[0] - first[0]) * width) ** 2
            + ((point[1] - first[1]) * height) ** 2
            <= _ROI_CLOSE_RADIUS_PX ** 2
        )

    def _finish_roi_polygon(self) -> bool:
        """Close the active polygon once and publish the global ROI."""
        points = list(self._roi_polygon_points)
        if len(points) >= 2 and self._near_first_polygon_point(points[-1]):
            points.pop()
        if len(points) < 3:
            return False
        self._rois = [PreviewRoi("polygon", tuple(points))]
        self._roi_polygon_points = []
        self._roi_drag_current = None
        self.rois_changed.emit(list(self._rois))
        self.update()
        return True

    def _cancel_roi_polygon(self) -> None:
        self._roi_polygon_points = []
        self._roi_drag_current = None
        self.update()

    @staticmethod
    def _move_roi(
        roi: PreviewRoi,
        dx: float,
        dy: float,
    ) -> PreviewRoi:
        """Translate a ROI while keeping every point inside the image."""
        if not roi.points:
            return roi
        min_x = min(point[0] for point in roi.points)
        max_x = max(point[0] for point in roi.points)
        min_y = min(point[1] for point in roi.points)
        max_y = max(point[1] for point in roi.points)
        dx = max(-min_x, min(1.0 - max_x, dx))
        dy = max(-min_y, min(1.0 - max_y, dy))
        return PreviewRoi(
            roi.shape,
            tuple((x + dx, y + dy) for x, y in roi.points),
        )

    def sizeHint(self):  # noqa: N802
        if self._pixmap is None or self._pixmap.isNull():
            return QSize(640, 420)
        return QSize(
            max(1, int(self._pixmap.width() * self._scale)),
            max(1, int(self._pixmap.height() * self._scale)),
        )

    def set_scale(self, scale: float) -> None:
        self._scale = max(0.05, min(16.0, scale))
        self.updateGeometry()
        self.resize(self.sizeHint())
        self.update()

    def zoom_by(self, factor: float) -> None:
        self.set_scale(self._scale * factor)

    def set_content(
        self,
        pixmap: QPixmap | None,
        annotation: ImageAnnotation,
    ) -> None:
        self._pixmap = pixmap
        self._annotation = annotation
        self.updateGeometry()
        self.resize(self.sizeHint())
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(PALETTE["bg_deep"]))
        if self._pixmap is None or self._pixmap.isNull():
            painter.setPen(QColor(PALETTE["text_subtle"]))
            painter.drawText(self.rect(), Qt.AlignCenter, "图片加载失败")
            painter.end()
            return

        image_rect = QRect(0, 0, self.sizeHint().width(), self.sizeHint().height())
        scaled = self._pixmap.scaled(
            image_rect.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        image_rect = QRect(0, 0, scaled.width(), scaled.height())
        painter.drawPixmap(image_rect, scaled)
        _draw_annotation_overlays(
            painter,
            image_rect,
            self._annotation,
            self._class_colors,
            stroke_scale=max(1.0, self._scale),
            control_rules=self._control_rules,
            control_enabled=self._control_enabled,
            rois=self._rois,
        )
        self._paint_roi_preview(painter, image_rect)
        painter.end()

    def _paint_roi_preview(self, painter: QPainter, image_rect: QRect) -> None:
        preview: list[PreviewRoi] = []
        if self._roi_tool in {"rectangle", "ellipse"} and self._roi_drag_start and self._roi_drag_current:
            preview.append(PreviewRoi(
                self._roi_tool,
                (self._roi_drag_start, self._roi_drag_current),
            ))
        elif self._roi_tool == "polygon" and self._roi_polygon_points:
            points = list(self._roi_polygon_points)
            if self._roi_drag_current is not None:
                points.append(
                    self._roi_polygon_points[0]
                    if self._near_first_polygon_point(self._roi_drag_current)
                    else self._roi_drag_current
                )
            if len(points) >= 2:
                color = QColor(PALETTE["warning"])
                painter.setPen(QPen(color, 2, Qt.DashDotLine))
                painter.setBrush(Qt.NoBrush)
                painter.drawPolyline(QPolygonF([
                    _norm_point(image_rect, *point) for point in points
                ]))
                first = _norm_point(image_rect, *self._roi_polygon_points[0])
                painter.setBrush(QBrush(color))
                painter.drawEllipse(first, 5, 5)
                return
        _draw_preview_rois(painter, image_rect, preview, max(1.0, self._scale))

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        point = self._event_norm(event)
        if not self._roi_tool:
            if self._rois and _roi_contains_point(self._rois[0], point):
                self._roi_move_start = point
                self._roi_move_snapshot = self._rois[0]
                self.setCursor(Qt.ClosedHandCursor)
                event.accept()
                return
            super().mousePressEvent(event)
            return
        if self._roi_tool in {"rectangle", "ellipse"}:
            self._roi_drag_start = point
            self._roi_drag_current = point
        else:
            if len(self._roi_polygon_points) >= 3 and self._near_first_polygon_point(point):
                self._finish_roi_polygon()
                event.accept()
                return
            self._roi_polygon_points.append(point)
            self._roi_drag_current = point
        self.update()
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._roi_move_start is not None and self._roi_move_snapshot is not None:
            current = self._event_norm(event)
            self._rois = [self._move_roi(
                self._roi_move_snapshot,
                current[0] - self._roi_move_start[0],
                current[1] - self._roi_move_start[1],
            )]
            self.setCursor(Qt.ClosedHandCursor)
            self.update()
            event.accept()
            return
        if not self._roi_tool:
            point = self._event_norm(event)
            self.setCursor(
                Qt.OpenHandCursor
                if self._rois and _roi_contains_point(self._rois[0], point)
                else Qt.ArrowCursor
            )
            super().mouseMoveEvent(event)
            return
        if (
            self._roi_tool == "polygon"
            or (self._roi_drag_start is not None and event.buttons() & Qt.LeftButton)
        ):
            current = self._event_norm(event)
            self._roi_drag_current = (
                self._constrain_roi_end(self._roi_drag_start, current)
                if self._roi_drag_start is not None
                else current
            )
            self.update()
            event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self._roi_move_start is not None:
            self._roi_move_start = None
            self._roi_move_snapshot = None
            self.setCursor(Qt.OpenHandCursor)
            self.rois_changed.emit(list(self._rois))
            self.update()
            event.accept()
            return
        if (
            event.button() == Qt.LeftButton
            and self._roi_tool in {"rectangle", "ellipse"}
            and self._roi_drag_start is not None
        ):
            start = self._roi_drag_start
            end = self._constrain_roi_end(start, self._event_norm(event))
            if abs(end[0] - start[0]) > 0.002 and abs(end[1] - start[1]) > 0.002:
                # The ROI is global and singular. Drawing another shape
                # replaces the previous control region for every image.
                self._rois = [PreviewRoi(self._roi_tool, (start, end))]
                self.rois_changed.emit(list(self._rois))
            self._roi_drag_start = None
            self._roi_drag_current = None
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if self._roi_tool == "polygon" and event.button() == Qt.LeftButton:
            # Qt sends a normal press before the double-click event; remove the
            # duplicate endpoint so the saved polygon does not self-overlap.
            if len(self._roi_polygon_points) >= 2:
                previous = self._roi_polygon_points[-2]
                current = self._roi_polygon_points[-1]
                width = max(1, self.sizeHint().width())
                height = max(1, self.sizeHint().height())
                if (
                    ((current[0] - previous[0]) * width) ** 2
                    + ((current[1] - previous[1]) * height) ** 2
                    <= _ROI_CLOSE_RADIUS_PX ** 2
                ):
                    self._roi_polygon_points.pop()
            self._finish_roi_polygon()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if self._roi_tool == "polygon":
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self._finish_roi_polygon()
                event.accept()
                return
            if event.key() == Qt.Key_Escape:
                self._cancel_roi_polygon()
                event.accept()
                return
        super().keyPressEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802
        factor = _ZOOM_FACTOR if event.angleDelta().y() > 0 else 1.0 / _ZOOM_FACTOR
        self.zoom_by(factor)
        event.accept()

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        if self._roi_tool == "polygon" and self._roi_polygon_points:
            self._cancel_roi_polygon()
            event.accept()
            return
        menu = QMenu(self)
        edit_action = menu.addAction("编辑")
        chosen = menu.exec_(event.globalPos())
        if chosen == edit_action:
            self.edit_requested.emit()


class PreviewDetailDialog(QDialog):
    """Dialog for inspecting one image at a larger scale."""

    edit_requested = pyqtSignal(object)  # Path
    rois_changed = pyqtSignal(object)  # global list[PreviewRoi]

    def __init__(
        self,
        image_path: Path,
        pixmap: QPixmap | None,
        annotation: ImageAnnotation,
        class_colors: dict[str, str],
        summary: PreviewSummary,
        project: ProjectManager | None = None,
        image_paths: list[Path] | None = None,
        current_index: int = 0,
        control_rules: dict[str, PreviewAdvancedFilter] | None = None,
        control_enabled: bool = False,
        rois: list[PreviewRoi] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._project = project
        self._image_paths = list(image_paths or [image_path])
        self._current_index = max(0, min(current_index, len(self._image_paths) - 1))
        self._class_colors = dict(class_colors)
        self._control_rules = dict(control_rules or {})
        self._control_enabled = control_enabled
        self._rois = list(rois or [])
        self._current_path = image_path
        self.setWindowTitle(f"预览 - {image_path.name}")
        self.resize(1100, 760)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(6)

        self._zoom_out = QPushButton(icon("zoom_out"), "")
        self._zoom_out.setToolTip("缩小")
        set_button_role(self._zoom_out, "icon")
        self._zoom_out.clicked.connect(lambda: self._canvas.zoom_by(1.0 / _ZOOM_FACTOR))
        toolbar.addWidget(self._zoom_out)

        self._zoom_in = QPushButton(icon("zoom_in"), "")
        self._zoom_in.setToolTip("放大")
        set_button_role(self._zoom_in, "icon")
        self._zoom_in.clicked.connect(lambda: self._canvas.zoom_by(_ZOOM_FACTOR))
        toolbar.addWidget(self._zoom_in)

        self._fit = QPushButton(icon("zoom_fit"), "")
        self._fit.setToolTip("适应窗口")
        set_button_role(self._fit, "icon")
        self._fit.clicked.connect(self._fit_to_window)
        toolbar.addWidget(self._fit)

        self._roi_tool_combo = QComboBox()
        self._roi_tool_combo.addItem("ROI：移动", "")
        self._roi_tool_combo.addItem("ROI：矩形", "rectangle")
        self._roi_tool_combo.addItem("ROI：圆形", "ellipse")
        self._roi_tool_combo.addItem("ROI：多边形", "polygon")
        self._roi_tool_combo.setToolTip(
            "移动模式下按住 ROI 内部拖动；ROI 仅用于预览分析，不写入标注或训练数据；多边形双击完成"
        )
        self._roi_tool_combo.currentIndexChanged.connect(self._on_roi_tool_changed)
        toolbar.addWidget(self._roi_tool_combo)

        self._clear_roi_btn = QPushButton("清除当前 ROI")
        set_button_role(self._clear_roi_btn, "secondary")
        toolbar.addWidget(self._clear_roi_btn)

        self._export_btn = QPushButton(icon("export"), "导出当前图")
        self._export_btn.setToolTip("导出包含标注、面积、卡控结果和 ROI 的当前预览图")
        set_button_role(self._export_btn, "secondary")
        self._export_btn.clicked.connect(self._export_current_preview)
        toolbar.addWidget(self._export_btn)

        self._title = QLabel("")
        self._title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._title.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        toolbar.addWidget(self._title)
        layout.addLayout(toolbar)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(False)
        self._scroll.setAlignment(Qt.AlignCenter)
        self._scroll.setStyleSheet(
            "QScrollArea {"
            f" background-color: {PALETTE['bg_deep']};"
            f" border: 1px solid {PALETTE['line']};"
            "}"
        )
        self._canvas = DetailPreviewCanvas(
            pixmap,
            annotation,
            class_colors,
            self._control_rules,
            self._control_enabled,
            self._current_rois(),
        )
        self._canvas.edit_requested.connect(self._request_edit_current)
        self._canvas.rois_changed.connect(self._on_canvas_rois_changed)
        self._clear_roi_btn.clicked.connect(self._canvas.clear_rois)
        self._scroll.setWidget(self._canvas)
        layout.addWidget(self._scroll, 1)
        self._set_title(image_path, summary)
        self._fit_to_window()

    def _current_rois(self) -> list[PreviewRoi]:
        return list(self._rois)

    def _on_roi_tool_changed(self, _index: int) -> None:
        self._canvas.set_roi_tool(str(self._roi_tool_combo.currentData() or ""))

    def _on_canvas_rois_changed(self, rois: list[PreviewRoi]) -> None:
        self._rois = list(rois)
        self._control_enabled = bool(self._control_rules or self._rois)
        self._canvas._control_enabled = self._control_enabled
        self.rois_changed.emit(list(rois))

    def _fit_to_window(self) -> None:
        pixmap = self._canvas._pixmap
        if pixmap is None or pixmap.isNull():
            return
        viewport = self._scroll.viewport().size()
        scale = min(
            max(1, viewport.width() - 16) / pixmap.width(),
            max(1, viewport.height() - 16) / pixmap.height(),
        )
        self._canvas.set_scale(scale)

    def _set_title(self, path: Path, summary: PreviewSummary) -> None:
        total = len(self._image_paths)
        prefix = f"{self._current_index + 1}/{total} · " if total > 1 else ""
        self._title.setText(f"{prefix}{path.name} | {summary.text}")
        self.setWindowTitle(f"预览 - {path.name}")

    def _load_annotation(self, path: Path) -> ImageAnnotation:
        if self._project is None:
            return ImageAnnotation(str(path), get_image_size(path))
        annotation = load_annotation(self._project.label_path_for(path))
        if annotation is None:
            return ImageAnnotation(str(path), get_image_size(path))
        return annotation

    def _show_index(self, index: int) -> None:
        if not self._image_paths:
            return
        self._current_index = index % len(self._image_paths)
        path = self._image_paths[self._current_index]
        annotation = self._load_annotation(path)
        summary = _summary_for_annotation(annotation, self._class_colors)
        self._current_path = path
        self._canvas.set_content(load_pixmap(path), annotation)
        self._canvas.set_rois(self._current_rois())
        self._set_title(path, summary)
        self._fit_to_window()

    def _export_current_preview(self) -> None:
        default_dir = self._current_path.parent
        if self._project is not None:
            default_dir = self._project.project_dir / "preview_exports"
            if _path_is_within(default_dir, self._project.image_root()):
                default_dir = (
                    self._project.project_dir.parent
                    / f"{self._project.project_dir.name}_preview_exports"
                )
        default_path = default_dir / f"{self._current_path.stem}_preview.png"
        output_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "导出当前预览图",
            str(default_path),
            "PNG 图像 (*.png)",
        )
        if not output_path:
            return
        target = Path(output_path)
        if target.suffix.casefold() != ".png":
            target = target.with_suffix(".png")
        if (
            self._project is not None
            and _path_is_within(target, self._project.image_root())
        ):
            QMessageBox.warning(
                self,
                "导出位置无效",
                "不能保存到项目图片目录或其子目录，否则预览图会被当成训练图片。",
            )
            return
        try:
            saved = save_preview_image(
                target,
                self._canvas._pixmap,
                self._canvas._annotation,
                self._class_colors,
                self._control_rules,
                self._control_enabled,
                self._current_rois(),
            )
        except OSError as exc:
            QMessageBox.warning(self, "导出失败", str(exc))
            return
        if not saved:
            QMessageBox.warning(self, "导出失败", "当前预览图无法生成或保存。")
            return
        QMessageBox.information(self, "导出完成", f"预览图已保存：\n{target}")

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_A and not event.modifiers():
            self._show_index(self._current_index - 1)
            event.accept()
            return
        if event.key() == Qt.Key_D and not event.modifiers():
            self._show_index(self._current_index + 1)
            event.accept()
            return
        super().keyPressEvent(event)

    def _request_edit_current(self) -> None:
        path = self._current_path
        self.accept()
        self.edit_requested.emit(path)


class PreviewAdvancedFilterDialog(QDialog):
    """Edit per-class detection-box control rules and commit on Apply."""

    _RANGE_ROWS = (
        ("width", "宽度", " px", 0.0, 1_000_000.0, 0),
        ("height", "高度", " px", 0.0, 1_000_000.0, 0),
        ("area", "面积", " px²", 0.0, 1_000_000_000_000.0, 0),
        ("confidence", "置信度", "", 0.0, 1.0, 2),
        ("center_x", "中心点 X", " px", 0.0, 1_000_000.0, 0),
        ("center_y", "中心点 Y", " px", 0.0, 1_000_000.0, 0),
    )

    def __init__(
        self,
        current: dict[str, PreviewAdvancedFilter],
        available_classes: list[str],
        annotation_filters_enabled: bool = True,
        pixel_limits: dict[str, float] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("标注卡控")
        self.setMinimumWidth(480)
        self._rules = dict(current)
        self._available_classes = list(available_classes)
        self._range_controls: dict[
            str, tuple[QCheckBox, QDoubleSpinBox, QDoubleSpinBox]
        ] = {}

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        class_row = QHBoxLayout()
        class_row.addWidget(QLabel("缺陷类别"))
        self._class_combo = QComboBox()
        self._class_combo.addItems(self._available_classes)
        self._class_combo.currentTextChanged.connect(self._on_class_changed)
        class_row.addWidget(self._class_combo, 1)
        layout.addLayout(class_row)

        annotation_group = QGroupBox("检测框卡控参数")
        annotation_layout = QFormLayout(annotation_group)
        annotation_layout.setHorizontalSpacing(16)
        annotation_layout.setVerticalSpacing(8)
        for key, label, suffix, lower, fallback_upper, decimals in self._RANGE_ROWS:
            current_range = NumericRange()
            current_upper = max(
                current_range.minimum or lower,
                current_range.maximum or lower,
            )
            upper = max(
                fallback_upper if pixel_limits is None else pixel_limits.get(key, fallback_upper),
                current_upper,
                1.0,
            )
            enabled = QCheckBox(label)
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)
            minimum = self._make_spin(lower, upper, decimals, suffix)
            maximum = self._make_spin(lower, upper, decimals, suffix)
            minimum.setValue(lower)
            maximum.setValue(upper)
            minimum.valueChanged.connect(maximum.setMinimum)
            maximum.valueChanged.connect(minimum.setMaximum)
            row_layout.addWidget(QLabel("最小"))
            row_layout.addWidget(minimum)
            row_layout.addWidget(QLabel("～ 最大"))
            row_layout.addWidget(maximum)
            annotation_layout.addRow(enabled, row)
            self._range_controls[key] = (enabled, minimum, maximum)
            # Keep range fields editable even when the condition is inactive.
            # Editing either endpoint is itself an intent to enable the row;
            # unchecking only controls whether the range participates.
            minimum.valueChanged.connect(
                lambda _value, checkbox=enabled: checkbox.setChecked(True)
            )
            maximum.valueChanged.connect(
                lambda _value, checkbox=enabled: checkbox.setChecked(True)
            )

        annotation_group.setEnabled(annotation_filters_enabled)
        if not annotation_filters_enabled:
            annotation_group.setToolTip("分类任务没有目标框，不能按标注属性筛选")
        layout.addWidget(annotation_group)

        hint = QLabel(
            "宽度、高度和中心位置使用原图像素坐标；面积使用原图像素面积。\n"
            "每个检测框按所属缺陷类别分别判断：满足全部启用条件为 OK，否则为 NG。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {PALETTE['text_subtle']};")
        layout.addWidget(hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Reset | QDialogButtonBox.Apply | QDialogButtonBox.Cancel
        )
        buttons.button(QDialogButtonBox.Reset).setText("重置全部条件")
        buttons.button(QDialogButtonBox.Apply).setText("应用")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.button(QDialogButtonBox.Reset).clicked.connect(self._reset_all)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._loading_class = False
        self._active_class = ""
        if self._class_combo.count():
            self._load_class(self._class_combo.currentText())

    @staticmethod
    def _make_spin(
        lower: float,
        upper: float,
        decimals: int,
        suffix: str,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(lower, upper)
        spin.setDecimals(decimals)
        spin.setSingleStep(0.05 if upper == 1.0 else 1.0)
        if suffix:
            spin.setSuffix(suffix)
        spin.setMinimumWidth(110)
        return spin

    def _restore_range(self, key: str, value: NumericRange) -> None:
        enabled, minimum, maximum = self._range_controls[key]
        lower = minimum.minimum()
        upper = maximum.maximum()
        minimum.setValue(
            lower if value.minimum is None else value.minimum
        )
        maximum.setValue(
            upper if value.maximum is None else value.maximum
        )
        enabled.setChecked(value.is_active())

    def _reset_all(self) -> None:
        self._rules.clear()
        for key, (enabled, minimum, maximum) in self._range_controls.items():
            minimum.setValue(0.0)
            maximum.setValue(maximum.maximum())
            # Value edits normally auto-enable a condition. Reset values first,
            # then explicitly leave every condition inactive.
            enabled.setChecked(False)

    def _controls_value(self) -> PreviewAdvancedFilter:
        values = {}
        for key, (enabled, minimum, maximum) in self._range_controls.items():
            if not enabled.isChecked():
                values[key] = NumericRange()
                continue
            values[key] = NumericRange(
                minimum=minimum.value(),
                maximum=maximum.value(),
            )
        return PreviewAdvancedFilter(**values)

    def _save_active_class(self) -> None:
        if not self._active_class or self._loading_class:
            return
        value = self._controls_value()
        if value.has_annotation_constraints():
            self._rules[self._active_class] = value
        else:
            self._rules.pop(self._active_class, None)

    def _load_class(self, class_name: str) -> None:
        self._loading_class = True
        value = self._rules.get(class_name, PreviewAdvancedFilter())
        for key in self._range_controls:
            self._restore_range(key, getattr(value, key))
        self._active_class = class_name
        self._loading_class = False

    def _on_class_changed(self, class_name: str) -> None:
        self._save_active_class()
        self._load_class(class_name)

    def value(self) -> dict[str, PreviewAdvancedFilter]:
        self._save_active_class()
        return dict(self._rules)


class PreviewPanel(QWidget):
    """Preview all project images with saved labels overlaid."""

    status_changed = pyqtSignal(str)
    edit_requested = pyqtSignal(object)  # Path

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project: ProjectManager | None = None
        self._class_colors: dict[str, str] = {}
        self._status_filter: str | None = None
        self._class_filter: str | None = None
        self._data_folder_filter: str | None = None
        self._tag_filter = TagFilter()
        self._control_rules: dict[str, PreviewAdvancedFilter] = {}
        self._control_enabled = False
        self._available_tags: list[str] = []
        self._pixel_filter_limits: dict[str, float] = {}
        # One global, preview-only ROI applies to every image. It deliberately
        # stays outside Annotation and ProjectManager so it cannot enter labels,
        # exports, or training.
        self._preview_rois: list[PreviewRoi] = []
        self._loader: ThumbnailLoader | None = None
        self._init_ui()
        self._create_loader()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self._toolbar = QToolBar()
        self._toolbar.setMovable(False)
        self._btn_refresh = QPushButton(icon("refresh"), "刷新")
        self._btn_refresh.setToolTip("重新读取全部图片和标注")
        set_button_role(self._btn_refresh, "secondary")
        self._btn_refresh.clicked.connect(self.refresh)
        self._toolbar.addWidget(self._btn_refresh)

        self._toolbar.addSeparator()
        self._status_combo = QComboBox()
        self._status_combo.addItems(["全部", "已确认", "待确认", "未标注"])
        self._status_combo.setMinimumWidth(86)
        self._status_combo.currentTextChanged.connect(self._on_status_filter_changed)
        self._toolbar.addWidget(QLabel(" 筛选 "))
        self._toolbar.addWidget(self._status_combo)

        self._class_combo = QComboBox()
        self._class_combo.addItem("所有类别")
        self._class_combo.setMinimumWidth(96)
        self._class_combo.currentTextChanged.connect(self._on_class_filter_changed)
        self._toolbar.addWidget(QLabel(" 类别 "))
        self._toolbar.addWidget(self._class_combo)

        self._data_folder_combo = QComboBox()
        self._data_folder_combo.addItem(_ALL_DATA_FOLDERS_TEXT, "")
        self._data_folder_combo.setMinimumWidth(112)
        self._data_folder_combo.currentIndexChanged.connect(
            self._on_data_folder_filter_changed
        )
        self._toolbar.addWidget(QLabel(" 数据版本 "))
        self._toolbar.addWidget(self._data_folder_combo)

        self._tag_filter_bar = TagFilterBar()
        self._tag_filter_bar.filter_changed.connect(self._on_tag_filter_changed)
        self._toolbar.addWidget(self._tag_filter_bar)

        self._control_btn = QPushButton("标注卡控（未开启）")
        self._control_btn.setToolTip("按缺陷类别设置检测框宽高、面积、置信度和中心点卡控")
        set_button_role(self._control_btn, "secondary")
        self._control_btn.clicked.connect(self._show_annotation_control)
        self._toolbar.addWidget(self._control_btn)

        self._toolbar.addSeparator()
        self._toolbar.addWidget(QLabel("缩略图 "))
        self._size_slider = QSlider(Qt.Horizontal)
        self._size_slider.setRange(120, 280)
        self._size_slider.setSingleStep(20)
        self._size_slider.setPageStep(40)
        self._size_slider.setValue(176)
        self._size_slider.setFixedWidth(140)
        self._size_slider.valueChanged.connect(self._on_size_changed)
        self._toolbar.addWidget(self._size_slider)

        self._summary = QLabel("未打开项目")
        self._summary.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._summary.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._toolbar.addWidget(self._summary)
        layout.addWidget(self._toolbar)

        self._grid = PreviewGrid()
        self._grid.setStyleSheet(
            "QListWidget#previewGrid {"
            f" background-color: {PALETTE['bg']};"
            f" border: 1px solid {PALETTE['line']};"
            "}"
        )
        self._grid.itemClicked.connect(self._open_item_preview)
        self._grid.edit_requested.connect(self.edit_requested.emit)
        layout.addWidget(self._grid, 1)

    def _create_loader(self) -> None:
        self._loader = ThumbnailLoader(parent=self)
        self._loader.loaded.connect(self._on_thumbnail_loaded)

    def set_project(self, project: ProjectManager) -> None:
        self._project = project
        self._preview_rois.clear()
        self._tag_filter = TagFilter()
        self._control_rules = {}
        self._control_enabled = False
        self._update_control_button()
        self._class_colors = {
            cls: project.config.get_class_color(cls)
            for cls in project.config.classes
        }
        self._refresh_data_folder_filter()
        self._refresh_class_filter()
        self.set_available_tags(project.config.tags)
        delegate = self._grid.itemDelegate()
        if isinstance(delegate, PreviewDelegate):
            delegate.set_class_colors(self._class_colors)
            delegate.set_control_rules(self._control_rules, self._control_enabled)
        self.refresh()

    def set_available_tags(self, tags: list[str]) -> None:
        self._available_tags = sorted(set(tags))
        self._tag_filter_bar.set_available_tags(self._available_tags)
        self._tag_filter_bar.set_filter(self._tag_filter)
        self._tag_filter = self._tag_filter_bar.current_filter()

    def refresh(self) -> None:
        if self._project is None:
            self._grid.clear()
            self._summary.setText("未打开项目")
            return
        self._refresh_data_folder_filter()
        self._reset_loader()
        self._grid.clear()
        data_folder = self._data_folder_filter if self._data_folder_filter is not None else ""
        images = self._project.list_images(data_folder)
        self._refresh_class_filter(images)
        counts = {"confirmed": 0, "pending": 0, "unlabeled": 0}
        thumb_size = self._grid.iconSize()
        max_width = 0
        max_height = 0
        max_area = 0

        for path in images:
            annotation = self._load_preview_annotation(path)
            image_width, image_height = annotation.image_size
            max_width = max(max_width, image_width)
            max_height = max(max_height, image_height)
            max_area = max(max_area, image_width * image_height)
            summary = _summary_for_annotation(annotation, self._class_colors)
            counts[summary.status] += 1
            if not self._passes_filters(annotation, summary):
                continue
            self._grid.add_preview_item(
                path,
                annotation,
                summary,
                self._preview_rois,
            )
            if self._loader is not None:
                self._loader.enqueue(path, thumb_size)

        self._pixel_filter_limits = {
            "width": float(max_width),
            "height": float(max_height),
            "area": float(max_area),
            "center_x": float(max_width),
            "center_y": float(max_height),
        }

        visible = self._grid.count()
        text = (
            f"显示 {visible} / 全部 {len(images)} 张 | "
            f"已确认 {counts['confirmed']} | "
            f"待确认 {counts['pending']} | "
            f"未标注 {counts['unlabeled']}"
        )
        self._summary.setText(text)
        self.status_changed.emit(text)

    def cleanup(self) -> None:
        self._stop_loader()

    def _load_preview_annotation(self, path: Path) -> ImageAnnotation:
        if self._project is None:
            return ImageAnnotation(str(path), get_image_size(path))
        annotation = load_annotation(self._project.label_path_for(path))
        if annotation is None:
            return ImageAnnotation(str(path), get_image_size(path))
        return annotation

    def _refresh_class_filter(self, images: list[Path] | None = None) -> None:
        if self._project is None:
            return
        prev = self._class_combo.currentText()
        self._class_combo.blockSignals(True)
        self._class_combo.clear()
        self._class_combo.addItem("所有类别")
        for cls in merged_project_annotation_classes(self._project, images):
            self._class_combo.addItem(cls)
        idx = self._class_combo.findText(prev)
        self._class_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._class_combo.blockSignals(False)
        current = self._class_combo.currentText()
        self._class_filter = None if current == "所有类别" else current

    def _refresh_data_folder_filter(self) -> None:
        if self._project is None:
            return
        prev = self._data_folder_filter or ""
        folders = self._project.list_data_folders()
        self._data_folder_combo.blockSignals(True)
        self._data_folder_combo.clear()
        self._data_folder_combo.addItem(_ALL_DATA_FOLDERS_TEXT, "")
        for folder in folders:
            self._data_folder_combo.addItem(folder, folder)
        idx = self._data_folder_combo.findData(prev)
        self._data_folder_combo.setCurrentIndex(idx if idx >= 0 else 0)
        current_data = self._data_folder_combo.currentData()
        self._data_folder_filter = str(current_data) if current_data else None
        self._data_folder_combo.blockSignals(False)

    def _on_status_filter_changed(self, text: str) -> None:
        mapping = {
            "全部": None,
            "已确认": "confirmed",
            "待确认": "pending",
            "未标注": "unlabeled",
        }
        self._status_filter = mapping.get(text)
        self.refresh()

    def _on_class_filter_changed(self, text: str) -> None:
        self._class_filter = None if text == "所有类别" else text
        self.refresh()

    def _on_data_folder_filter_changed(self, _index: int = -1) -> None:
        current_data = self._data_folder_combo.currentData()
        self._data_folder_filter = str(current_data) if current_data else None
        self.refresh()

    def _on_tag_filter_changed(self, tag_filter) -> None:
        self._tag_filter = tag_filter if tag_filter is not None else TagFilter()
        self.refresh()

    def _show_annotation_control(self) -> None:
        task_type = self._project.config.task_type if self._project is not None else "detect"
        dialog = PreviewAdvancedFilterDialog(
            self._control_rules,
            [
                self._class_combo.itemText(index)
                for index in range(1, self._class_combo.count())
            ],
            annotation_filters_enabled=task_type != "classify",
            pixel_limits=self._pixel_filter_limits,
            parent=self,
        )
        if dialog.exec_() != QDialog.Accepted:
            return
        self._control_rules = dialog.value()
        self._control_enabled = bool(self._control_rules or self._preview_rois)
        self._update_control_button()
        delegate = self._grid.itemDelegate()
        if isinstance(delegate, PreviewDelegate):
            delegate.set_control_rules(self._control_rules, self._control_enabled)
        self.refresh()

    def _update_control_button(self) -> None:
        if not self._control_enabled:
            text = "标注卡控（未开启）"
        else:
            parts = []
            if self._preview_rois:
                parts.append("全局 ROI")
            if self._control_rules:
                parts.append(f"{len(self._control_rules)} 类参数")
            text = f"标注卡控（已开启 · {' + '.join(parts)}）"
        self._control_btn.setText(text)

    def _passes_filters(
        self,
        annotation: ImageAnnotation,
        summary: PreviewSummary,
    ) -> bool:
        if self._status_filter is not None and summary.status != self._status_filter:
            return False
        if not self._tag_filter.is_empty() and not self._tag_filter.matches(annotation.tags):
            return False
        if not _image_matches_annotation_filters(
            annotation,
            self._class_filter,
        ):
            return False
        return True

    def _on_size_changed(self, value: int) -> None:
        self._grid.setIconSize(QSize(value, int(value * 0.75)))
        self.refresh()

    def _on_thumbnail_loaded(self, path, pixmap) -> None:
        self._grid.update_thumbnail(Path(path), pixmap)

    def _visible_preview_paths(self) -> list[Path]:
        return [
            Path(self._grid.item(i).data(_PATH_ROLE))
            for i in range(self._grid.count())
        ]

    def export_all_previews(self) -> Path | None:
        """Export every project image with the active preview overlays."""
        if self._project is None:
            QMessageBox.information(self, "无法导出", "请先打开项目。")
            return None
        image_paths = self._project.list_images()
        if not image_paths:
            QMessageBox.information(self, "无法导出", "当前项目没有图片。")
            return None
        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "导出全部预览图",
            str(self._project.project_dir),
        )
        if not selected_dir:
            return None

        output_root = Path(selected_dir)
        image_root = self._project.image_root()
        if _path_is_within(output_root, image_root):
            QMessageBox.warning(
                self,
                "导出目录无效",
                "不能导出到项目图片目录或其子目录，否则预览图会被当成训练图片。",
            )
            return None
        progress = QProgressDialog(
            "正在导出全部预览图…",
            "取消",
            0,
            len(image_paths),
            self,
        )
        progress.setWindowTitle("导出预览图")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        exported = 0
        failed: list[Path] = []

        for index, image_path in enumerate(image_paths, start=1):
            progress.setLabelText(
                f"正在导出 {index}/{len(image_paths)}：{image_path.name}"
            )
            progress.setValue(index - 1)
            QApplication.processEvents()
            if progress.wasCanceled():
                break
            try:
                relative_path = image_path.resolve().relative_to(
                    image_root.resolve()
                )
            except (OSError, ValueError):
                relative_path = Path(image_path.name)
            output_path = (
                output_root
                / relative_path.parent
                / f"{relative_path.stem}_preview.png"
            )
            try:
                annotation = self._load_preview_annotation(image_path)
                saved = save_preview_image(
                    output_path,
                    load_pixmap(image_path),
                    annotation,
                    self._class_colors,
                    self._control_rules,
                    self._control_enabled,
                    self._preview_rois,
                )
            except OSError:
                saved = False
            if not saved:
                failed.append(image_path)
            else:
                exported += 1

        canceled = progress.wasCanceled()
        if not canceled:
            progress.setValue(len(image_paths))
        message = f"已导出 {exported} 张预览图。\n{output_root}"
        if canceled:
            message = f"导出已取消，{message}"
        if failed:
            message += f"\n失败 {len(failed)} 张。"
            QMessageBox.warning(self, "导出完成（存在失败）", message)
        else:
            QMessageBox.information(self, "导出完成", message)
        self.status_changed.emit(
            f"预览图导出完成: {exported}/{len(image_paths)} | {output_root}"
        )
        return output_root

    def _open_item_preview(self, item: QListWidgetItem) -> None:
        if self._project is None:
            return
        path = Path(item.data(_PATH_ROLE))
        annotation = self._load_preview_annotation(path)
        summary = _summary_for_annotation(annotation, self._class_colors)
        pixmap = load_pixmap(path)
        paths = self._visible_preview_paths()
        try:
            current_index = paths.index(path)
        except ValueError:
            current_index = 0
        dialog = PreviewDetailDialog(
            path,
            pixmap,
            annotation,
            self._class_colors,
            summary,
            project=self._project,
            image_paths=paths,
            current_index=current_index,
            control_rules=self._control_rules,
            control_enabled=self._control_enabled,
            rois=self._preview_rois,
            parent=self,
        )
        dialog.edit_requested.connect(self.edit_requested.emit)
        dialog.rois_changed.connect(self._on_preview_rois_changed)
        dialog.exec_()

    def _on_preview_rois_changed(self, rois: list[PreviewRoi]) -> None:
        self._preview_rois = list(rois)
        self._control_enabled = bool(self._control_rules or self._preview_rois)
        self._update_control_button()
        delegate = self._grid.itemDelegate()
        if isinstance(delegate, PreviewDelegate):
            delegate.set_control_rules(self._control_rules, self._control_enabled)
        for index in range(self._grid.count()):
            item = self._grid.item(index)
            item.setData(_ROI_ROLE, list(self._preview_rois))
        self._grid.viewport().update()

    def _reset_loader(self) -> None:
        self._stop_loader()
        self._create_loader()

    def _stop_loader(self) -> None:
        if self._loader is None:
            return
        self._loader.stop()
        if self._loader.isRunning():
            self._loader.wait(1500)
        self._loader.deleteLater()
        self._loader = None
