"""Read-only project preview grid with annotation overlays."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PyQt5.QtCore import Qt, QPointF, QRect, QSize, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
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

from src.core.annotation import ImageAnnotation
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
_LABEL_BAR_H = 22
_CARD_PAD = 8
_ZOOM_FACTOR = 1.15
_ALL_DATA_FOLDERS_TEXT = "所有版本"


@dataclass(frozen=True)
class PreviewSummary:
    text: str
    status: str
    color: str


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
) -> None:
    """Draw saved detect/pose annotations over an image rect."""
    painter.setRenderHint(QPainter.Antialiasing)
    line_width = max(2, int(2 * stroke_scale))
    point_radius = max(3, int(4 * stroke_scale))
    for ann in annotation.annotations:
        color = QColor(class_colors.get(ann.class_name, PALETTE["primary"]))
        if ann.polygon:
            polygon = QPolygonF([
                _norm_point(image_rect, x, y)
                for x, y in ann.polygon
            ])
            fill = QColor(color)
            fill.setAlpha(46)
            pen = QPen(color, line_width)
            if not ann.confirmed:
                pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(fill)
            painter.drawPolygon(polygon)
            painter.setBrush(Qt.NoBrush)

        if ann.bbox and not ann.polygon:
            cx, cy, w, h = ann.bbox
            x1, y1 = _norm_xy(image_rect, cx - w / 2, cy - h / 2)
            x2, y2 = _norm_xy(image_rect, cx + w / 2, cy + h / 2)
            pen = QPen(color, line_width)
            if not ann.confirmed:
                pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1)))

        for kp in ann.keypoints:
            x, y = _norm_xy(image_rect, kp.x, kp.y)
            painter.setPen(QPen(color, line_width))
            painter.setBrush(color if kp.visible == 2 else Qt.NoBrush)
            painter.drawEllipse(
                int(x - point_radius),
                int(y - point_radius),
                point_radius * 2,
                point_radius * 2,
            )


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
    ) -> QListWidgetItem:
        item = QListWidgetItem(self)
        key = str(path)
        item.setData(_PATH_ROLE, key)
        item.setData(_ANNOTATION_ROLE, annotation)
        item.setData(_SUMMARY_ROLE, summary)
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

    def set_class_colors(self, colors: dict[str, str]) -> None:
        self._class_colors = dict(colors)

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

    def __init__(
        self,
        pixmap: QPixmap | None,
        annotation: ImageAnnotation,
        class_colors: dict[str, str],
        parent=None,
    ):
        super().__init__(parent)
        self._pixmap = pixmap
        self._annotation = annotation
        self._class_colors = dict(class_colors)
        self._scale = 1.0
        self.setMinimumSize(240, 180)

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
        )
        painter.end()

    def wheelEvent(self, event) -> None:  # noqa: N802
        factor = _ZOOM_FACTOR if event.angleDelta().y() > 0 else 1.0 / _ZOOM_FACTOR
        self.zoom_by(factor)
        event.accept()

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        menu = QMenu(self)
        edit_action = menu.addAction("编辑")
        chosen = menu.exec_(event.globalPos())
        if chosen == edit_action:
            self.edit_requested.emit()


class PreviewDetailDialog(QDialog):
    """Dialog for inspecting one image at a larger scale."""

    edit_requested = pyqtSignal(object)  # Path

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
        parent=None,
    ):
        super().__init__(parent)
        self._project = project
        self._image_paths = list(image_paths or [image_path])
        self._current_index = max(0, min(current_index, len(self._image_paths) - 1))
        self._class_colors = dict(class_colors)
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
        self._canvas = DetailPreviewCanvas(pixmap, annotation, class_colors)
        self._canvas.edit_requested.connect(self._request_edit_current)
        self._scroll.setWidget(self._canvas)
        layout.addWidget(self._scroll, 1)
        self._set_title(image_path, summary)
        self._fit_to_window()

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
        self._set_title(path, summary)
        self._fit_to_window()

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
        self._tag_filter: TagFilter = TagFilter()
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
        self.refresh()

    def set_available_tags(self, tags: list[str]) -> None:
        self._tag_filter_bar.set_available_tags(tags)

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

        for path in images:
            annotation = self._load_preview_annotation(path)
            summary = _summary_for_annotation(annotation, self._class_colors)
            counts[summary.status] += 1
            if not self._passes_filters(annotation, summary):
                continue
            self._grid.add_preview_item(path, annotation, summary)
            if self._loader is not None:
                self._loader.enqueue(path, thumb_size)

        visible = self._grid.count()
        text = (
            f"全部 {len(images)} 张 | "
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

    def _passes_filters(
        self,
        annotation: ImageAnnotation,
        summary: PreviewSummary,
    ) -> bool:
        if self._status_filter is not None and summary.status != self._status_filter:
            return False
        if self._class_filter is not None:
            classes = set(annotation.image_tags)
            classes.update(
                ann.class_name
                for ann in annotation.annotations
                if ann.class_name
            )
            if self._class_filter not in classes:
                return False
        if not self._tag_filter.is_empty() and not self._tag_filter.matches(annotation.tags):
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
            parent=self,
        )
        dialog.edit_requested.connect(self.edit_requested.emit)
        dialog.exec_()

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
