"""Detect/Pose view extracted from LabelPanel for the shell/view refactor.

Behavior matches the pre-refactor LabelPanel for detect/pose projects.
The shell (LabelPanel) supplies the shared `image_cache` and `_undo_stacks`
dictionary so view switches preserve cache reuse but per-view undo state.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QPushButton,
    QToolBar,
    QLabel,
    QMessageBox,
    QInputDialog,
    QFileDialog,
    QMenu,
    QTreeWidget,
    QTreeWidgetItem,
)

from src.core.annotation import Annotation, ImageAnnotation, Keypoint
from src.core.annotation_classes import merged_project_annotation_classes
from src.core.label_io import save_annotation, load_annotation
from src.core.project import ProjectManager
from src.core.resources import TREE_CLOSED_SVG, TREE_OPEN_SVG, stylesheet_url
from src.ui.canvas import AnnotationCanvas
from src.ui.file_list import FileListWidget
from src.ui.properties import AnnotationPanel
from src.ui.class_picker import ClassPickerPopup, KeypointLabelPicker
from src.ui.icons import icon
from src.ui.theme import PALETTE, set_button_role
from src.ui.views.base import TaskView
from src.utils.image import get_image_size, ImageCache
from src.utils.undo import UndoStack

logger = logging.getLogger(__name__)


class DetectPoseView(TaskView):
    """Detect/Pose UI: drawing tools + file list + canvas + annotation panel."""

    _UNDO_MAX_IMAGES = 20

    # Shell-level signals exposed for backwards compat with MainWindow wiring
    batch_confirm_visible_requested = pyqtSignal()
    batch_revert_visible_requested = pyqtSignal()

    def __init__(
        self,
        image_cache: ImageCache,
        undo_stacks: "OrderedDict[str, UndoStack]",
        parent=None,
    ):
        super().__init__(parent)
        self._project: ProjectManager | None = None
        self._current_image_path: Path | None = None
        self._current_annotation: ImageAnnotation | None = None
        self._image_cache = image_cache
        self._undo_stacks = undo_stacks
        self._last_class: str | None = None
        self._clipboard: list[dict] | None = None
        self._stats_cache: dict = {}
        self._prev_annotations_snapshot: list[tuple] | None = None
        self._refreshing_data_tree = False
        self._last_data_tree_item: QTreeWidgetItem | None = None
        self._obb_tool_active = False
        self._task_type = "detect"
        self._switching_image = False
        self._queued_image_path: Path | None = None

        self._init_ui()
        self._connect_signals()



    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # View-local toolbar: drawing tools + per-image / per-visible confirm actions
        self._toolbar = QToolBar()
        self._toolbar.setMovable(False)

        self._btn_select = QPushButton(icon("cursor"), "移动")
        self._btn_select.setCheckable(True)
        self._btn_select.setChecked(True)
        self._btn_select.setToolTip("选择/移动工具 (V)")
        set_button_role(self._btn_select, "secondary")
        self._btn_bbox = QPushButton(icon("bbox"), "矩形框")
        self._btn_bbox.setCheckable(True)
        self._btn_bbox.setToolTip("绘制矩形框 (W)")
        set_button_role(self._btn_bbox, "secondary")
        self._btn_polygon = QPushButton(icon("polygon"), "多边形")
        self._btn_polygon.setCheckable(True)
        self._btn_polygon.setToolTip("绘制多边形掩膜 (P)")
        set_button_role(self._btn_polygon, "secondary")
        self._btn_obb = QPushButton(icon("obb"), "旋转框")
        self._btn_obb.setCheckable(True)
        self._btn_obb.setToolTip("拖动绘制旋转框，完成后可旋转 (O)")
        set_button_role(self._btn_obb, "secondary")
        self._btn_keypoint = QPushButton(icon("keypoint"), "关键点")
        self._btn_keypoint.setCheckable(True)
        self._btn_keypoint.setToolTip("绘制关键点 (K)")
        set_button_role(self._btn_keypoint, "secondary")

        for btn in [
            self._btn_select,
            self._btn_bbox,
            self._btn_polygon,
            self._btn_obb,
            self._btn_keypoint,
        ]:
            btn.setMinimumWidth(80)
            self._toolbar.addWidget(btn)

        self._toolbar.addSeparator()

        self._btn_confirm_visible = QPushButton(icon("confirm_visible"), "确认可见标注")
        self._btn_confirm_visible.setToolTip("确认当前可见图片的所有未确认标注 (Ctrl+Space)")
        set_button_role(self._btn_confirm_visible, "primary")
        self._toolbar.addWidget(self._btn_confirm_visible)

        self._btn_revert_visible = QPushButton(icon("revert_visible"), "撤销可见预标注")
        self._btn_revert_visible.setToolTip("删除当前可见图片的所有未确认标注")
        set_button_role(self._btn_revert_visible, "danger")
        self._toolbar.addWidget(self._btn_revert_visible)

        layout.addWidget(self._toolbar)

        # Splitter: file_list | canvas | properties
        self._splitter = QSplitter(Qt.Horizontal)

        self._file_list = FileListWidget()
        self._file_list.setObjectName("annotationFileList")
        self._file_list.setStyleSheet(
            f"""
            QListWidget#annotationFileList::indicator:unchecked {{
                width: 14px;
                height: 14px;
                border: 1px solid {PALETTE['line_strong']};
                border-radius: 3px;
                background-color: {PALETTE['panel_alt']};
            }}

            QListWidget#annotationFileList::indicator:unchecked:hover {{
                border: 2px solid {PALETTE['primary_hover']};
                background-color: {PALETTE['primary_soft']};
            }}

            QListWidget#annotationFileList::item:hover {{
                background-color: {PALETTE['panel_raised']};
                color: {PALETTE['text']};
            }}

            QListWidget#annotationFileList::item:selected {{
                background-color: {PALETTE['primary_soft']};
                color: {PALETTE['text']};
            }}
            """
        )
        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        version_header = QHBoxLayout()
        version_header.setContentsMargins(0, 0, 0, 0)
        version_header.setSpacing(4)
        version_label = QLabel("数据版本")
        version_header.addWidget(version_label)
        version_header.addStretch(1)
        self._btn_add_data_folder = QPushButton(icon("new_project"), "新建")
        self._btn_add_data_folder.setFixedHeight(28)
        self._btn_add_data_folder.setMinimumWidth(64)
        self._btn_add_data_folder.setToolTip("新建数据版本")
        set_button_role(self._btn_add_data_folder, "secondary")
        version_header.addWidget(self._btn_add_data_folder)
        left_layout.addLayout(version_header)

        self._data_tree = QTreeWidget()
        self._data_tree.setHeaderHidden(True)
        self._data_tree.setObjectName("dataFolderTree")

        tree_closed_icon = stylesheet_url(TREE_CLOSED_SVG)
        tree_open_icon = stylesheet_url(TREE_OPEN_SVG)
        self._data_tree.setStyleSheet(
            "QTreeWidget#dataFolderTree {"
            " show-decoration-selected: 1;"
            "}"
            "QTreeWidget#dataFolderTree::item {"
            " min-height: 28px;"
            " padding: 5px 8px;"
            " margin: 0px;"
            " border: none;"
            "}"
            "QTreeWidget#dataFolderTree::item:hover {"
            " background-color: transparent;"
            f" color: {PALETTE['text']};"
            " padding: 5px 8px;"
            " margin: 0px;"
            " border: none;"
            "}"
            "QTreeWidget#dataFolderTree::item:selected,"
            "QTreeWidget#dataFolderTree::item:selected:hover {"
            f" background-color: {PALETTE['primary_soft']};"
            f" color: {PALETTE['text']};"
            " padding: 5px 8px;"
            " margin: 0px;"
            " border: none;"
            "}"
            "QTreeWidget#dataFolderTree::branch {"
            " width: 18px;"
            " min-height: 28px;"
            " background: transparent;"
            "}"
            "QTreeWidget#dataFolderTree::branch:has-children:closed {"
            f" image: url({tree_closed_icon});"
            "}"
            "QTreeWidget#dataFolderTree::branch:has-children:open {"
            f" image: url({tree_open_icon});"
            "}"
            "QTreeWidget#dataFolderTree::branch:!has-children {"
            " image: none;"
            "}"
        )
        self._data_tree.setMaximumHeight(150)
        self._data_tree.setIndentation(14)
        self._data_tree.setRootIsDecorated(True)
        self._data_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        left_layout.addWidget(self._data_tree, 0)

        left_layout.addWidget(self._file_list, 1)
        left_pane.setMinimumWidth(220)
        left_pane.setMaximumWidth(320)
        self._splitter.addWidget(left_pane)

        self._canvas = AnnotationCanvas()
        self._canvas.setMinimumSize(420, 320)
        self._splitter.addWidget(self._canvas)

        self._ann_panel = AnnotationPanel()
        self._ann_panel.setMinimumWidth(280)
        self._ann_panel.setMaximumWidth(340)
        self._splitter.addWidget(self._ann_panel)

        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setStretchFactor(2, 0)
        self._splitter.setSizes([320, 960, 300])

        layout.addWidget(self._splitter, 1)

    def _connect_signals(self) -> None:
        # Tool buttons
        self._btn_select.clicked.connect(lambda: self._set_tool("select"))
        self._btn_bbox.clicked.connect(lambda: self._set_tool("draw_bbox"))
        self._btn_polygon.clicked.connect(lambda: self._activate_polygon_tool(False))
        self._btn_obb.clicked.connect(lambda: self._activate_polygon_tool(True))
        self._btn_keypoint.clicked.connect(lambda: self._set_tool("draw_keypoint"))

        # File list
        self._file_list.image_selected.connect(self._on_image_selected)
        self._file_list.images_dropped.connect(self.images_dropped.emit)
        self._file_list.batch_confirm_requested.connect(self._on_batch_confirm)
        self._file_list.batch_delete_requested.connect(self._on_batch_delete)
        self._file_list.delete_images_requested.connect(self._on_delete_images)
        self._file_list.move_images_requested.connect(self._on_move_images)
        self._data_tree.currentItemChanged.connect(self._on_data_folder_selected)
        self._data_tree.itemClicked.connect(self._on_data_folder_clicked)
        self._data_tree.customContextMenuRequested.connect(
            self._on_data_folder_context_menu
        )
        self._btn_add_data_folder.clicked.connect(self._on_add_data_folder)

        # Canvas signals
        self._canvas.annotation_selected.connect(self._on_annotation_selected)
        self._canvas.annotation_created.connect(self._on_annotation_created)
        self._canvas.annotation_modified.connect(self._on_annotation_modified)
        self._canvas.annotation_deleted.connect(self._on_annotation_deleted)
        self._canvas.class_requested.connect(self._on_class_requested)
        self._canvas.class_change_requested.connect(self._on_class_change_requested)
        self._canvas.annotations_changed.connect(self._on_annotations_changed)
        self._canvas.annotation_copied.connect(self._on_annotation_copied)
        self._canvas.keypoint_attach_requested.connect(self._on_keypoint_attach_requested)
        self._canvas.keypoint_selected.connect(self._ann_panel.select_keypoint)

        self._canvas.tool_mode_requested.connect(self._on_canvas_tool_mode_requested)

        # Properties panel
        self._ann_panel.annotation_clicked.connect(self._canvas.select_annotation)
        self._ann_panel.annotation_class_change_requested.connect(
            self._on_panel_annotation_class_change
        )
        self._ann_panel.annotation_confirm_requested.connect(
            self._on_panel_annotation_confirm
        )
        self._ann_panel.annotation_delete_requested.connect(self._on_annotation_deleted)
        self._ann_panel.clear_all_annotations_requested.connect(
            self._on_clear_all_annotations
        )
        self._ann_panel.keypoint_add_requested.connect(self._on_panel_keypoint_add)
        self._ann_panel.keypoint_clicked.connect(self._on_panel_keypoint_clicked)
        self._ann_panel.keypoint_rename_requested.connect(self._on_keypoint_rename)
        self._ann_panel.keypoint_visibility_requested.connect(self._on_keypoint_visibility)
        self._ann_panel.keypoint_delete_requested.connect(self._on_keypoint_delete)
        self._ann_panel.default_class_changed.connect(self._on_default_class_changed)
        self._ann_panel.image_user_tags_changed.connect(self._on_user_tags_edited)

        self._btn_confirm_visible.clicked.connect(self._batch_confirm_visible)
        self._btn_revert_visible.clicked.connect(self._batch_revert_visible)



    def set_project(self, project: ProjectManager) -> None:
        self._project = project
        self._current_image_path = None
        self._current_annotation = None
        self._canvas.clear()

        # Normalize legacy/case-variant task values before configuring tools.
        self._task_type = str(project.config.task_type).strip().casefold()
        is_obb = self._task_type == "obb"
        self._btn_polygon.setVisible(self._task_type in {"segment", "obb"})
        self._btn_obb.setVisible(is_obb)
        self._obb_tool_active = False
        self._canvas.set_polygon_point_limit(None)
        self._canvas.set_obb_editing_enabled(is_obb)
        self._set_tool("select")
        self._btn_keypoint.setVisible(self._task_type == "pose")

        self._refresh_data_folder_tree()
        images = self._load_active_data_folder(select_first=True)
        logger.info("DetectPoseView loaded: %s (%d images)", project.config.name, len(images))
        self._init_stats_cache()

    def set_class_colors(self, colors: dict[str, str]) -> None:
        self._canvas.set_class_colors(colors)
        self._ann_panel.set_class_colors(colors)

    def set_classes(self, classes: list[str]) -> None:
        self._ann_panel.set_classes(classes)

    def set_available_tags(self, tags: list[str]) -> None:
        """Push the project's known-tag registry into the AnnotationPanel chip popup."""
        self._ann_panel.set_available_tags(tags)

    def _folder_item(self, folder: str, label: str | None = None) -> QTreeWidgetItem:
        item = QTreeWidgetItem([label or folder])
        item.setData(0, Qt.UserRole, folder)
        return item

    def _refresh_data_folder_tree(self) -> None:
        if self._project is None:
            return
        self._refreshing_data_tree = True
        self._data_tree.clear()

        all_item = self._folder_item("", "全部图片")
        self._data_tree.addTopLevelItem(all_item)
        items_by_folder = {"": all_item}
        for folder in self._project.list_data_folders():
            parts = folder.split("/")
            parent_folder = "/".join(parts[:-1])
            item = self._folder_item(folder, parts[-1])
            parent = items_by_folder.get(parent_folder, all_item)
            parent.addChild(item)
            items_by_folder[folder] = item

        self._data_tree.expandAll()
        all_item.setExpanded(True)
        current = self._project.config.active_data_folder
        item = items_by_folder.get(current, all_item)
        self._data_tree.setCurrentItem(item)
        self._last_data_tree_item = item
        self._refreshing_data_tree = False

    def _load_active_data_folder(self, select_first: bool = False) -> list[Path]:
        if self._project is None:
            return []
        images = self._project.list_images()
        self._file_list.set_image_paths(images)
        for img_path in images:
            label_path = self._project.label_path_for(img_path)
            ia = load_annotation(label_path)
            if ia:
                self._file_list.set_status(img_path, ia.status)
                classes_in_img = {a.class_name for a in ia.annotations}
                self._file_list.set_image_classes(img_path, classes_in_img)
                self._file_list.set_image_tags(img_path, set(ia.tags))
        if select_first and images:
            self._file_list.setCurrentRow(0)
        if not images:
            self._current_image_path = None
            self._current_annotation = None
            self._canvas.clear()
            self._ann_panel.set_annotations([])
        self._emit_status()
        return images

    def _selected_data_folder(self) -> str:
        item = self._data_tree.currentItem()
        return item.data(0, Qt.UserRole) if item is not None else ""

    def _on_data_folder_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        if self._refreshing_data_tree or item is None:
            return
        if item.childCount() <= 0:
            self._last_data_tree_item = item
            return
        if item is self._last_data_tree_item:
            item.setExpanded(not item.isExpanded())
        self._last_data_tree_item = item

    def _on_data_folder_context_menu(self, pos) -> None:
        item = self._data_tree.itemAt(pos)
        if item is not None:
            self._data_tree.setCurrentItem(item)

        folder = self._selected_data_folder()
        menu = QMenu(self)
        add_action = menu.addAction("新建数据版本...")
        import_files_action = import_dir_action = None
        if folder:
            import_files_action = menu.addAction("添加图片...")
            import_dir_action = menu.addAction("添加图片目录...")
        rename_action = delete_action = None
        if folder:
            menu.addSeparator()
            rename_action = menu.addAction("重命名")
            delete_action = menu.addAction("删除")

        chosen = menu.exec_(self._data_tree.viewport().mapToGlobal(pos))
        if chosen == add_action:
            self._on_add_data_folder()
        elif import_files_action is not None and chosen == import_files_action:
            self._on_import_images_to_data_folder()
        elif import_dir_action is not None and chosen == import_dir_action:
            self._on_import_image_directory_to_data_folder()
        elif rename_action is not None and chosen == rename_action:
            self._on_rename_data_folder()
        elif delete_action is not None and chosen == delete_action:
            self._on_delete_data_folder()

    def _on_data_folder_selected(self, item: QTreeWidgetItem | None, _prev) -> None:
        if self._project is None or self._refreshing_data_tree or item is None:
            return
        self._last_data_tree_item = item
        folder = item.data(0, Qt.UserRole) or ""
        if folder == self._project.config.active_data_folder:
            return
        self._save_current()
        self._project.config.active_data_folder = folder
        self._project.save()
        self._current_image_path = None
        self._current_annotation = None
        images = self._load_active_data_folder(select_first=True)
        self._init_stats_cache()
        name = folder or "全部图片"
        self.status_changed.emit(f"已切换数据版本: {name} ({len(images)} 张)")

    def _on_add_data_folder(self) -> None:
        if self._project is None:
            return
        name, ok = QInputDialog.getText(self, "新建数据版本", "文件夹名称:")
        if not (ok and name.strip()):
            return
        try:
            folder = self._project.create_data_folder(name.strip())
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "新建数据版本失败", str(exc))
            return
        self._project.save()
        self._refresh_data_folder_tree()
        self.status_changed.emit(f"已新建数据版本: {folder}，当前仍显示全部图片")

    def _on_import_images_to_data_folder(self) -> None:
        if self._project is None:
            return
        folder = self._selected_data_folder()
        if not folder:
            return
        files, _selected_filter = QFileDialog.getOpenFileNames(
            self,
            "添加图片到数据版本",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp);;All Files (*)",
        )
        if not files:
            return
        self._import_sources_to_data_folder([Path(p) for p in files], folder)

    def _on_import_image_directory_to_data_folder(self) -> None:
        if self._project is None:
            return
        folder = self._selected_data_folder()
        if not folder:
            return
        directory = QFileDialog.getExistingDirectory(
            self,
            "添加图片目录到数据版本",
            "",
        )
        if not directory:
            return
        self._import_sources_to_data_folder([Path(directory)], folder)

    def _import_sources_to_data_folder(self, sources: list[Path], folder: str) -> None:
        if self._project is None or not sources:
            return
        self._save_current()
        try:
            imported, skipped = self._project.import_images_to_folder(sources, folder)
        except OSError as exc:
            QMessageBox.warning(self, "添加图片失败", str(exc))
            return

        if not imported and skipped:
            QMessageBox.information(
                self,
                "添加图片",
                "没有图片被添加，目标数据版本可能已有同名图片。",
            )
            return

        self._project.save()
        self._refresh_data_folder_tree()
        images = self._project.list_images()
        self._file_list.refresh_paths(images)
        self._refresh_project_stats()
        target_name = folder or "全部图片"
        msg = f"已添加 {len(imported)} 张图片到 {target_name}"
        if skipped:
            msg += f"，跳过 {len(skipped)} 张"
        self.status_changed.emit(msg)

    def _on_rename_data_folder(self) -> None:
        if self._project is None:
            return
        old = self._selected_data_folder()
        if not old:
            return
        name, ok = QInputDialog.getText(
            self, "重命名数据版本", "新文件夹名称:", text=old
        )
        if not (ok and name.strip()):
            return
        try:
            new = self._project.rename_data_folder(old, name.strip())
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "重命名数据版本失败", str(exc))
            return
        self._project.config.active_data_folder = new
        self._project.save()
        self._refresh_data_folder_tree()
        self._load_active_data_folder(select_first=True)
        self.status_changed.emit(f"已重命名数据版本: {old} -> {new}")

    def _on_delete_data_folder(self) -> None:
        if self._project is None:
            return
        folder = self._selected_data_folder()
        if not folder:
            return
        reply = QMessageBox.question(
            self,
            "删除数据版本",
            f"确定从程序中移除数据版本「{folder}」吗？\n"
            "该目录下的图片也会解除程序索引，图片和标注等原始文件仍保留在磁盘中。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._project.delete_data_folder(folder)
        self._project.save()
        self._refresh_data_folder_tree()
        self._load_active_data_folder(select_first=True)
        self.status_changed.emit(f"已解除数据版本及其图片索引，原文件已保留: {folder}")

    def set_filter(self, status: str | None) -> None:
        self._file_list.set_filter(status)

    def set_class_filter(self, cls: str | None) -> None:
        self._file_list.set_class_filter(cls)

    def set_tag_filter(self, tag_filter) -> None:
        self._file_list.set_tag_filter(tag_filter)

    def get_selected_image_paths(self) -> list[Path]:
        return list(self._file_list.get_selected_paths())

    def refresh_image_tags(self, path: Path, tags: list[str]) -> None:
        self._file_list.set_image_tags(path, set(tags))
        if path == self._current_image_path:
            self._ann_panel.set_image_user_tags(list(tags))
            # Sync to memory so next _save_current() doesn't overwrite disk
            if self._current_annotation is not None:
                self._current_annotation.tags = list(tags)

    def get_focused_image(self) -> Path | None:
        return self._current_image_path

    def get_visible_paths(self) -> list[Path]:
        return self._file_list.get_visible_paths()

    def get_all_paths(self) -> list[Path]:
        return self._file_list.get_paths()

    def focus_image(self, path: Path) -> bool:
        path = Path(path)
        for row in range(self._file_list.count()):
            item = self._file_list.item(row)
            if Path(item.data(Qt.UserRole)) == path:
                self._file_list.setCurrentRow(row)
                self._file_list.scrollToItem(item, QAbstractItemView.PositionAtCenter)
                return True
        return False

    def reload_current(self) -> None:
        """Discard in-memory state and reload current image's annotations from disk.

        Used after external writers (e.g. batch worker) modify the on-disk JSON
        for the focused image. We must NOT call _save_current() first, because
        the canvas/in-memory state is stale relative to those external writes.
        """
        if not self._project or not self._current_image_path:
            return
        label_path = self._project.label_path_for(self._current_image_path)
        ia = load_annotation(label_path)
        if ia is None:
            w, h = get_image_size(self._current_image_path)
            ia = ImageAnnotation(
                image_path=self._current_image_path.name, image_size=(w, h),
            )
        self._current_annotation = ia
        self._canvas.set_annotations(list(ia.annotations))
        self._ann_panel.set_annotations(list(ia.annotations))
        self._emit_status()
        self._prev_annotations_snapshot = self._stats_snapshot(ia.annotations)

    def commit_pending_save(self) -> None:
        self._save_current()

    def add_auto_class_prediction(self, path, class_name, confidence):
        raise NotImplementedError("DetectPoseView does not support classify predictions")

    def add_auto_annotations(self, anns: list[Annotation], iou: float = 0.5) -> None:
        from src.core.annotation import find_conflicts
        existing = self._canvas.annotations
        conflicts, clean = find_conflicts(existing, anns, iou)
        self._canvas.add_annotations(clean)
        if conflicts:
            self._canvas.add_annotations([p for _, p in conflicts])
            self._canvas.set_conflict_pairs([(e.id, p.id) for e, p in conflicts])
        self._push_undo()
        self._sync_annotations_to_panel()
        if self._current_image_path is not None:
            self.annotations_changed.emit(self._current_image_path)



    def _set_tool(self, mode: str) -> None:
        self._obb_tool_active = mode == "draw_obb"
        self._btn_select.setChecked(mode == "select")
        self._btn_bbox.setChecked(mode == "draw_bbox")
        self._btn_polygon.setChecked(mode == "draw_polygon")
        self._btn_obb.setChecked(mode == "draw_obb")
        self._btn_keypoint.setChecked(mode == "draw_keypoint")
        self._canvas.set_tool_mode(mode)

    def _activate_polygon_tool(self, oriented_box: bool) -> None:
        self._canvas.set_polygon_point_limit(None)
        self._set_tool("draw_obb" if oriented_box else "draw_polygon")

    def _on_canvas_tool_mode_requested(self, mode: str) -> None:
        if mode == "draw_obb":
            self._activate_polygon_tool(True)
        elif mode == "draw_polygon":
            self._activate_polygon_tool(False)
        else:
            self._set_tool(mode)



    def _on_image_selected(self, path: Path) -> None:
        """Switch image and annotations as one non-reentrant UI transaction."""
        path = Path(path)
        if self._switching_image:
            # Keep only the latest request. This protects against synchronous
            # selection signals fired while status/filter state is refreshed.
            self._queued_image_path = path
            return

        self._switching_image = True
        try:
            self._switch_to_image(path)
        finally:
            self._switching_image = False

        queued_path = self._queued_image_path
        self._queued_image_path = None
        if queued_path is not None and queued_path != self._current_image_path:
            self._on_image_selected(queued_path)

    def _switch_to_image(self, path: Path) -> None:
        self._save_current()
        self._canvas.cancel_interaction()
        self._current_image_path = path

        # Reassert the project tool capability whenever the canvas changes
        # image, so clearing/reloading the canvas cannot drop the OBB menu.
        self._canvas.set_obb_editing_enabled(self._task_type == "obb")

        if self._project:
            label_path = self._project.label_path_for(path)
            ia = load_annotation(label_path)
            if ia is None:
                w, h = get_image_size(path)
                ia = ImageAnnotation(
                    image_path=path.name,
                    image_size=(w, h),
                )
            self._current_annotation = ia

        self._canvas.set_loading(True)
        try:
            pixmap = self._image_cache.get(path)
            if pixmap:
                self._canvas.set_pixmap(pixmap)
            else:
                self._canvas.load_image(str(path))
        finally:
            self._canvas.set_loading(False)
        logger.debug("Image selected: %s", path.name)

        if self._project and self._current_annotation is not None:
            self._canvas.set_annotations(list(self._current_annotation.annotations))
            self._ann_panel.set_annotations(list(self._current_annotation.annotations))
            self._ann_panel.set_image_user_tags(list(self._current_annotation.tags))

            self._emit_status()

            key = str(path)
            if key not in self._undo_stacks:
                self._undo_stacks[key] = UndoStack()
                self._undo_stacks[key].push(self._current_annotation.to_dict())
            else:
                self._undo_stacks.move_to_end(key)
            while len(self._undo_stacks) > self._UNDO_MAX_IMAGES:
                self._undo_stacks.popitem(last=False)

            self._prev_annotations_snapshot = self._stats_snapshot(self._current_annotation.annotations)

        self.image_focus_changed.emit(path)

    def _save_current(self) -> None:
        if not self._project or not self._current_image_path or not self._current_annotation:
            return
        self._current_annotation.annotations = list(self._canvas.annotations)
        label_path = self._project.label_path_for(self._current_image_path)
        save_annotation(self._current_annotation, label_path)
        logger.debug("Saved annotations for %s", self._current_image_path.name)
        self._file_list.set_status(self._current_image_path, self._current_annotation.status)
        self._file_list.set_image_classes(
            self._current_image_path,
            {ann.class_name for ann in self._current_annotation.annotations},
        )
        old_snap = self._prev_annotations_snapshot or []
        new_snap = self._stats_snapshot(self._current_annotation.annotations)
        if old_snap != new_snap:
            self._update_stats_incremental(old_snap, new_snap)
            self._prev_annotations_snapshot = new_snap
            # Notify shell so it can mirror stats / push undo if needed
            self.annotations_changed.emit(self._current_image_path)



    def _on_annotation_selected(self, ann_id) -> None:
        self._ann_panel.select_annotation(ann_id)

    def _on_annotation_created(self, ann) -> None:
        self._push_undo()
        self._sync_annotations_to_panel()

    def _on_annotation_modified(self, ann_id: str) -> None:
        self._push_undo()
        self._sync_annotations_to_panel()

    def _on_annotation_deleted(self, ann_id: str) -> None:
        self._canvas.remove_annotation(ann_id)
        self._push_undo()
        self._sync_annotations_to_panel()

    def _on_clear_all_annotations(self) -> None:
        count = len(self._canvas.annotations)
        if count == 0:
            return
        reply = QMessageBox.question(
            self,
            "清空全部标注",
            f"确定清除当前图片的全部 {count} 个标注吗？\n该操作可通过 Ctrl+Z 撤销。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._canvas.set_annotations([])
        self._push_undo()
        self._sync_annotations_to_panel()

    def _on_annotations_changed(self) -> None:
        self._sync_annotations_to_panel()

    def _show_class_picker_at(
        self,
        default_class: str | None,
        global_pos,
        classes: list[str] | None = None,
    ) -> str | None:
        if not self._project:
            return None
        class_names = list(classes) if classes is not None else list(self._project.config.classes)
        colors = {cls: self._project.config.get_class_color(cls) for cls in class_names}

        picker = ClassPickerPopup(
            classes=class_names,
            colors=colors,
            default_class=default_class,
            parent=self,
        )
        picker.move_near(global_pos)
        if not picker.exec_():
            return None

        cls_name = picker.get_selected_class()
        if cls_name is None:
            return None

        if picker.is_new_class() or cls_name not in self._project.config.classes:
            self._project.add_class(cls_name)
            self._project.save()
            colors[cls_name] = self._project.config.get_class_color(cls_name)
            self._canvas.set_class_colors(colors)
            self._ann_panel.set_class_colors(colors)
            self._ann_panel.set_classes(self._project.config.classes)
            self.classes_changed.emit()

        return cls_name

    def _show_class_picker(self, default_class: str | None, px: float, py: float) -> str | None:
        global_pos = self._canvas.mapToGlobal(QPoint(int(px), int(py)))
        return self._show_class_picker_at(default_class, global_pos)

    def _on_class_requested(self, px: float, py: float) -> None:
        cls_name = self._show_class_picker(self._last_class, px, py)
        if cls_name is None:
            self._clear_draw_state()
            return

        cls_id = self._project.config.get_class_id(cls_name)
        self._set_default_class(cls_name)
        if self._canvas.tool_mode == "draw_bbox":
            self._canvas.create_bbox_from_draw(cls_name, cls_id)
        elif self._canvas.tool_mode == "draw_obb":
            self._canvas.create_obb_from_draw(cls_name, cls_id)
        elif self._canvas.tool_mode == "draw_polygon":
            self._canvas.create_polygon_from_draw(cls_name, cls_id)
        elif self._canvas.tool_mode == "draw_keypoint":
            self._canvas.create_keypoint_at(cls_name, cls_id)

    def _on_default_class_changed(self, cls_name) -> None:
        """Class set/cleared via the right-side project class list.

        cls_name is the class name when set, or None when the user toggled
        the current default off by re-double-clicking it.
        """
        self._last_class = cls_name
        if cls_name:
            self.status_changed.emit(f"默认类别: {cls_name}")
        else:
            self.status_changed.emit("已取消默认类别")

    def _on_user_tags_edited(self, new_tags: list) -> None:
        """User added/removed a chip in the per-image Tag bar.

        Persists the change, updates file_list filter cache, and fans out a
        signal so the shell can sync the project tag registry.
        """
        if not self._project or not self._current_image_path or self._current_annotation is None:
            return
        tags = [str(t) for t in new_tags]
        self._current_annotation.tags = list(tags)
        # Reuse the canonical save path so annotations stay in sync.
        self._save_current()
        self._file_list.set_image_tags(self._current_image_path, set(tags))
        self.user_tags_changed.emit(self._current_image_path, list(tags))

    def _set_default_class(self, cls_name: str) -> None:
        """Update `_last_class` and keep the side-panel highlight in sync."""
        self._last_class = cls_name
        self._ann_panel.set_default_class(cls_name)

    def _clear_draw_state(self) -> None:
        self._canvas.clear_draw_state()
        self._set_tool("select")

    def _on_class_change_requested(self, ann_id: str, px: float, py: float) -> None:
        ann = next((a for a in self._canvas.annotations if a.id == ann_id), None)
        if ann is None:
            return

        cls_name = self._show_class_picker(ann.class_name, px, py)
        if cls_name is None or cls_name == ann.class_name:
            return

        old_class = ann.class_name
        if self._apply_selected_images_class_change(old_class, cls_name):
            return
        self._apply_annotation_class_change(ann, cls_name)

    def _apply_annotation_class_change(
        self,
        annotation: Annotation,
        new_class: str,
    ) -> None:
        """Change exactly one annotation, identified by its object/unique ID."""
        if not self._project or annotation.class_name == new_class:
            return
        annotation.class_name = new_class
        annotation.class_id = self._project.config.get_class_id(new_class)
        self._push_undo()
        self._canvas.update()
        self._sync_annotations_to_panel()

    def _apply_selected_images_class_change(
        self,
        old_class: str,
        new_class: str,
    ) -> bool:
        """Relabel matching annotations across the multi-image selection.

        Returns ``True`` only when a multi-image selection was handled. A
        normal single-image change keeps the precise annotation-by-ID behavior.
        """
        if not self._project or old_class == new_class:
            return False
        selected_paths = list(dict.fromkeys(self._file_list.get_selected_paths()))
        if len(selected_paths) <= 1:
            return False
        if (
            self._current_image_path is not None
            and self._current_image_path not in selected_paths
        ):
            selected_paths.append(self._current_image_path)

        self._save_current()
        new_class_id = self._project.config.get_class_id(new_class)
        changed_images = 0
        changed_annotations = 0
        current_changed = False

        for image_path in selected_paths:
            label_path = self._project.label_path_for(image_path)
            image_annotation = load_annotation(label_path)
            if image_annotation is None:
                continue
            old_state = image_annotation.to_dict()
            old_snapshot = self._stats_snapshot(image_annotation.annotations)
            changed_here = 0
            for annotation in image_annotation.annotations:
                if annotation.class_name != old_class:
                    continue
                annotation.class_name = new_class
                annotation.class_id = new_class_id
                changed_here += 1
            if changed_here <= 0:
                continue

            save_annotation(image_annotation, label_path)
            self._file_list.set_status(image_path, image_annotation.status)
            self._file_list.set_image_classes(
                image_path,
                {annotation.class_name for annotation in image_annotation.annotations},
            )
            self._update_stats_incremental(
                old_snapshot,
                self._stats_snapshot(image_annotation.annotations),
            )

            key = str(image_path)
            stack = self._undo_stacks.get(key)
            if stack is None:
                stack = UndoStack()
                stack.push(old_state)
                self._undo_stacks[key] = stack
            else:
                self._undo_stacks.move_to_end(key)
            stack.push(image_annotation.to_dict())
            while len(self._undo_stacks) > self._UNDO_MAX_IMAGES:
                self._undo_stacks.popitem(last=False)

            changed_images += 1
            changed_annotations += changed_here
            current_changed = current_changed or image_path == self._current_image_path
            self.annotations_changed.emit(image_path)

        if current_changed:
            self.reload_current()
        self.status_changed.emit(
            f"已批量修改类别: {old_class} → {new_class}，"
            f"{changed_images} 张图片 / {changed_annotations} 个标注"
        )
        return True

    def _on_panel_annotation_class_change(self, ann_id: str) -> None:
        """Change one annotation, or relabel its class across selected images."""
        if not self._project:
            return

        ann = next((a for a in self._canvas.annotations if a.id == ann_id), None)
        if ann is None:
            return

        old_class = ann.class_name
        classes = merged_project_annotation_classes(self._project)
        cls_name = self._show_class_picker_at(
            old_class,
            QCursor.pos(),
            classes=classes,
        )
        if not cls_name or cls_name == old_class:
            return

        if self._apply_selected_images_class_change(old_class, cls_name):
            return
        self._apply_annotation_class_change(ann, cls_name)
        self.status_changed.emit(f"已将当前标注从「{old_class}」修改为「{cls_name}」")

    def _on_panel_annotation_confirm(self, ann_id: str, confirmed: bool) -> None:
        ann = next((a for a in self._canvas.annotations if a.id == ann_id), None)
        if ann is None:
            return
        ann.confirmed = bool(confirmed)
        self._push_undo()
        self._canvas.update()
        self._sync_annotations_to_panel()
        if self._current_image_path is not None:
            self.annotations_changed.emit(self._current_image_path)

    def _on_panel_keypoint_add(self, ann_id: str) -> None:
        ann = next((a for a in self._canvas.annotations if a.id == ann_id), None)
        if ann is None:
            return
        label, ok = QInputDialog.getText(
            self,
            "新增关键点",
            "关键点名称:",
            text=f"kp_{len(ann.keypoints)}",
        )
        if not (ok and label.strip()):
            return
        if ann.bbox:
            x, y = ann.bbox[0], ann.bbox[1]
        elif ann.polygon:
            x = sum(p[0] for p in ann.polygon) / len(ann.polygon)
            y = sum(p[1] for p in ann.polygon) / len(ann.polygon)
        else:
            x, y = 0.5, 0.5
        self._canvas.add_keypoint_to_annotation(
            ann_id, Keypoint(x=x, y=y, visible=2, label=label.strip())
        )

    def _on_keypoint_attach_requested(self, ann_id: str, px: float, py: float) -> None:
        ann = next((a for a in self._canvas.annotations if a.id == ann_id), None)
        if ann is None or not self._canvas._draw_start:
            self._clear_draw_state()
            return

        existing_labels: list[str] = []
        seen: set[str] = set()
        for a in self._canvas.annotations:
            for kp in a.keypoints:
                if kp.label not in seen:
                    existing_labels.append(kp.label)
                    seen.add(kp.label)

        default_label = f"kp_{len(ann.keypoints)}"

        picker = KeypointLabelPicker(
            existing_labels=existing_labels,
            default_label=default_label,
            parent=self,
        )
        global_pos = self._canvas.mapToGlobal(QPoint(int(px), int(py)))
        picker.move_near(global_pos)

        if not picker.exec_():
            self._clear_draw_state()
            return

        label = picker.get_label()
        if not label:
            self._clear_draw_state()
            return

        nx, ny = self._canvas._draw_start
        kp = Keypoint(x=nx, y=ny, visible=2, label=label)
        self._canvas.add_keypoint_to_annotation(ann_id, kp)
        self._canvas._draw_start = None
        self._push_undo()
        self._sync_annotations_to_panel()

    def _on_panel_keypoint_clicked(self, ann_id: str, kp_idx: int) -> None:
        self._canvas.select_keypoint(ann_id, kp_idx)

    def _on_keypoint_rename(self, ann_id: str, kp_idx: int, new_label: str) -> None:
        self._canvas.rename_keypoint(ann_id, kp_idx, new_label)
        self._push_undo()
        self._sync_annotations_to_panel()

    def _on_keypoint_visibility(self, ann_id: str, kp_idx: int) -> None:
        self._canvas.cycle_keypoint_visibility(ann_id, kp_idx)
        self._push_undo()
        self._sync_annotations_to_panel()

    def _on_keypoint_delete(self, ann_id: str, kp_idx: int) -> None:
        self._canvas.remove_keypoint(ann_id, kp_idx)
        self._push_undo()
        self._sync_annotations_to_panel()

    def _sync_annotations_to_panel(self) -> None:
        self._ann_panel.set_annotations(list(self._canvas.annotations))
        self._save_current()
        self._emit_status()

    def _emit_status(self) -> None:
        if not self._current_image_path:
            return
        idx, total = self._file_list.get_index_info()
        n_ann = len(self._canvas.annotations)
        n_confirmed = sum(1 for a in self._canvas.annotations if a.confirmed)
        n_pending = n_ann - n_confirmed
        parts = [
            self._current_image_path.name,
            f"{idx}/{total}",
            f"标注: {n_ann}",
        ]
        if n_pending > 0:
            parts.append(f"确认: {n_confirmed} 待确认: {n_pending}")
        self.status_changed.emit(" | ".join(parts))



    def _compute_project_stats(self) -> dict:
        if not self._project:
            return {}
        stats = {
            "total_images": 0,
            "labeled_images": 0,
            "confirmed_images": 0,
            "total_annotations": 0,
            "class_counts": {},
        }
        images = self._project.list_images()
        stats["total_images"] = len(images)
        for img_path in images:
            label_path = self._project.label_path_for(img_path)
            ia = load_annotation(label_path)
            if ia is None or len(ia.annotations) == 0:
                continue
            stats["labeled_images"] += 1
            all_confirmed = all(a.confirmed for a in ia.annotations)
            if all_confirmed:
                stats["confirmed_images"] += 1
            for ann in ia.annotations:
                stats["total_annotations"] += 1
                stats["class_counts"][ann.class_name] = stats["class_counts"].get(ann.class_name, 0) + 1
        return stats

    def _init_stats_cache(self) -> None:
        self._stats_cache = self._compute_project_stats()
        self._ann_panel.set_project_stats(self._stats_cache)

    def _update_stats_incremental(self, old_snap: list[tuple], new_snap: list[tuple]) -> None:
        if not self._stats_cache:
            return
        had_old = len(old_snap) > 0
        has_new = len(new_snap) > 0

        if had_old and not has_new:
            self._stats_cache["labeled_images"] -= 1
        elif not had_old and has_new:
            self._stats_cache["labeled_images"] += 1

        old_all_confirmed = had_old and all(c for _, c in old_snap)
        new_all_confirmed = has_new and all(c for _, c in new_snap)
        if old_all_confirmed and not new_all_confirmed:
            self._stats_cache["confirmed_images"] -= 1
        elif not old_all_confirmed and new_all_confirmed:
            self._stats_cache["confirmed_images"] += 1

        for cls, _ in old_snap:
            self._stats_cache["total_annotations"] -= 1
            self._stats_cache["class_counts"][cls] = self._stats_cache["class_counts"].get(cls, 1) - 1
            if self._stats_cache["class_counts"][cls] <= 0:
                del self._stats_cache["class_counts"][cls]

        for cls, _ in new_snap:
            self._stats_cache["total_annotations"] += 1
            self._stats_cache["class_counts"][cls] = self._stats_cache["class_counts"].get(cls, 0) + 1

        self._ann_panel.set_project_stats(self._stats_cache)

    @staticmethod
    def _stats_snapshot(anns) -> list[tuple]:
        return [(a.class_name, a.confirmed) for a in anns]

    def _refresh_project_stats(self) -> None:
        """Recompute and refresh stats panel after a project-wide change."""
        self._init_stats_cache()

    def _confirm_all(self) -> None:
        for ann in self._canvas.annotations:
            ann.confirmed = True
        self._push_undo()
        self._canvas.update()
        self._sync_annotations_to_panel()
        if self._current_image_path is not None:
            status = "confirmed"
            if self._current_annotation is not None:
                self._current_annotation.annotations = list(self._canvas.annotations)
                status = self._current_annotation.status
            self._file_list.set_status(self._current_image_path, status)
            self.annotations_changed.emit(self._current_image_path)



    def _copy_annotation(self) -> None:
        ann = self._canvas.get_selected_annotation()
        if ann:
            self._clipboard = [ann.to_dict()]
            logger.debug("Copied annotation: %s", ann.class_name)

    def _on_annotation_copied(self, ann_id: str) -> None:
        for ann in self._canvas.annotations:
            if ann.id == ann_id:
                self._clipboard = [ann.to_dict()]
                logger.debug("Copied annotation via menu: %s", ann.class_name)
                break

    def _paste_annotation(self) -> None:
        if not self._clipboard or self._canvas.is_locked:
            return
        import uuid as _uuid
        new_anns = []
        for ann_dict in self._clipboard:
            new_dict = dict(ann_dict)
            new_dict["id"] = str(_uuid.uuid4())
            new_dict["confirmed"] = False
            new_anns.append(Annotation.from_dict(new_dict))
        self._canvas.add_annotations(new_anns)
        self._push_undo()
        self._sync_annotations_to_panel()
        logger.debug("Pasted %d annotations", len(self._clipboard))



    def _push_undo(self) -> None:
        if not self._current_image_path or not self._current_annotation:
            return
        self._current_annotation.annotations = list(self._canvas.annotations)
        key = str(self._current_image_path)
        if key not in self._undo_stacks:
            self._undo_stacks[key] = UndoStack()
        else:
            self._undo_stacks.move_to_end(key)
        self._undo_stacks[key].push(self._current_annotation.to_dict())
        while len(self._undo_stacks) > self._UNDO_MAX_IMAGES:
            self._undo_stacks.popitem(last=False)

    def undo(self) -> None:
        if not self._current_image_path:
            return
        key = str(self._current_image_path)
        stack = self._undo_stacks.get(key)
        if not stack or not stack.can_undo:
            return
        state = stack.undo()
        if state:
            self._restore_state(state)

    def redo(self) -> None:
        if not self._current_image_path:
            return
        key = str(self._current_image_path)
        stack = self._undo_stacks.get(key)
        if not stack or not stack.can_redo:
            return
        state = stack.redo()
        if state:
            self._restore_state(state)

    def _restore_state(self, state: dict) -> None:
        ia = ImageAnnotation.from_dict(state)
        self._current_annotation = ia
        self._canvas.set_annotations(list(ia.annotations))
        self._sync_annotations_to_panel()



    def keyPressEvent(self, event) -> None:
        key = event.key()
        mod = event.modifiers()

        if key == Qt.Key_W:
            self._set_tool("draw_bbox")
        elif key == Qt.Key_P:
            self._activate_polygon_tool(False)
        elif key == Qt.Key_O and self._btn_obb.isVisible():
            self._activate_polygon_tool(True)
        elif key == Qt.Key_K:
            self._set_tool("draw_keypoint")
        elif key in (Qt.Key_Return, Qt.Key_Enter) and self._canvas.tool_mode == "draw_polygon":
            self._canvas.finish_polygon()
        elif key == Qt.Key_Escape and self._canvas.tool_mode in {
            "draw_bbox", "draw_obb", "draw_polygon", "draw_keypoint"
        }:
            if self._canvas.tool_mode == "draw_polygon":
                self._canvas.cancel_polygon()
            else:
                self._canvas.clear_draw_state()
            self._set_tool("select")
        elif key == Qt.Key_V and not (mod & Qt.ControlModifier):
            self._set_tool("select")
        elif key == Qt.Key_D or key == Qt.Key_Right:
            self._save_current()
            self._file_list.go_next()
        elif (key == Qt.Key_A and not (mod & (Qt.ShiftModifier | Qt.ControlModifier))) or key == Qt.Key_Left:
            self._save_current()
            self._file_list.go_prev()
        elif key == Qt.Key_Space and mod & Qt.ControlModifier:
            self._batch_confirm_visible()
        elif key == Qt.Key_Space:
            ann = self._canvas.get_selected_annotation()
            if ann:
                ann.confirmed = True
                self._push_undo()
                self._canvas.update()
                self._sync_annotations_to_panel()
        elif key == Qt.Key_Delete:
            ann = self._canvas.get_selected_annotation()
            if ann:
                self._on_annotation_deleted(ann.id)
        elif key == Qt.Key_C and mod & Qt.ControlModifier:
            self._copy_annotation()
        elif key == Qt.Key_V and mod & Qt.ControlModifier:
            self._paste_annotation()
        elif key in (Qt.Key_Plus, Qt.Key_Equal) and mod & Qt.ControlModifier:
            self._canvas.zoom_in()
        elif key == Qt.Key_Minus and mod & Qt.ControlModifier:
            self._canvas.zoom_out()
        elif key == Qt.Key_0 and mod & Qt.ControlModifier:
            self._canvas.zoom_fit()
        else:
            super().keyPressEvent(event)



    def _on_batch_confirm(self, paths: list[Path]) -> None:
        if not self._project:
            return
        self._save_current()
        count = 0
        for img_path in paths:
            label_path = self._project.label_path_for(img_path)
            ia = load_annotation(label_path)
            if ia and ia.annotations:
                old_snap = self._stats_snapshot(ia.annotations)
                for ann in ia.annotations:
                    ann.confirmed = True
                save_annotation(ia, label_path)
                self._file_list.set_status(img_path, ia.status)
                self._update_stats_incremental(old_snap, self._stats_snapshot(ia.annotations))
                count += 1
        if self._current_image_path and self._current_image_path in paths:
            self.reload_current()
        self.status_changed.emit(f"批量确认: {count} 张图片")
        logger.info("Batch confirmed %d images", count)

    def _on_batch_delete(self, paths: list[Path]) -> None:
        if not self._project:
            return
        self._save_current()
        count = 0
        for img_path in paths:
            label_path = self._project.label_path_for(img_path)
            ia = load_annotation(label_path)
            if ia and ia.annotations:
                old_snap = self._stats_snapshot(ia.annotations)
                ia.annotations.clear()
                save_annotation(ia, label_path)
                self._file_list.set_status(img_path, "unlabeled")
                self._file_list.set_image_classes(img_path, set())
                self._update_stats_incremental(old_snap, [])
                count += 1
        if self._current_image_path and self._current_image_path in paths:
            self.reload_current()
        self.status_changed.emit(f"批量删除标注: {count} 张图片")
        logger.info("Batch deleted annotations for %d images", count)

    def _on_delete_images(self, paths: list[Path]) -> None:
        """Delete image files and their labels from disk after confirmation."""
        if not self._project or not paths:
            return
        paths = list(paths)

        labeled_count = 0
        for p in paths:
            ia = load_annotation(self._project.label_path_for(p))
            if ia and ia.annotations:
                labeled_count += 1

        n = len(paths)
        if labeled_count:
            msg = (
                f"确定要删除 {n} 张图片吗？\n"
                f"其中 {labeled_count} 张包含标注，将一并删除。\n\n"
                "此操作不可撤销。"
            )
        else:
            msg = f"确定要删除 {n} 张图片吗？\n\n此操作不可撤销。"
        reply = QMessageBox.question(
            self, "删除图片", msg,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._save_current()
        current_row = self._file_list.currentRow()
        scroll_value = self._file_list.verticalScrollBar().value()
        current_in_deleted = (
            self._current_image_path is not None
            and self._current_image_path in paths
        )

        img_n, lbl_n = self._project.delete_images(paths)

        for p in paths:
            self._image_cache.invalidate(p)
            self._undo_stacks.pop(str(p), None)
        self._file_list.forget_paths(paths)

        if current_in_deleted:
            self._current_image_path = None
            self._current_annotation = None
            self._prev_annotations_snapshot = None
            self._canvas.clear()
            self._ann_panel.set_annotations([])

        remaining = self._project.list_images()
        self._file_list.refresh_paths(remaining)

        if current_in_deleted and remaining:
            self._file_list.select_nearest_visible_row(current_row, scroll_value)

        self._refresh_project_stats()
        self.status_changed.emit(
            f"已删除 {img_n} 张图片，{lbl_n} 个标注文件"
        )
        logger.info("Deleted %d images, %d labels", img_n, lbl_n)

    def _on_move_images(self, paths: list[Path]) -> None:
        """Move selected images and their labels into another data folder."""
        if not self._project or not paths:
            return
        folders = ["全部图片"] + self._project.list_data_folders()
        folders.append("新建文件夹...")
        choice, ok = QInputDialog.getItem(
            self,
            "移动到数据版本",
            "目标文件夹:",
            folders,
            0,
            False,
        )
        if not ok:
            return
        if choice == "新建文件夹...":
            name, ok = QInputDialog.getText(
                self, "新建数据版本", "文件夹名称:"
            )
            if not (ok and name.strip()):
                return
            try:
                target_folder = self._project.create_data_folder(name.strip())
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, "新建数据版本失败", str(exc))
                return
        else:
            target_folder = "" if choice == "全部图片" else choice

        self._save_current()
        current_in_moved = (
            self._current_image_path is not None
            and self._current_image_path in paths
        )
        try:
            moved, skipped = self._project.move_images_to_folder(paths, target_folder)
        except OSError as exc:
            QMessageBox.warning(self, "移动图片失败", str(exc))
            return

        if not moved and skipped:
            QMessageBox.information(
                self,
                "移动图片",
                "没有图片被移动，目标文件夹可能已有同名文件。",
            )
            return

        for p in paths:
            self._image_cache.invalidate(p)
            self._undo_stacks.pop(str(p), None)
        for p in moved:
            self._image_cache.invalidate(p)

        if current_in_moved:
            self._current_image_path = None
            self._current_annotation = None
            self._prev_annotations_snapshot = None
            self._canvas.clear()
            self._ann_panel.set_annotations([])

        self._refresh_data_folder_tree()
        images = self._project.list_images()
        self._file_list.refresh_paths(images)
        if current_in_moved and images:
            self._file_list.setCurrentRow(0)
        self._refresh_project_stats()

        target_name = target_folder or "全部图片"
        msg = f"已移动 {len(moved)} 张图片到 {target_name}"
        if skipped:
            msg += f"，跳过 {len(skipped)} 张"
        self.status_changed.emit(msg)
        logger.info("Moved %d images to %s, skipped %d", len(moved), target_name, len(skipped))

    def _collect_unconfirmed(self, visible_paths: list[Path]):
        affected = []
        total = 0
        for img_path in visible_paths:
            label_path = self._project.label_path_for(img_path)
            ia = load_annotation(label_path)
            if ia:
                unconfirmed = sum(1 for a in ia.annotations if not a.confirmed)
                if unconfirmed > 0:
                    affected.append((img_path, label_path, ia))
                    total += unconfirmed
        return affected, total

    def _batch_confirm_visible(self) -> None:
        if not self._project:
            return
        visible_paths = self._file_list.get_visible_paths()
        if not visible_paths:
            return

        affected, total = self._collect_unconfirmed(visible_paths)
        if total == 0:
            self.status_changed.emit("没有需要确认的预标注")
            return

        reply = QMessageBox.question(
            self, "确认可见预标注",
            f"将确认 {len(affected)} 张图片中的 {total} 个未确认标注，是否继续？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._save_current()
        count = 0
        for img_path, label_path, ia in affected:
            old_snap = self._stats_snapshot(ia.annotations)
            for ann in ia.annotations:
                if not ann.confirmed:
                    ann.confirmed = True
            save_annotation(ia, label_path)
            self._file_list.set_status(img_path, ia.status)
            self._update_stats_incremental(old_snap, self._stats_snapshot(ia.annotations))
            count += 1

        if self._current_image_path and self._current_image_path in visible_paths:
            self.reload_current()
        self.status_changed.emit(f"已确认可见预标注: {count} 张图片")
        logger.info("Batch confirmed visible unconfirmed annotations for %d images", count)

    def _batch_revert_visible(self) -> None:
        if not self._project:
            return
        visible_paths = self._file_list.get_visible_paths()
        if not visible_paths:
            return

        affected, total = self._collect_unconfirmed(visible_paths)
        if total == 0:
            self.status_changed.emit("没有需要撤销的预标注")
            return

        reply = QMessageBox.question(
            self, "撤销可见预标注",
            f"将删除 {len(affected)} 张图片中的 {total} 个未确认标注，此操作不可撤销，是否继续？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._save_current()
        count = 0
        for img_path, label_path, ia in affected:
            old_snap = self._stats_snapshot(ia.annotations)
            ia.annotations = [a for a in ia.annotations if a.confirmed]
            save_annotation(ia, label_path)
            self._file_list.set_status(img_path, ia.status)
            self._file_list.set_image_classes(
                img_path,
                {ann.class_name for ann in ia.annotations},
            )
            self._update_stats_incremental(old_snap, self._stats_snapshot(ia.annotations))
            count += 1

        if self._current_image_path and self._current_image_path in visible_paths:
            self.reload_current()
        self.status_changed.emit(f"已撤销可见预标注: {count} 张图片")
        logger.info("Batch reverted visible unconfirmed annotations for %d images", count)



    def get_unlabeled_image_paths(self) -> list[Path]:
        if not self._project:
            return []
        result = []
        for img_path in self._project.list_images():
            label_path = self._project.label_path_for(img_path)
            ia = load_annotation(label_path)
            if ia is None or len(ia.annotations) == 0:
                result.append(img_path)
        return result

    def refresh_image_list(self, images: list[Path]) -> None:
        """Called by shell after rescan / drop to refresh the file list."""
        self._refresh_data_folder_tree()
        self._file_list.refresh_paths(images)
        # Recompute stats from scratch (project-wide change)
        self._refresh_project_stats()
