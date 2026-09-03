"""Annotation list and properties panel."""
from __future__ import annotations

from PyQt5.QtWidgets import (
    QSizePolicy,
    QGridLayout,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QMenu,
    QInputDialog,
    QFrame,
    QPushButton,
    QStyle,
    QStyledItemDelegate,
)
from PyQt5.QtCore import Qt, pyqtSignal, QRect, QSize, QPoint
from PyQt5.QtGui import QColor, QPixmap, QIcon, QPainter, QPen, QFont, QFontMetrics

from src.core.annotation import Annotation, annotation_area_text
from src.ui.collapsible_group import CollapsibleGroupBox
from src.ui.tag_widget import TagChipBar
from src.ui.theme import PALETTE, set_button_role, text_style


class _ClassRow(QWidget):
    """Per-row widget for the project class list.

    QListWidget.itemDoubleClicked does not fire when setItemWidget is used —
    the item widget covers the viewport area and intercepts mouse events even
    with WA_TransparentForMouseEvents. So each row carries its own signal.
    """

    clicked = pyqtSignal(str)  # class name

    def __init__(self, cls_name: str, parent=None):
        super().__init__(parent)
        self._cls_name = cls_name

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self._cls_name)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class _AnnCardDelegate(QStyledItemDelegate):
    """标注列表卡片渲染器（设计稿样式）。

    顶层标注项画成一张卡片：色块 + 状态图标 + 类别名（第一行）、
    形状·面积（第二行）、右侧状态圈（已确认=绿圈✓，待确认=虚线圈）。
    关键点子项走默认渲染。
    """

    CARD_ROLE = Qt.UserRole + 2   # 形状·面积 文本（有值 = 卡片模式）
    CONFIDENCE_ROLE = Qt.UserRole + 3
    STATE_ROLE = Qt.UserRole + 4  # bool 已确认

    @staticmethod
    def _elided_text(font: QFont, text: str, width: int) -> str:
        """Keep card copy within its reserved lane at narrow inspector widths."""
        return QFontMetrics(font).elidedText(str(text), Qt.ElideRight, max(0, width))

    def paint(self, painter, option, index):
        shape = index.data(self.CARD_ROLE)
        if not shape:
            super().paint(painter, option, index)
            return
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        rect = option.rect.adjusted(-option.rect.x() + 2, 3, -2, -3)
        selected = bool(option.state & QStyle.State_Selected)

        painter.setPen(QPen(QColor(PALETTE["primary"] if selected
                                   else PALETTE["line"]), 1))
        painter.setBrush(QColor(PALETTE["primary_soft"] if selected
                                else PALETTE["bg"]))
        painter.drawRoundedRect(rect, 8, 8)

        # 色块
        color = QColor(index.data(Qt.UserRole) and
                       PALETTE.get("primary") or PALETTE["primary"])
        try:
            color = QColor(index.data(Qt.ForegroundRole))
        except (TypeError, ValueError):
            pass
        if not color.isValid():
            color = QColor(PALETTE["primary"])
        sw = rect.adjusted(10, rect.height() // 2 - 7, 0, 0)
        sw.setSize(QSize(14, 14))
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)
        painter.drawRoundedRect(sw, 4, 4)

        # 第一行：类别名 + 置信度
        name_x = sw.right() + 10
        name = index.data(Qt.DisplayRole) or ""
        f1 = QFont(option.font)
        f1.setPixelSize(13)
        f1.setWeight(QFont.DemiBold)
        confidence = index.data(self.CONFIDENCE_ROLE) or "—"
        confidence_rect = QRect(rect.right() - 66, rect.top() + 8, 34, 20)
        name_rect = QRect(
            name_x, rect.top() + 8,
            max(0, confidence_rect.left() - name_x - 8), 20,
        )
        painter.setFont(f1)
        painter.setPen(QColor(PALETTE["text"]))
        painter.drawText(
            name_rect, Qt.AlignLeft | Qt.AlignVCenter,
            self._elided_text(f1, name, name_rect.width()),
        )

        f_conf = QFont(option.font)
        f_conf.setPixelSize(11)
        f_conf.setFamily("Menlo")
        painter.setFont(f_conf)
        painter.setPen(QColor(PALETTE["text_muted"]))
        painter.drawText(confidence_rect, Qt.AlignRight | Qt.AlignVCenter, confidence)

        # 第二行：形状 · 面积
        f2 = QFont(option.font)
        f2.setPixelSize(10)
        f2.setFamily("Menlo")
        painter.setFont(f2)
        painter.setPen(QColor(PALETTE["text_subtle"]))
        shape_rect = QRect(
            name_x, rect.top() + 31,
            max(0, rect.right() - 38 - name_x - 8), 18,
        )
        painter.drawText(
            shape_rect, Qt.AlignLeft | Qt.AlignVCenter,
            self._elided_text(f2, shape, shape_rect.width()),
        )

        # 右侧状态圈
        confirmed = index.data(self.STATE_ROLE)
        cx = rect.right() - 18
        cy = rect.center().y()
        if confirmed:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(62, 207, 142, 40))
            painter.drawEllipse(QPoint(cx, cy), 9, 9)
            painter.setPen(QPen(QColor(PALETTE["success"]), 1.4))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPoint(cx, cy), 8, 8)
            painter.setPen(QColor(PALETTE["success"]))
            painter.setFont(QFont(option.font))
            painter.setFont(f2)
            painter.drawText(
                QRect(cx - 8, cy - 8, 16, 16), Qt.AlignCenter, "\u2713")
        else:
            painter.setPen(QPen(QColor(PALETTE["text_subtle"]), 1.2, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPoint(cx, cy), 7, 7)
        painter.restore()

    def sizeHint(self, option, index):
        if index.data(self.CARD_ROLE):
            return QSize(option.rect.width() if option.rect.width() > 0 else 240, 58)
        return super().sizeHint(option, index)


class AnnotationPanel(QWidget):
    """Right-side panel showing project class list, annotation tree, and properties.

    Signals:
        annotation_clicked(str): Annotation ID clicked in the tree.
        keypoint_clicked(str, int): Keypoint clicked — (ann_id, kp_index).
        keypoint_rename_requested(str, int, str): (ann_id, kp_idx, new_label).
        keypoint_visibility_requested(str, int): (ann_id, kp_idx) — cycle visibility.
        keypoint_delete_requested(str, int): (ann_id, kp_idx).
        default_class_changed(str): Class name clicked in the project class
            list — caller should treat this as the new default for drawing.
    """

    annotation_clicked = pyqtSignal(str)
    annotation_class_change_requested = pyqtSignal(str)
    annotation_confirm_requested = pyqtSignal(str, bool)
    annotation_delete_requested = pyqtSignal(str)
    clear_all_annotations_requested = pyqtSignal()
    keypoint_add_requested = pyqtSignal(str)
    keypoint_clicked = pyqtSignal(str, int)
    keypoint_rename_requested = pyqtSignal(str, int, str)
    keypoint_visibility_requested = pyqtSignal(str, int)
    keypoint_delete_requested = pyqtSignal(str, int)
    # str class name when set, None when cleared (toggled off)
    default_class_changed = pyqtSignal(object)
    # Emitted when the user edits the per-image dataset Tag chip bar.
    image_user_tags_changed = pyqtSignal(list)  # list[str]
    # 主页式检查器：数据版本行点击 → 视图切换筛选
    data_folder_selected = pyqtSignal(str)
    open_preview_requested = pyqtSignal()
    manage_data_folders_requested = pyqtSignal()
    tag_manage_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._annotations: list[Annotation] = []
        self._selected_id: str | None = None
        self._selected_kp_idx: int | None = None
        self._classes: list[str] = []
        self._class_colors: dict[str, str] = {}
        self._project_class_counts: dict[str, int] = {}
        self._default_class: str | None = None
        self._ann_list_expanded = False
        self._init_ui()

    def _init_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self._outer_layout = outer
        self._sections: dict[str, QWidget] = {}
        self._section_titles: dict[str, QLabel] = {}

        # ── 当前图片（设计稿：单富文本表格，键列 76px，无边框线） ──
        cur_box, cur_lay = self._flat_section("当前图片", "在新窗口预览",
                                              self.open_preview_requested)
        self._img_info_label = QLabel()
        self._img_info_label.setTextFormat(Qt.RichText)
        self._img_info_label.setStyleSheet(
            f"color:{PALETTE['text']};font-size:12px;border:none;"
            "background:transparent;")
        cur_lay.addWidget(self._img_info_label)
        # 兼容旧属性名
        self._img_file_label = self._img_info_label
        self._img_size_label = self._img_info_label
        self._img_status_label = self._img_info_label
        self._img_folder_label = self._img_info_label

        # ── 标注列表 ──
        ann_box, ann_layout = self._flat_section("标注列表", "全部展开",
                                                self._expand_annotation_tree)
        self._ann_tree = QTreeWidget()
        self._ann_tree.setHeaderHidden(True)
        self._ann_tree.setIndentation(12)
        self._ann_tree.setMinimumHeight(44)
        self._ann_tree.setMaximumHeight(380)
        # 卡片外观完全由 _AnnCardDelegate 绘制；branch 用不透明深底压掉选中填充
        self._ann_tree.setStyleSheet(
            "QTreeWidget{background:transparent;border:none;}"
            "QTreeWidget::item{background:transparent;border:none;}"
            "QTreeWidget::item:hover{background:transparent;}"
            "QTreeWidget::item:selected{background:transparent;border:none;}"
            f"QTreeWidget::branch{{background:{PALETTE['bg']};}}"
            f"QTreeWidget::branch:selected{{background:{PALETTE['bg']};}}"
        )
        self._ann_tree.setItemDelegate(_AnnCardDelegate(self._ann_tree))
        self._ann_tree.currentItemChanged.connect(self._on_tree_item_changed)
        self._ann_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self._ann_tree.customContextMenuRequested.connect(self._on_tree_context_menu)
        self._ann_tree.itemExpanded.connect(lambda _item: self._sync_annotation_tree_height())
        self._ann_tree.itemCollapsed.connect(lambda _item: self._sync_annotation_tree_height())
        ann_layout.addWidget(self._ann_tree)
        act_row = QHBoxLayout()
        act_row.setContentsMargins(0, 0, 0, 0)
        act_row.setSpacing(8)
        self._btn_confirm_ann = QPushButton("确认")
        self._btn_confirm_ann.setToolTip("确认当前选中的标注")
        self._btn_confirm_ann.setObjectName("annConfirmBtn")
        self._btn_confirm_ann.setFixedHeight(36)
        self._btn_confirm_ann.setStyleSheet(
            "QPushButton#annConfirmBtn{background:" + PALETTE["panel_alt"]
            + ";border:1px solid " + PALETTE["line_strong"]
            + ";border-radius:8px;color:" + PALETTE["text"]
            + ";font-weight:600;}"
            "QPushButton#annConfirmBtn:hover{border-color:"
            + PALETTE["primary"] + ";}")
        self._btn_confirm_ann.clicked.connect(self._confirm_selected_annotation)
        self._btn_delete_ann = QPushButton("删除")
        self._btn_delete_ann.setToolTip("删除当前选中的标注")
        self._btn_delete_ann.setObjectName("annDeleteBtn")
        self._btn_delete_ann.setFixedHeight(36)
        self._btn_delete_ann.setStyleSheet(
            "QPushButton#annDeleteBtn{background:" + PALETTE["panel_alt"]
            + ";border:1px solid " + PALETTE["line_strong"]
            + ";border-radius:8px;color:" + PALETTE["text"]
            + ";font-weight:600;}"
            "QPushButton#annDeleteBtn:hover{border-color:"
            + PALETTE["primary"] + ";}")
        self._btn_delete_ann.clicked.connect(self._delete_selected_annotation)
        act_row.addWidget(self._btn_confirm_ann, 1)
        act_row.addWidget(self._btn_delete_ann, 1)
        ann_layout.addLayout(act_row)
        outer.addWidget(ann_box)
        self._sections["标注列表"] = ann_box

        # ── 类别 · 点击设默认 ──
        cls_box, cls_layout = self._flat_section("类别 · 点击设默认")
        self._classes_list = QListWidget()
        self._classes_list.setToolTip(
            "点击设为下次画框/关键点的默认类别；再次点击当前类可取消"
        )
        self._classes_list.setMinimumHeight(96)
        self._classes_list.setStyleSheet(
            "QListWidget{background:transparent;border:none;}"
            "QListWidget::item{margin:0;padding:0px;}"
        )
        self._classes_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        cls_layout.addWidget(self._classes_list, 0)
        # （设计稿：去掉类别提示文字）
        outer.addWidget(cls_box)
        self._sections["类别"] = cls_box

        # ── 数据版本 ──
        ver_box, ver_layout = self._flat_section("数据版本", "管理",
                                                 self.manage_data_folders_requested)
        self._version_list = QListWidget()
        self._version_list.setMaximumHeight(132)
        self._version_list.setStyleSheet(
            "QListWidget{background:transparent;border:none;}"
            "QListWidget::item{min-height:30px;padding:2px 8px;"
            "border-radius:6px;margin:1px 0;}"
            "QListWidget::item:selected{"
            f"background:{PALETTE['primary']};color:#FFFFFF;}}"
        )
        self._version_list.itemClicked.connect(self._on_version_row_clicked)
        ver_layout.addWidget(self._version_list)
        seg_row = QHBoxLayout()
        seg_row.setContentsMargins(0, 2, 0, 2)
        seg_row.setSpacing(2)
        self._seg_ok = QLabel()
        self._seg_pending = QLabel()
        self._seg_rest = QLabel()
        for seg in (self._seg_ok, self._seg_pending, self._seg_rest):
            seg.setFixedHeight(4)
            seg.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            seg_row.addWidget(seg, 1)
        ver_layout.addLayout(seg_row)
        self._seg_legend = QLabel("—")
        self._seg_legend.setStyleSheet(text_style("hint"))
        ver_layout.addWidget(self._seg_legend)
        outer.addWidget(ver_box)
        self._sections["数据版本"] = ver_box

        # ── TAG ──
        tag_box, tag_layout = self._flat_section("TAG", "管理",
                                                 self.tag_manage_requested)
        self._tag_bar = TagChipBar()
        self._tag_bar.tags_changed.connect(self.image_user_tags_changed)
        tag_layout.addWidget(self._tag_bar)
        tag_hint = QLabel("用于按 tag 筛选数据/训练子集，与分类标签独立。")
        tag_hint.setWordWrap(True)
        tag_hint.setStyleSheet(text_style("hint"))
        tag_layout.addWidget(tag_hint)
        outer.addWidget(tag_box)
        self._sections["TAG"] = tag_box
        # Keep the historical attribute name alive in case external code reads it.
        self._tag_group = tag_box

        # ── 属性（选中标注详情；默认隐藏，接口保留） ──
        attr_box = QWidget(self)
        attr_layout = QVBoxLayout(attr_box)
        attr_layout.setContentsMargins(18, 12, 18, 14)
        attr_layout.setSpacing(4)
        self._stats_label = QLabel("")
        self._stats_label.setStyleSheet(text_style("hint"))
        attr_layout.addWidget(self._stats_label)
        self._class_label = QLabel("")
        self._conf_label = QLabel("")
        self._status_label = QLabel("")
        self._source_label = QLabel("")
        self._bbox_label = QLabel("")
        for lbl in [self._class_label, self._conf_label, self._status_label,
                    self._source_label, self._bbox_label]:
            lbl.setStyleSheet(text_style("muted"))
            attr_layout.addWidget(lbl)
        attr_layout.addStretch(1)
        self._sections_extra = {"属性": attr_box}

        # ── 项目统计（默认隐藏；信息由 数据版本图例 + 类别行 覆盖） ──
        stats_box = QWidget(self)
        stats_layout = QVBoxLayout(stats_box)
        stats_layout.setContentsMargins(18, 12, 18, 14)
        self._project_total_label = QLabel("总图片: 0")
        self._project_labeled_label = QLabel("已标注: 0")
        self._project_confirmed_label = QLabel("全确认: 0")
        self._project_ann_count_label = QLabel("总标注数: 0")
        self._class_dist_label = QLabel("类别分布:")
        self._class_dist_list = QListWidget()
        self._class_dist_list.setMaximumHeight(120)
        self._sections_extra = getattr(self, "_sections_extra", {})
        self._sections_extra["项目统计"] = stats_box

        self._sync_annotation_tree_height()

    def _flat_section(self, title: str, right: str | None = None,
                      right_signal=None):
        """设计稿扁平分区：小标题（+右侧链接）+ 内容 + 1px 底部分隔线。"""
        box = QWidget(self)
        v = QVBoxLayout(box)
        v.setContentsMargins(10, 2, 10, 2)
        v.setSpacing(1)
        head = QHBoxLayout()
        t = QLabel(title)
        t.setStyleSheet(
            f"color:{PALETTE['text_subtle']};font-size:11px;font-weight:700;"
            "letter-spacing:0.08em;border:none;background:transparent;")
        self._section_titles[title] = t
        head.addWidget(t)
        if right:
            r = QLabel(right)
            r.setStyleSheet(
                f"color:{PALETTE['text_subtle']};font-size:13px;font-weight:600;"
                "border:none;background:transparent;")
            if right_signal is not None:
                r.setCursor(Qt.PointingHandCursor)
                r.mouseReleaseEvent = lambda ev, sg=right_signal: sg.emit()
            head.addStretch(1)
            head.addWidget(r)
        v.addLayout(head)
        box.setStyleSheet(f"border-bottom:1px solid {PALETTE['line']};")
        self._outer_layout.addWidget(box)
        return box, v

    def _expand_annotation_tree(self) -> None:
        self._ann_list_expanded = True
        self._ann_tree.expandAll()
        self._sync_annotation_tree_height()

    def set_classes(self, classes: list[str]) -> None:
        """Set the project's class list (drives the project class panel)."""
        self._classes = list(classes)
        self._rebuild_classes_list()

    # ── State persistence ──────────────────────────────────────

    def _rebalance_sections(self) -> None:
        pass


    def _sync_annotation_tree_height(self) -> None:
        """Keep annotation list height tied to visible row count."""
        if not hasattr(self, "_ann_tree"):
            return
        visible_rows = 0
        for i in range(self._ann_tree.topLevelItemCount()):
            top = self._ann_tree.topLevelItem(i)
            visible_rows += 1
            if top.isExpanded():
                visible_rows += top.childCount()
        row_limit = visible_rows if self._ann_list_expanded else min(visible_rows, 3)
        rows = max(1, row_limit)
        row_h = self._ann_tree.sizeHintForRow(0)
        if row_h <= 0:
            row_h = 24
        frame = self._ann_tree.frameWidth() * 2 + 12
        self._ann_tree.setFixedHeight(rows * row_h + frame)

    def save_state(self) -> dict:
        """扁平分区无可折叠状态；保留接口兼容。"""
        return {"collapsed": {}}

    def restore_state(self, state: dict) -> None:
        """Reapply a previous snapshot. Missing or malformed keys are no-ops."""
        self._sync_annotation_tree_height()

    def set_class_colors(self, colors: dict[str, str]) -> None:
        """Set class color mapping. Refreshes the project class panel icons."""
        self._class_colors = dict(colors)
        self._rebuild_classes_list()

    def set_default_class(self, class_name: str | None) -> None:
        """Mark a class as the current drawing default; bold it in the list."""
        self._default_class = class_name
        self._refresh_default_highlight()

    def set_annotations(self, annotations: list[Annotation],
                        image_size: tuple[int, int] | None = None) -> None:
        """Update the annotation tree（设计稿卡片：名称+置信度 / 形状·面积）。"""
        self._annotations = list(annotations)
        if "标注列表" in self._section_titles:
            self._section_titles["标注列表"].setText(f"标注列表 · {len(annotations)}")
        self._ann_tree.blockSignals(True)
        self._ann_tree.clear()

        for ann in annotations:
            color = QColor(self._class_colors.get(ann.class_name, PALETTE["primary"]))
            if ann.polygon:
                type_hint = f"多边形 \u00d7{len(ann.polygon)}"
            elif ann.bbox and ann.keypoints:
                type_hint = f"矩形框+关键点 \u00d7{len(ann.keypoints)}"
            elif ann.bbox:
                type_hint = "矩形框"
            elif ann.keypoints:
                type_hint = f"关键点 \u00d7{len(ann.keypoints)}"

            top_item = QTreeWidgetItem([ann.class_name])
            top_item.setData(0, Qt.UserRole, ann.id)
            top_item.setData(0, Qt.UserRole + 1, -1)  # -1 = annotation level
            top_item.setData(0, Qt.ForegroundRole, color)  # 委托色块颜色
            area_text = ""
            if image_size:
                area_text = annotation_area_text(ann, image_size)
            shape_line = " · ".join(
                part for part in (type_hint, area_text) if part)
            top_item.setData(0, _AnnCardDelegate.CARD_ROLE, shape_line)
            top_item.setData(0, _AnnCardDelegate.CONFIDENCE_ROLE,
                             f"{ann.confidence:.2f}")
            top_item.setData(0, _AnnCardDelegate.STATE_ROLE, ann.confirmed)
            self._ann_tree.addTopLevelItem(top_item)
            # 关键点子行不再入树（设计稿）：卡片第二行显示“关键点 ×N”，
            # 重命名/可见性/删除等操作经画布右键菜单完成

            if ann.keypoints:
                top_item.setExpanded(True)

        self._ann_tree.blockSignals(False)
        self._sync_annotation_tree_height()
        self._update_stats()
        self._refresh_class_counts()

    def select_annotation(self, ann_id: str | None) -> None:
        """Select an annotation in the tree and show its properties."""
        self._selected_id = ann_id
        self._selected_kp_idx = None
        if ann_id is None:
            self._ann_tree.clearSelection()
            self._clear_properties()
            return

        for i in range(self._ann_tree.topLevelItemCount()):
            item = self._ann_tree.topLevelItem(i)
            if item.data(0, Qt.UserRole) == ann_id:
                self._ann_tree.blockSignals(True)
                self._ann_tree.setCurrentItem(item)
                self._ann_tree.blockSignals(False)
                break

        ann = self._find_annotation(ann_id)
        if ann:
            self._show_properties(ann)

    def select_keypoint(self, ann_id: str, kp_idx: int) -> None:
        """Select a specific keypoint in the tree."""
        self._selected_id = ann_id
        self._selected_kp_idx = kp_idx

        for i in range(self._ann_tree.topLevelItemCount()):
            top = self._ann_tree.topLevelItem(i)
            if top.data(0, Qt.UserRole) == ann_id:
                top.setExpanded(True)
                if 0 <= kp_idx < top.childCount():
                    self._ann_tree.blockSignals(True)
                    self._ann_tree.setCurrentItem(top.child(kp_idx))
                    self._ann_tree.blockSignals(False)
                break

        ann = self._find_annotation(ann_id)
        if ann and 0 <= kp_idx < len(ann.keypoints):
            self._show_keypoint_properties(ann, kp_idx)

    def set_data_folders(self, rows: list[tuple[str, str]], active: str) -> None:
        """rows: [(key, 显示名)]；active 为当前筛选 key（""=全部图片）。"""
        self._version_list.blockSignals(True)
        self._version_list.clear()
        current_item = None
        for key, display in rows:
            item = QListWidgetItem(display)
            item.setData(Qt.UserRole, key)
            self._version_list.addItem(item)
            if key == active:
                self._version_list.setCurrentItem(item)
                current_item = item
        self._version_list.blockSignals(False)
        if current_item is not None:
            self._version_list.scrollToItem(current_item)

    def _on_version_row_clicked(self, item) -> None:
        self._version_list.setCurrentItem(item)  # 点击即选中（高亮）
        key = item.data(Qt.UserRole)
        if key is not None:
            self.data_folder_selected.emit(str(key))

    def _seg_layout(self):
        return self._seg_ok.parentWidget().layout()

    def _update_version_bar(self, confirmed: int, pending: int,
                            unlabeled: int) -> None:
        """按“标注个数”显示：绿=已确认标注，黄=待确认标注，灰=未标注图片。"""
        total = max(1, confirmed + pending + unlabeled)
        self._seg_ok.setStyleSheet(f"background:{PALETTE['success']};border-radius:2px;")
        self._seg_pending.setStyleSheet(f"background:{PALETTE['warning']};border-radius:2px;")
        self._seg_rest.setStyleSheet(f"background:{PALETTE['line']};border-radius:2px;")
        self._seg_layout().setStretch(0, confirmed)
        self._seg_layout().setStretch(1, pending)
        self._seg_layout().setStretch(2, unlabeled if unlabeled else
                                      (0 if (confirmed or pending) else 1))
        self._seg_legend.setText(
            f"<span style='color:{PALETTE['success']}'>●</span> 已确认 {confirmed}"
            f"　<span style='color:{PALETTE['warning']}'>●</span> 待确认 {pending}"
            f"　<span style='color:{PALETTE['text_subtle']}'>●</span> 未标注 {unlabeled}"
        )

    def set_current_image_info(self, file_name: str, size: str, status: str,
                               folder: str) -> None:
        import html as _html
        styles = {
            "已确认": ("rgba(62,207,142,0.12)", PALETTE["success"]),
            "待确认": ("rgba(245,184,61,0.12)", PALETTE["warning"]),
            "未标注": ("rgba(93,110,136,0.15)", PALETTE["text_muted"]),
        }
        bg, fg = styles.get(status, ("rgba(93,110,136,0.15)", PALETTE["text_muted"]))
        key_style = f"color:{PALETTE['text_subtle']};font-size:12px;"
        val_style = ("font-family:Menlo,'SF Mono',monospace;font-size:11.5px;"
                     f"color:{PALETTE['text']};")
        plain = f"color:{PALETTE['text']};font-size:12px;"
        pill = (f"background:{bg};color:{fg};border-radius:5px;"
                "padding:1px 8px;font-size:11px;")
        esc = _html.escape
        self._img_info_label.setText(
            "<table cellspacing='7' cellpadding='0'>"
            f"<tr><td style='width:76px;{key_style}'>文件</td>"
            f"<td style='{val_style}'>{esc(file_name or '—')}</td></tr>"
            f"<tr><td style='{key_style}'>尺寸</td>"
            f"<td style='{val_style}'>{esc(size or '—')}</td></tr>"
            f"<tr><td style='{key_style}'>状态</td>"
            f"<td><span style='{pill}'>● {esc(status or '—')}</span></td></tr>"
            f"<tr><td style='{key_style}'>数据版本</td>"
            f"<td style='{plain}'>{esc(folder or '全部图片')}</td></tr>"
            "</table>")

    def _current_ann_id(self):
        item = self._ann_tree.currentItem()
        return item.data(0, Qt.UserRole) if item is not None else None

    def _confirm_selected_annotation(self) -> None:
        ann_id = self._current_ann_id()
        ann = self._find_annotation(ann_id) if ann_id else None
        if ann is not None:
            self.annotation_confirm_requested.emit(ann_id, not ann.confirmed)

    def _delete_selected_annotation(self) -> None:
        ann_id = self._current_ann_id()
        if ann_id:
            self.annotation_delete_requested.emit(ann_id)

    def set_project_stats(self, stats: dict) -> None:
        """Update project-level statistics."""
        self._project_total_label.setText(f"总图片: {stats.get('total_images', 0)}")
        self._project_labeled_label.setText(f"已标注: {stats.get('labeled_images', 0)}")
        self._project_confirmed_label.setText(f"全确认: {stats.get('confirmed_images', 0)}")
        self._project_ann_count_label.setText(f"总标注数: {stats.get('total_annotations', 0)}")
        confirmed_anns = int(stats.get("confirmed_annotations", 0) or 0)
        pending_anns = int(stats.get("pending_annotations", 0) or 0)
        unlabeled_imgs = (int(stats.get("total_images", 0) or 0)
                          - int(stats.get("labeled_images", 0) or 0))
        self._update_version_bar(confirmed_anns, pending_anns, max(0, unlabeled_imgs))

        class_counts = stats.get("class_counts", {})
        self._project_class_counts = {
            str(name): int(count or 0) for name, count in class_counts.items()
        }
        self._refresh_class_counts()
        self._class_dist_list.clear()
        for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            color = self._class_colors.get(cls_name, PALETTE["primary"])
            item = QListWidgetItem(f"{cls_name}: {count}")
            item.setForeground(QColor(color))
            self._class_dist_list.addItem(item)

    def clear(self) -> None:
        """Clear all state."""
        self._annotations = []
        self._selected_id = None
        self._selected_kp_idx = None
        self._ann_tree.clear()
        self._sync_annotation_tree_height()
        self._refresh_class_counts()
        self._clear_properties()
        self._stats_label.setText("")
        self._class_dist_list.clear()
        self._tag_bar.set_tags([])

    # ── Per-image user tags (dataset Tag) ─────────────────────

    def set_available_tags(self, tags: list[str]) -> None:
        """Set the project's known-tag registry (populates the chip popup)."""
        self._tag_bar.set_available_tags(tags)

    def set_image_user_tags(self, tags: list[str]) -> None:
        """Populate the per-image chip bar with the loaded image's tags."""
        # Block our own signal so loading a new image doesn't look like an edit.
        self._tag_bar.blockSignals(True)
        try:
            self._tag_bar.set_tags(tags)
        finally:
            self._tag_bar.blockSignals(False)

    def get_image_user_tags(self) -> list[str]:
        return self._tag_bar.get_tags()

    def _show_properties(self, ann: Annotation) -> None:
        self._class_label.setText(f"类别: {ann.class_name}")
        self._conf_label.setText(f"置信度: {ann.confidence:.2f}")
        self._status_label.setText(f"状态: {'已确认' if ann.confirmed else '待确认'}")
        self._source_label.setText(f"来源: {'手动' if ann.source == 'manual' else '自动'}")
        if ann.bbox:
            cx, cy, w, h = ann.bbox
            self._bbox_label.setText(f"Bbox: ({cx:.3f}, {cy:.3f}, {w:.3f}, {h:.3f})")
        elif ann.polygon:
            xs = [p[0] for p in ann.polygon]
            ys = [p[1] for p in ann.polygon]
            self._bbox_label.setText(
                f"Polygon: {len(ann.polygon)} points | "
                f"bounds ({min(xs):.3f}, {min(ys):.3f})-({max(xs):.3f}, {max(ys):.3f})"
            )
        elif ann.keypoints:
            self._bbox_label.setText(f"关键点: {len(ann.keypoints)} 个")
        else:
            self._bbox_label.setText("")

    def _show_keypoint_properties(self, ann: Annotation, kp_idx: int) -> None:
        kp = ann.keypoints[kp_idx]
        vis_names = ["不可见", "被遮挡", "可见"]
        vis = vis_names[kp.visible] if kp.visible < 3 else "?"
        self._class_label.setText(f"关键点: {kp.label}")
        self._conf_label.setText(f"所属: {ann.class_name}")
        self._status_label.setText(f"可见性: {vis}")
        self._source_label.setText(f"坐标: ({kp.x:.4f}, {kp.y:.4f})")
        self._bbox_label.setText(f"索引: {kp_idx}/{len(ann.keypoints)}")

    def _clear_properties(self) -> None:
        self._class_label.setText("")
        self._conf_label.setText("")
        self._status_label.setText("")
        self._source_label.setText("")
        self._bbox_label.setText("")

    def _update_stats(self) -> None:
        total = len(self._annotations)
        confirmed = sum(1 for a in self._annotations if a.confirmed)
        pending = total - confirmed
        self._stats_label.setText(f"标注: {total} | 确认: {confirmed} | 待确认: {pending}")

    def _find_annotation(self, ann_id: str) -> Annotation | None:
        for ann in self._annotations:
            if ann.id == ann_id:
                return ann
        return None

    def _on_tree_item_changed(self, current, previous) -> None:
        if current is None:
            return
        ann_id = current.data(0, Qt.UserRole)
        kp_idx = current.data(0, Qt.UserRole + 1)
        if ann_id is None:
            return
        if kp_idx is not None and kp_idx >= 0:
            self._selected_kp_idx = kp_idx
            self.keypoint_clicked.emit(ann_id, kp_idx)
            ann = self._find_annotation(ann_id)
            if ann:
                self._show_keypoint_properties(ann, kp_idx)
        else:
            self._selected_kp_idx = None
            self.annotation_clicked.emit(ann_id)
            ann = self._find_annotation(ann_id)
            if ann:
                self._show_properties(ann)

    def _on_tree_context_menu(self, pos) -> None:
        item = self._ann_tree.itemAt(pos)
        if not item:
            return
        ann_id = item.data(0, Qt.UserRole)
        kp_idx = item.data(0, Qt.UserRole + 1)
        if ann_id is None:
            return

        ann = self._find_annotation(ann_id)
        if not ann:
            return

        if kp_idx is not None and kp_idx >= 0 and kp_idx < len(ann.keypoints):
            kp = ann.keypoints[kp_idx]
            menu = QMenu(self)

            add_kp = menu.addAction("新增关键点")
            add_kp.triggered.connect(
                lambda _, aid=ann_id: self.keypoint_add_requested.emit(aid))

            rename = menu.addAction(f"重命名 ({kp.label})")
            rename.triggered.connect(
                lambda _, aid=ann_id, ki=kp_idx, old=kp.label: self._rename_keypoint(aid, ki, old))

            vis_names = ["不可见", "被遮挡", "可见"]
            vis = vis_names[kp.visible] if kp.visible < 3 else "?"
            toggle_vis = menu.addAction(f"切换可见性 ({vis})")
            toggle_vis.triggered.connect(
                lambda _, aid=ann_id, ki=kp_idx: self.keypoint_visibility_requested.emit(aid, ki))

            delete = menu.addAction("删除关键点")
            delete.triggered.connect(
                lambda _, aid=ann_id, ki=kp_idx: self.keypoint_delete_requested.emit(aid, ki))

            menu.exec_(self._ann_tree.viewport().mapToGlobal(pos))
            return

        menu = QMenu(self)

        add_kp = menu.addAction("新增关键点")
        add_kp.triggered.connect(
            lambda _, aid=ann_id: self.keypoint_add_requested.emit(aid))

        change_cls = menu.addAction(f"修改类别 ({ann.class_name})")
        change_cls.triggered.connect(
            lambda _, aid=ann_id: self.annotation_class_change_requested.emit(aid))

        confirm_text = "取消确认" if ann.confirmed else "确认标注"
        confirm = menu.addAction(confirm_text)
        confirm.triggered.connect(
            lambda _, aid=ann_id, state=not ann.confirmed:
                self.annotation_confirm_requested.emit(aid, state))

        delete = menu.addAction("删除标注")
        delete.triggered.connect(
            lambda _, aid=ann_id: self.annotation_delete_requested.emit(aid))

        menu.exec_(self._ann_tree.viewport().mapToGlobal(pos))

    def _rename_keypoint(self, ann_id: str, kp_idx: int, old_label: str) -> None:
        new_label, ok = QInputDialog.getText(self, "重命名关键点", "标签:", text=old_label)
        if ok and new_label.strip():
            self.keypoint_rename_requested.emit(ann_id, kp_idx, new_label.strip())

    # ── Project class panel helpers ────────────────────────────

    @staticmethod
    def _swatch_style(color_hex: str) -> str:
        """Return QSS for a small color swatch that visually echoes the bbox
        stroke color rendered on canvas (same hex source)."""
        return f"background-color:{color_hex};border:none;border-radius:3px;"

    def _make_class_row(self, idx: int, cls_name: str, color: str) -> QWidget:
        """Build the per-row widget: swatch + name (left) + count (right).

        The row owns its own `clicked(str)` signal — see _ClassRow.
        """
        row = _ClassRow(cls_name)
        row.clicked.connect(self._on_class_clicked)

        hl = QHBoxLayout(row)
        hl.setContentsMargins(8, 3, 8, 3)
        hl.setSpacing(8)

        swatch = QLabel()
        swatch.setFixedSize(14, 14)
        swatch.setStyleSheet(self._swatch_style(color))
        swatch.setObjectName("swatch")

        row.setStyleSheet(
            "QWidget{background:transparent;border:none;border-radius:0;}"
            "QWidget QLabel{background:transparent;border:none;}"
        )
        name_lbl = QLabel(cls_name)
        name_lbl.setMinimumWidth(0)
        name_lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        f = name_lbl.font()
        f.setUnderline(False)
        name_lbl.setFont(f)
        name_lbl.setObjectName("name_lbl")

        count_lbl = QLabel("0")
        count_lbl.setStyleSheet(
            f"color:{PALETTE['text_subtle']};font-family:Menlo,'SF Mono',monospace;"
        )
        count_lbl.setObjectName("count_lbl")

        hl.addWidget(swatch)
        hl.addWidget(name_lbl, 1)
        hl.addWidget(count_lbl)
        return row

    def _rebuild_classes_list(self) -> None:
        """Populate the project class panel from `_classes` + `_class_colors`."""
        self._classes_list.blockSignals(True)
        self._classes_list.clear()
        for idx, cls_name in enumerate(self._classes):
            color = self._class_colors.get(cls_name, PALETTE["primary"])
            item = QListWidgetItem()
            item.setData(Qt.UserRole, cls_name)
            self._classes_list.addItem(item)
            row = self._make_class_row(idx, cls_name, color)
            hint = row.sizeHint()
            hint.setHeight(max(20, hint.height() + 2))
            # The inspector is a 292px-wide fixed rail; a bounded item width
            # keeps long class names from creating a horizontal scrollbar
            # while leaving the label its share of the row.
            item.setSizeHint(QSize(240, hint.height()))
            self._classes_list.setItemWidget(item, row)
        self._classes_list.blockSignals(False)
        # 最多展示 4 个类别，多于 4 个时右侧出现滚动条
        row_h = self._classes_list.sizeHintForRow(0)
        if row_h <= 0:
            row_h = 32
        self._classes_list.setFixedHeight(min(len(self._classes), 4) * row_h)
        self._classes_list.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._refresh_class_counts()
        self._refresh_default_highlight()

    def _class_count_text(self, cls_name: str) -> str:
        """Project-wide class count, matching the design inspector."""
        if cls_name in self._project_class_counts:
            return f"{self._project_class_counts[cls_name]:,}"
        ann_count = 0
        kp_count = 0
        for a in self._annotations:
            if a.class_name != cls_name:
                continue
            ann_count += 1
            kp_count += len(a.keypoints)
        text = str(ann_count)
        if kp_count > 0:
            text += f"  ({kp_count} kp)"
        return text

    def _refresh_class_counts(self) -> None:
        """Update per-class count labels without rebuilding rows."""
        if self._classes_list.count() != len(self._classes):
            self._rebuild_classes_list()
            return
        for idx, cls_name in enumerate(self._classes):
            item = self._classes_list.item(idx)
            row = self._classes_list.itemWidget(item)
            if row is None:
                continue
            count_lbl = row.findChild(QLabel, "count_lbl")
            if count_lbl is not None:
                count_lbl.setText(self._class_count_text(cls_name))

    def _refresh_default_highlight(self) -> None:
        """Bold the name label of the row matching `_default_class`."""
        for i in range(self._classes_list.count()):
            item = self._classes_list.item(i)
            row = self._classes_list.itemWidget(item)
            if row is None:
                continue
            name_lbl = row.findChild(QLabel, "name_lbl")
            if name_lbl is None:
                continue
            is_default = item.data(Qt.UserRole) == self._default_class
            font = name_lbl.font()
            font.setBold(is_default)
            name_lbl.setFont(font)

    def _on_class_clicked(self, cls_name: str) -> None:
        if not cls_name:
            return
        if cls_name == self._default_class:
            # Toggle off: clear default
            self._default_class = None
            self._refresh_default_highlight()
            self.default_class_changed.emit(None)
        else:
            self._default_class = cls_name
            self._refresh_default_highlight()
            self.default_class_changed.emit(cls_name)
