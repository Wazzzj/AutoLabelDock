"""Model management panel — model list, load/switch, auto-label settings."""
from __future__ import annotations

import csv
import logging
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QDoubleSpinBox,
    QComboBox,
    QAbstractItemView,
    QFileDialog,
    QSpinBox,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, pyqtSignal, QUrl
from PyQt5.QtGui import QColor, QDesktopServices, QImage, QPixmap

from src.engine.model_manager import ModelInfo
from src.core.project import IMAGE_EXTENSIONS
from src.core.annotation import Annotation
from src.ui.canvas import AnnotationCanvas
from src.ui.icons import icon
from src.ui.loading import LoadingRow
from src.ui.theme import PALETTE, set_button_role, text_style

logger = logging.getLogger(__name__)

_DEVICE_OPTIONS = {
    "AUTO": "",
    "CUDA:0": "0",
    "CUDA:1": "1",
    "CPU": "cpu",
}
_DEVICE_VALUE_TO_LABEL = {value: label for label, value in _DEVICE_OPTIONS.items()}


_BASE_PARAM_GROUPS: list[tuple[str, list[str]]] = [
    ("基础", ["epochs", "batch", "imgsz", "device", "freeze", "workers", "patience"]),
    ("优化器", ["optimizer", "lr0", "lrf", "momentum", "weight_decay",
              "warmup_epochs", "warmup_momentum", "warmup_bias_lr"]),
    ("数据增强（通用）", ["hsv_h", "hsv_s", "hsv_v", "scale", "flipud", "fliplr"]),
]

_DETECT_AUG_KEYS = ["degrees", "translate", "shear", "perspective",
                    "mosaic", "mixup"]
_SEGMENT_KEYS = ["mask_ratio", "overlap_mask", "copy_paste", "copy_paste_mode"]
_CLASSIFY_AUG_KEYS = ["erasing", "auto_augment", "dropout"]
_POSE_LOSS_KEYS = ["pose", "kobj"]


def _format_param_value(key: str, value) -> str:
    if value is None:
        return f"{key}=默认"
    if isinstance(value, float):
        return f"{key}={value:g}"
    return f"{key}={value}"


def _format_train_params(params: dict, task: str) -> str:
    """Render train_params dict as grouped rich text. Returns '无' if empty."""
    if not params:
        return "无"

    sections: list[tuple[str, list[str]]] = []

    for group_name, keys in _BASE_PARAM_GROUPS:
        items = [_format_param_value(k, params[k]) for k in keys if k in params]
        if items:
            sections.append((group_name, items))

    if task in {"detect", "segment"} and params.get("include_detect_params"):
        items = [_format_param_value(k, params[k]) for k in _DETECT_AUG_KEYS if k in params]
        if items:
            sections.append(("Detect 增强", items))
    if task == "segment":
        items = [_format_param_value(k, params[k]) for k in _SEGMENT_KEYS if k in params]
        if items:
            sections.append(("分割参数", items))
    elif task == "classify" and params.get("include_classify_params"):
        items = [_format_param_value(k, params[k]) for k in _CLASSIFY_AUG_KEYS if k in params]
        if items:
            sections.append(("Classify 增强", items))
    elif task == "pose":
        pose_items: list[str] = []
        kpt_shape = params.get("kpt_shape")
        if kpt_shape is not None:
            pose_items.append(f"kpt_shape={kpt_shape}")
        if params.get("include_pose_params"):
            pose_items.extend(
                _format_param_value(k, params[k]) for k in _POSE_LOSS_KEYS if k in params
            )
        if pose_items:
            sections.append(("Pose 参数", pose_items))

    if not sections:
        return "无"
    return "<br>".join(
        f"<b>[{name}]</b> " + ", ".join(items) for name, items in sections
    )


class ModelPanel(QWidget):
    """Model management panel.

    Signals:
        model_load_requested(str): Request to load model by ID.
        model_delete_requested(str): Request to delete model by ID.
    """

    model_load_requested = pyqtSignal(str)
    model_delete_requested = pyqtSignal(str)
    model_rename_requested = pyqtSignal(str)
    model_import_requested = pyqtSignal()  # Request to import external .pt file
    model_predict_requested = pyqtSignal(str)  # Request inference on an arbitrary image path
    model_export_pt_requested = pyqtSignal(str)  # Request PT export by model ID
    model_export_onnx_requested = pyqtSignal(str)  # Request ONNX export by model ID

    def __init__(self, parent=None):
        super().__init__(parent)
        self._models: list[ModelInfo] = []
        self._project_dir: Path | None = None
        self._predict_image_path: Path | None = None
        self._predict_source_path: Path | None = None
        self._predict_source_is_dir = False
        self._predict_image_paths: list[Path] = []
        self._predict_image_index: int = -1
        self._last_predict_result_dir: Path | None = None
        self._selected_train_dir: Path | None = None
        self._init_ui()
        self._connect_signals()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        body = QHBoxLayout()
        body.setSpacing(12)
        left_col = QWidget()
        left_layout = QVBoxLayout(left_col)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        right_col = QWidget()
        right_layout = QVBoxLayout(right_col)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # Model list
        list_group = QGroupBox("已注册模型")
        list_layout = QVBoxLayout(list_group)

        self._model_list = QListWidget()
        self._model_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        list_layout.addWidget(self._model_list)

        btn_layout = QHBoxLayout()
        self._btn_load = QPushButton(icon("load_model"), "加载模型")
        set_button_role(self._btn_load, "primary")
        self._btn_load.setToolTip("加载选中模型用于自动标注")
        self._btn_delete = QPushButton(icon("delete"), "删除模型")
        set_button_role(self._btn_delete, "danger")
        self._btn_delete.setToolTip("从注册表中删除选中模型")
        self._btn_rename = QPushButton("重命名")
        set_button_role(self._btn_rename, "secondary")
        self._btn_rename.setToolTip("修改选中模型的显示名称")
        self._btn_import = QPushButton(icon("import"), "导入模型")
        set_button_role(self._btn_import, "secondary")
        self._btn_import.setToolTip("从文件导入外部模型（.pt、.onnx 等）")
        self._btn_export_pt = QPushButton(icon("export"), "导出PT")
        set_button_role(self._btn_export_pt, "secondary")
        self._btn_export_pt.setToolTip("将选中的 .pt 模型复制导出到指定位置")
        self._btn_export_onnx = QPushButton(icon("export"), "导出ONNX")
        set_button_role(self._btn_export_onnx, "secondary")
        self._btn_export_onnx.setToolTip("将选中的 .pt 模型转换为 ONNX 并保存到指定位置")
        btn_layout.addWidget(self._btn_load)
        btn_layout.addWidget(self._btn_delete)
        btn_layout.addWidget(self._btn_rename)
        btn_layout.addWidget(self._btn_import)
        btn_layout.addWidget(self._btn_export_pt)
        btn_layout.addWidget(self._btn_export_onnx)
        list_layout.addLayout(btn_layout)

        # Model details
        self._detail_group = QGroupBox("模型详情")
        detail_layout = QFormLayout(self._detail_group)
        self._detail_name = QLabel("")
        self._detail_task = QLabel("")
        self._detail_base = QLabel("")
        self._detail_classes = QLabel("")
        self._detail_metrics = QLabel("")
        self._detail_trained = QLabel("")
        self._detail_epochs = QLabel("")
        self._detail_dataset = QLabel("")
        self._detail_backend = QLabel("")
        self._detail_train_params = QLabel("")
        self._detail_train_params.setTextFormat(Qt.RichText)
        self._detail_artifacts = QLabel("")
        self._detail_artifacts.setTextFormat(Qt.RichText)
        self._btn_open_train_dir = QPushButton(icon("open_project"), "打开训练目录")
        self._btn_open_train_dir.setToolTip("打开该模型对应的 Ultralytics 训练输出目录")
        set_button_role(self._btn_open_train_dir, "secondary")
        self._btn_open_train_dir.setEnabled(False)

        for label_name, widget in [
            ("名称:", self._detail_name),
            ("任务:", self._detail_task),
            ("基础模型:", self._detail_base),
            ("类别:", self._detail_classes),
            ("指标:", self._detail_metrics),
            ("训练时间:", self._detail_trained),
            ("Epochs:", self._detail_epochs),
            ("数据集:", self._detail_dataset),
            ("Backend:", self._detail_backend),
            ("训练参数:", self._detail_train_params),
            ("训练产物:", self._detail_artifacts),
        ]:
            widget.setStyleSheet(text_style("body"))
            widget.setWordWrap(True)
            detail_layout.addRow(label_name, widget)
        detail_layout.addRow("", self._btn_open_train_dir)

        right_layout.addWidget(self._detail_group, 1)

        # Auto-label settings
        auto_group = QGroupBox("自动标注设置")
        auto_form = QFormLayout(auto_group)

        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.01, 1.0)
        self._conf_spin.setDecimals(2)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setValue(0.5)
        auto_form.addRow("置信度阈值:", self._conf_spin)

        self._iou_spin = QDoubleSpinBox()
        self._iou_spin.setRange(0.01, 1.0)
        self._iou_spin.setDecimals(2)
        self._iou_spin.setSingleStep(0.05)
        self._iou_spin.setValue(0.45)
        auto_form.addRow("IoU 阈值:", self._iou_spin)

        self._overlap_iou_spin = QDoubleSpinBox()
        self._overlap_iou_spin.setRange(0.01, 1.0)
        self._overlap_iou_spin.setDecimals(2)
        self._overlap_iou_spin.setSingleStep(0.05)
        self._overlap_iou_spin.setValue(0.5)
        self._overlap_iou_spin.setToolTip("预测框与已确认框重叠超过此阈值时触发冲突二选一")
        auto_form.addRow("重叠 IoU 阈值:", self._overlap_iou_spin)

        self._class_match_mode_combo = QComboBox()
        self._class_match_mode_combo.addItem("按 class_id", "class_id")
        self._class_match_mode_combo.addItem("按 class_name", "class_name")
        self._class_match_mode_combo.setToolTip("控制自动标注时模型类别与项目类别的匹配方式")
        auto_form.addRow("类别匹配方式:", self._class_match_mode_combo)

        left_layout.addWidget(auto_group)

        # Current model indicator
        self._current_label = QLabel("当前模型: 无")
        self._current_label.setStyleSheet(text_style("warning"))
        right_layout.addWidget(self._current_label)
        self._busy_row = LoadingRow()
        right_layout.addWidget(self._busy_row)

        infer_group = QGroupBox("模型推理")
        infer_layout = QVBoxLayout(infer_group)
        infer_layout.setSpacing(8)


        self._predict_canvas = AnnotationCanvas()
        self._predict_canvas.setMinimumHeight(360)
        self._predict_canvas.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
        infer_layout.addWidget(self._predict_canvas, 1)


        self._predict_image_label = QLabel("未选择图片")
        self._predict_image_label.setStyleSheet(text_style("hint"))
        self._predict_image_label.setWordWrap(True)
        infer_layout.addWidget(self._predict_image_label)

        self._predict_result_label = QLabel("预测结果将在这里显示")
        self._predict_result_label.setStyleSheet(text_style("muted"))
        self._predict_result_label.setWordWrap(True)
        infer_layout.addWidget(self._predict_result_label)


        infer_form = QFormLayout()
        infer_form.setContentsMargins(0, 0, 0, 0)
        infer_form.setHorizontalSpacing(8)
        infer_form.setVerticalSpacing(6)

        self._predict_imgsz_spin = QSpinBox()
        self._predict_imgsz_spin.setRange(32, 4096)
        self._predict_imgsz_spin.setSingleStep(32)
        self._predict_imgsz_spin.setValue(640)
        self._predict_imgsz_spin.setToolTip("推理输入尺寸；例如训练/手写代码使用 768 时这里也设为 768")
        infer_form.addRow("imgsz:", self._predict_imgsz_spin)

        self._predict_device_combo = QComboBox()
        self._predict_device_combo.addItems(list(_DEVICE_OPTIONS.keys()))
        self._predict_device_combo.setToolTip("AUTO=自动选择；CUDA:0/1=指定显卡；CPU=强制使用 CPU")
        infer_form.addRow("device:", self._predict_device_combo)

        infer_layout.addLayout(infer_form)


        nav_btns = QHBoxLayout()
        nav_btns.setContentsMargins(0, 0, 0, 0)
        nav_btns.setSpacing(8)

        self._btn_select_predict_image = QPushButton(icon("open_project"), "选择图片")
        set_button_role(self._btn_select_predict_image, "secondary")
        self._btn_select_predict_image.setToolTip("选择一张任意图片用于查看当前模型预测效果")
        self._btn_select_predict_image.setMinimumHeight(38)

        self._btn_select_predict_dir = QPushButton(icon("open_project"), "选择目录")
        set_button_role(self._btn_select_predict_dir, "secondary")
        self._btn_select_predict_dir.setToolTip("选择一个图片目录，开始推理时会推理整个目录")
        self._btn_select_predict_dir.setMinimumHeight(38)

        self._btn_prev_predict_image = QPushButton("上一张")
        set_button_role(self._btn_prev_predict_image, "secondary")
        self._btn_prev_predict_image.setToolTip("切换到目录中的上一张图片")
        self._btn_prev_predict_image.setMinimumHeight(38)
        self._btn_prev_predict_image.setEnabled(False)

        self._btn_next_predict_image = QPushButton("下一张")
        set_button_role(self._btn_next_predict_image, "secondary")
        self._btn_next_predict_image.setToolTip("切换到目录中的下一张图片")
        self._btn_next_predict_image.setMinimumHeight(38)
        self._btn_next_predict_image.setEnabled(False)

        self._btn_predict_image = QPushButton(icon("auto_label"), "开始推理")
        set_button_role(self._btn_predict_image, "primary")
        self._btn_predict_image.setToolTip("使用当前加载模型对所选图片进行推理")
        self._btn_predict_image.setMinimumHeight(38)

        self._btn_open_predict_result_dir = QPushButton(icon("open_project"), "打开结果目录")
        set_button_role(self._btn_open_predict_result_dir, "secondary")
        self._btn_open_predict_result_dir.setToolTip("打开最近一次目录推理保存的结果目录")
        self._btn_open_predict_result_dir.setMinimumHeight(38)
        self._btn_open_predict_result_dir.setEnabled(False)

        nav_btns.addWidget(self._btn_prev_predict_image, 1)
        nav_btns.addWidget(self._btn_next_predict_image, 1)
        infer_layout.addLayout(nav_btns)

        infer_action_btns = QHBoxLayout()
        infer_action_btns.setContentsMargins(0, 0, 0, 0)
        infer_action_btns.setSpacing(8)
        infer_action_btns.addWidget(self._btn_select_predict_image, 1)
        infer_action_btns.addWidget(self._btn_select_predict_dir, 1)
        infer_action_btns.addWidget(self._btn_predict_image, 1)
        infer_action_btns.addWidget(self._btn_open_predict_result_dir, 1)
        infer_layout.addLayout(infer_action_btns)
        left_layout.insertWidget(0, infer_group, 2)
        right_layout.addWidget(list_group, 2)

        left_layout.addStretch()
        right_layout.addStretch()
        body.addWidget(left_col, 2)
        body.addWidget(right_col, 1)
        layout.addLayout(body, 1)

    def _connect_signals(self) -> None:
        self._model_list.currentRowChanged.connect(self._on_model_selected)
        self._btn_load.clicked.connect(self._on_load)
        self._btn_delete.clicked.connect(self._on_delete)
        self._btn_rename.clicked.connect(self._on_rename)
        self._btn_import.clicked.connect(lambda: self.model_import_requested.emit())
        self._btn_export_pt.clicked.connect(self._on_export_pt)
        self._btn_export_onnx.clicked.connect(self._on_export_onnx)
        self._btn_open_train_dir.clicked.connect(self._on_open_train_dir)
        self._btn_select_predict_image.clicked.connect(self._on_select_predict_image)
        self._btn_select_predict_dir.clicked.connect(self._on_select_predict_dir)
        self._btn_prev_predict_image.clicked.connect(lambda: self._step_predict_image(-1))
        self._btn_next_predict_image.clicked.connect(lambda: self._step_predict_image(1))
        self._btn_predict_image.clicked.connect(self._on_predict_image)
        self._btn_open_predict_result_dir.clicked.connect(self._on_open_predict_result_dir)

    def set_project_dir(self, project_dir: str | Path | None) -> None:
        self._project_dir = Path(project_dir) if project_dir else None

    def set_models(self, models: list[ModelInfo]) -> None:
        """Update the model list."""
        self._models = list(models)
        self._model_list.blockSignals(True)
        self._model_list.clear()

        task_colors = {
            "detect": PALETTE["primary"],
            "segment": PALETTE["teal"],
            "classify": PALETTE["success"],
            "pose": PALETTE["violet"],
        }
        for model in models:
            color = task_colors.get(model.task, PALETTE["text"])
            item = QListWidgetItem(f"[{model.task}] {model.name}")
            item.setData(Qt.UserRole, model.id)
            item.setForeground(QColor(color))
            self._model_list.addItem(item)

        self._model_list.blockSignals(False)
        logger.info("Model list updated: %d models", len(models))

    def set_current_model_name(self, name: str) -> None:
        """Display the currently loaded model name."""
        self._current_label.setText(f"当前模型: {name}")

    def set_predict_imgsz(self, imgsz: int) -> None:
        """Update the inference image-size control."""
        value = int(imgsz)
        value = max(self._predict_imgsz_spin.minimum(), value)
        value = min(self._predict_imgsz_spin.maximum(), value)
        self._predict_imgsz_spin.setValue(value)

    def set_prediction_class_colors(self, colors: dict[str, str]) -> None:
        self._predict_canvas.set_class_colors(colors)

    def set_busy(self, busy: bool, text: str = "处理中…") -> None:
        if busy:
            self._busy_row.start(text)
            self._predict_canvas.set_loading(True)
        else:
            self._busy_row.stop()
            self._predict_canvas.set_loading(False)

    def set_prediction_preview(
        self,
        image_path: str | Path,
        annotations: list[Annotation],
        summary: str,
    ) -> None:
        """Render arbitrary-image inference result in the model page preview."""
        path = Path(image_path)
        self._predict_image_path = path
        self._update_predict_image_label()
        self._predict_canvas.load_image(str(path))
        self._predict_canvas.set_annotations(list(annotations))
        self._predict_result_label.setText(summary)

    def set_prediction_native_image(
        self,
        image_path: str | Path,
        image: QImage,
        summary: str,
    ) -> None:
        """Render an already-plotted native YOLO prediction image."""
        path = Path(image_path)
        self._predict_image_path = path
        self._update_predict_image_label()
        self._predict_canvas.set_pixmap(QPixmap.fromImage(image))
        self._predict_canvas.set_annotations([])
        self._predict_result_label.setText(summary)

    def set_prediction_result_dir(
        self,
        result_dir: str | Path,
        summary: str,
        preview_image_path: str | Path | None = None,
    ) -> None:
        path = Path(result_dir)
        self._last_predict_result_dir = path
        self._btn_open_predict_result_dir.setEnabled(path.exists())
        preview_path = Path(preview_image_path) if preview_image_path else None
        result_images = []
        if preview_path is None and path.exists():
            result_images = [
                p for p in sorted(path.rglob("*"))
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ]
            preview_path = result_images[0] if result_images else None
        elif preview_path is not None:
            result_images = [preview_path]
        if preview_path is not None and preview_path.exists():
            if not result_images:
                result_images = [preview_path]
            self._predict_image_paths = result_images
            try:
                self._predict_image_index = result_images.index(preview_path)
            except ValueError:
                self._predict_image_index = 0
            self._set_predict_image_path(preview_path)
            self._predict_result_label.setText(
                f"{summary}，当前预览: {preview_path.name}"
            )
            return
        self._predict_result_label.setText(summary)

    def get_conf_threshold(self) -> float:
        return self._conf_spin.value()

    def get_iou_threshold(self) -> float:
        return self._iou_spin.value()

    def get_overlap_iou_threshold(self) -> float:
        return self._overlap_iou_spin.value()

    def get_class_match_mode(self) -> str:
        return str(self._class_match_mode_combo.currentData())

    def get_predict_imgsz(self) -> int:
        return int(self._predict_imgsz_spin.value())

    def get_predict_device(self) -> str:
        return _DEVICE_OPTIONS.get(self._predict_device_combo.currentText(), "")

    def get_panel_settings(self) -> dict:
        """Return user-facing model/inference settings for persistence."""
        return {
            "conf_threshold": self._conf_spin.value(),
            "iou_threshold": self._iou_spin.value(),
            "overlap_iou_threshold": self._overlap_iou_spin.value(),
            "class_match_mode": self.get_class_match_mode(),
            "predict_imgsz": self.get_predict_imgsz(),
            "predict_device": self.get_predict_device(),
        }

    def apply_panel_settings(self, settings: dict) -> None:
        """Apply persisted model/inference settings, ignoring unknown keys."""
        if not isinstance(settings, dict):
            return
        try:
            if "conf_threshold" in settings:
                self._conf_spin.setValue(float(settings["conf_threshold"]))
            if "iou_threshold" in settings:
                self._iou_spin.setValue(float(settings["iou_threshold"]))
            if "overlap_iou_threshold" in settings:
                self._overlap_iou_spin.setValue(float(settings["overlap_iou_threshold"]))
            if "predict_imgsz" in settings:
                self._predict_imgsz_spin.setValue(int(settings["predict_imgsz"]))
        except (TypeError, ValueError):
            logger.warning("Ignored invalid persisted model panel numeric settings")
        if "class_match_mode" in settings:
            mode = str(settings["class_match_mode"])
            idx = self._class_match_mode_combo.findData(mode)
            if idx >= 0:
                self._class_match_mode_combo.setCurrentIndex(idx)
        if "predict_device" in settings:
            label = _DEVICE_VALUE_TO_LABEL.get(str(settings["predict_device"]), "AUTO")
            self._predict_device_combo.setCurrentText(label)

    def _on_model_selected(self, row: int) -> None:
        if 0 <= row < len(self._models):
            model = self._models[row]
            self._detail_name.setText(model.name)
            self._detail_task.setText(model.task)
            self._detail_base.setText(model.base_model)
            self._detail_classes.setText(", ".join(model.classes))
            metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in model.metrics.items())
            self._detail_metrics.setText(metrics_str or "无")
            self._detail_trained.setText(model.trained_at)
            self._detail_epochs.setText(str(model.epochs))
            self._detail_dataset.setText(f"{model.dataset_size} 张")

            # Format backend info: backend_id (format) version
            backend_parts = [model.backend_id]
            if model.model_format and model.model_format != "pt":
                backend_parts.append(f"({model.model_format})")
            if model.backend_version:
                backend_parts.append(model.backend_version)
            backend_text = " ".join(backend_parts)
            self._detail_backend.setText(backend_text)
            self._detail_backend.setStyleSheet("color: #6c7086; font-size: 11px;")  # gray, smaller

            self._detail_train_params.setText(
                _format_train_params(model.train_params, model.task)
            )
            self._selected_train_dir = self._training_dir_for_model(model)
            self._detail_artifacts.setText(self._format_artifacts(self._selected_train_dir))
            self._btn_open_train_dir.setEnabled(
                self._selected_train_dir is not None and self._selected_train_dir.exists()
            )
        else:
            self._selected_train_dir = None
            self._detail_artifacts.setText("")
            self._btn_open_train_dir.setEnabled(False)

    def _get_selected_id(self) -> str | None:
        item = self._model_list.currentItem()
        return item.data(Qt.UserRole) if item else None

    def _on_load(self) -> None:
        model_id = self._get_selected_id()
        if model_id:
            self.model_load_requested.emit(model_id)

    def _on_delete(self) -> None:
        model_id = self._get_selected_id()
        if model_id:
            self.model_delete_requested.emit(model_id)

    def _on_rename(self) -> None:
        model_id = self._get_selected_id()
        if model_id:
            self.model_rename_requested.emit(model_id)

    def _on_export_onnx(self) -> None:
        model_id = self._get_selected_id()
        if model_id:
            self.model_export_onnx_requested.emit(model_id)

    def _on_export_pt(self) -> None:
        model_id = self._get_selected_id()
        if model_id:
            self.model_export_pt_requested.emit(model_id)

    def _on_open_train_dir(self) -> None:
        if self._selected_train_dir and self._selected_train_dir.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._selected_train_dir)))

    def _on_select_predict_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择推理图片",
            "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp);;所有文件 (*)",
        )
        if not file_path:
            return
        self._predict_source_path = Path(file_path)
        self._predict_source_is_dir = False
        self._last_predict_result_dir = None
        self._btn_open_predict_result_dir.setEnabled(False)
        self._predict_image_paths = [Path(file_path)]
        self._predict_image_index = 0
        self._set_predict_image_path(Path(file_path))

    def _on_select_predict_dir(self) -> None:
        dir_path = QFileDialog.getExistingDirectory(self, "选择推理图片目录")
        if not dir_path:
            return
        root = Path(dir_path)
        paths = sorted(
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not paths:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(self, "提示", "所选目录中没有支持的图片文件")
            return
        self._predict_source_path = root
        self._predict_source_is_dir = True
        self._last_predict_result_dir = None
        self._btn_open_predict_result_dir.setEnabled(False)
        self._predict_image_paths = paths
        self._predict_image_index = 0
        self._set_predict_image_path(paths[0])
        self._predict_result_label.setText(
            f"已加载目录: {root}，共 {len(paths)} 张，点击“开始推理”将推理整个目录"
        )

    def _set_predict_image_path(self, path: Path) -> None:
        self._predict_image_path = path
        self._update_predict_image_label()
        if self._predict_source_is_dir and self._predict_source_path is not None:
            self._predict_result_label.setText(
                f"当前预览目录中的图片；点击“开始推理”将推理整个目录: {self._predict_source_path}"
            )
        else:
            self._predict_result_label.setText("已选择图片，点击“开始推理”")
        self._predict_canvas.load_image(str(path))
        self._predict_canvas.set_annotations([])
        self._update_predict_nav_buttons()

    def _update_predict_image_label(self) -> None:
        if self._predict_image_path is None:
            self._predict_image_label.setText("未选择图片")
            return
        suffix = ""
        if self._predict_image_paths and 0 <= self._predict_image_index < len(self._predict_image_paths):
            suffix = f"  ({self._predict_image_index + 1} / {len(self._predict_image_paths)})"
        self._predict_image_label.setText(f"{self._predict_image_path}{suffix}")

    def _update_predict_nav_buttons(self) -> None:
        has_many = len(self._predict_image_paths) > 1
        self._btn_prev_predict_image.setEnabled(has_many and self._predict_image_index > 0)
        self._btn_next_predict_image.setEnabled(
            has_many and self._predict_image_index < len(self._predict_image_paths) - 1
        )

    def _step_predict_image(self, delta: int) -> None:
        if not self._predict_image_paths:
            return
        next_index = self._predict_image_index + delta
        if not (0 <= next_index < len(self._predict_image_paths)):
            return
        self._predict_image_index = next_index
        self._set_predict_image_path(self._predict_image_paths[next_index])

    def _on_predict_image(self) -> None:
        target = self._predict_source_path or self._predict_image_path
        if target is None:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(self, "提示", "请先选择一张图片或一个目录")
            return
        self.model_predict_requested.emit(str(target))

    def _on_open_predict_result_dir(self) -> None:
        if self._last_predict_result_dir and self._last_predict_result_dir.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_predict_result_dir)))

    def _resolve_model_path(self, model: ModelInfo) -> Path:
        path = Path(model.path)
        if not path.is_absolute() and self._project_dir is not None:
            path = self._project_dir / path
        return path

    def _training_dir_for_model(self, model: ModelInfo) -> Path | None:
        raw = Path(model.path)
        if not raw.is_absolute() and self._project_dir is None:
            return None
        path = self._resolve_model_path(model)
        if path.name in {"best.pt", "last.pt"} and path.parent.name == "weights":
            return path.parent.parent
        if path.parent.name == "weights":
            return path.parent.parent
        return path.parent if path.parent.exists() else None

    def _format_artifacts(self, train_dir: Path | None) -> str:
        if train_dir is None or not train_dir.exists():
            return "无"
        parts: list[str] = []
        results = train_dir / "results.csv"
        if results.exists():
            parts.append("<b>results.csv</b>")
            summary = _summarize_results_csv(results)
            if summary:
                parts.append(summary)
        image_names = _find_training_artifacts(train_dir)
        if image_names:
            parts.append("<b>图片产物</b>: " + ", ".join(image_names))
        return "<br>".join(parts) if parts else "未找到 results.csv / 曲线图 / 样例预测图"


def _summarize_results_csv(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return ""
    if not rows:
        return ""
    last = rows[-1]
    shown = []
    for key, value in last.items():
        clean = key.strip()
        if not value or clean in {"epoch", "time"}:
            continue
        if any(token in clean.lower() for token in ("map", "accuracy", "precision", "recall", "loss")):
            try:
                shown.append(f"{clean}={float(value):.4g}")
            except ValueError:
                shown.append(f"{clean}={value}")
        if len(shown) >= 8:
            break
    return "最后一轮: " + ", ".join(shown) if shown else ""


def _find_training_artifacts(train_dir: Path) -> list[str]:
    preferred = [
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "F1_curve.png",
        "P_curve.png",
        "R_curve.png",
        "val_batch0_pred.jpg",
        "val_batch1_pred.jpg",
        "train_batch0.jpg",
        "labels.jpg",
    ]
    found = [name for name in preferred if (train_dir / name).exists()]
    if found:
        return found
    return sorted(
        p.name for p in train_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )[:12]
