"""Training augmentation preview with sampled transforms and annotation overlays."""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QImage, QImageReader, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.core.label_io import load_annotation
from src.ui.icons import icon
from src.ui.theme import PALETTE, set_button_role, text_style


_MAX_PROCESS_SIZE = 640
_BORDER_COLOR = (114, 114, 114)


@dataclass
class _PreviewFrame:
    image: np.ndarray
    overlay: np.ndarray
    masks: list[np.ndarray]
    annotation_count: int = 0

    def copy(self) -> "_PreviewFrame":
        return _PreviewFrame(
            self.image.copy(),
            self.overlay.copy(),
            [mask.copy() for mask in self.masks],
            self.annotation_count,
        )


class _PreviewImageLabel(QLabel):
    """Image label that keeps its source pixmap crisp when the dialog resizes."""

    def __init__(self, image: QImage, parent=None):
        super().__init__(parent)
        self._source = QPixmap.fromImage(image)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(280, 190)

    def resizeEvent(self, event) -> None:
        self._refresh_pixmap()
        super().resizeEvent(event)

    def showEvent(self, event) -> None:
        self._refresh_pixmap()
        super().showEvent(event)

    def _refresh_pixmap(self) -> None:
        if self._source.isNull() or self.width() < 2 or self.height() < 2:
            return
        self.setPixmap(
            self._source.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )


class AugmentationPreviewDialog(QDialog):
    """Show repeatable single-image and full training augmentation samples."""

    def __init__(
        self,
        image_paths: Path | str | Sequence[Path | str],
        params: dict,
        parent=None,
        project=None,
    ):
        super().__init__(parent)
        if isinstance(image_paths, (str, Path)):
            image_paths = [image_paths]
        self._image_paths = [Path(path) for path in image_paths if Path(path).is_file()]
        self._params = dict(params)
        self._project = project
        self._frame_cache: dict[Path, _PreviewFrame] = {}

        self.setWindowTitle("数据增强预览")
        self.setMinimumSize(980, 700)
        self.resize(1180, 820)
        self._init_ui()
        QTimer.singleShot(0, self._generate)

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        controls.addWidget(QLabel("样本:"))
        self._source_combo = QComboBox()
        for path in self._image_paths:
            self._source_combo.addItem(path.name, path)
        self._source_combo.setMinimumWidth(260)
        self._source_combo.currentIndexChanged.connect(self._generate)
        controls.addWidget(self._source_combo, 1)

        controls.addWidget(QLabel("随机种子:"))
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 999999)
        self._seed_spin.setValue(random.SystemRandom().randrange(100000))
        self._seed_spin.setToolTip("相同种子会生成相同的增强采样")
        controls.addWidget(self._seed_spin)

        self._btn_refresh = QPushButton(icon("refresh"), "重新采样")
        set_button_role(self._btn_refresh, "secondary")
        self._btn_refresh.clicked.connect(self._resample)
        controls.addWidget(self._btn_refresh)
        layout.addLayout(controls)

        self._summary = QLabel(self._parameter_summary())
        self._summary.setStyleSheet(text_style("hint"))
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)

        self._grid_container = QWidget()
        self._grid_layout = QGridLayout(self._grid_container)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        self._grid_layout.setHorizontalSpacing(8)
        self._grid_layout.setVerticalSpacing(8)
        for column in range(3):
            self._grid_layout.setColumnStretch(column, 1)
        for row in range(2):
            self._grid_layout.setRowStretch(row, 1)
        layout.addWidget(self._grid_container, 1)

        self._loading_label = QLabel("正在生成训练增强采样...")
        self._loading_label.setAlignment(Qt.AlignCenter)
        self._loading_label.setStyleSheet(text_style("muted"))
        layout.addWidget(self._loading_label)

    def _parameter_summary(self) -> str:
        names = {
            "hsv_h": "H",
            "hsv_s": "S",
            "hsv_v": "V",
            "degrees": "旋转",
            "translate": "平移",
            "scale": "缩放",
            "shear": "剪切",
            "perspective": "透视",
            "flipud": "上下翻转",
            "fliplr": "左右翻转",
            "mosaic": "Mosaic",
            "mixup": "MixUp",
            "copy_paste": "Copy-Paste",
            "erasing": "Erasing",
            "auto_augment": "Auto Augment",
        }
        values = []
        for key, value in self._params.items():
            if key.startswith("_") or key not in names or value in (0, 0.0, "", None, "none"):
                continue
            values.append(f"{names[key]} {value}")
        return "当前训练参数: " + ("  |  ".join(values) if values else "未启用随机增强")

    def _resample(self) -> None:
        self._seed_spin.setValue(random.SystemRandom().randrange(100000))
        self._generate()

    def _generate(self) -> None:
        if not self._image_paths:
            self._loading_label.setText("没有可用图片")
            return

        self._btn_refresh.setEnabled(False)
        self._source_combo.setEnabled(False)
        self._clear_grid()
        self._loading_label.setText("正在生成训练增强采样...")
        self._loading_label.show()
        QApplication.processEvents()

        base_path = self._source_combo.currentData() or self._image_paths[0]
        base_path = Path(base_path)
        frame = self._training_input_frame(base_path)
        if frame is None:
            self._loading_label.setText("无法加载图片")
            self._btn_refresh.setEnabled(True)
            self._source_combo.setEnabled(True)
            return

        seed = self._seed_spin.value()
        donor_paths = [path for path in self._image_paths if path != base_path]
        donor_rng = random.Random(seed ^ 0xA5A5)
        if len(donor_paths) > 12:
            donor_paths = donor_rng.sample(donor_paths, 12)
        donors = [frame]
        donors.extend(
            loaded
            for path in donor_paths
            if (loaded := self._training_input_frame(path)) is not None
        )
        if not donors:
            donors = [frame]

        original = _compose_overlay(frame)
        self._grid_layout.addWidget(
            self._make_preview(
                "训练输入",
                _array_to_qimage(original),
                f"{base_path.name}  |  标注 {frame.annotation_count}",
            ),
            0,
            0,
        )

        sample_specs = [
            ("单图增强 A", False, seed + 101),
            ("单图增强 B", False, seed + 211),
            ("训练采样 1", True, seed + 307),
            ("训练采样 2", True, seed + 401),
            ("训练采样 3", True, seed + 503),
        ]
        for index, (title, include_mix, sample_seed) in enumerate(sample_specs, start=1):
            augmented, operations = _augment_frame(
                frame,
                donors,
                self._params,
                sample_seed,
                include_mix=include_mix,
            )
            row, column = divmod(index, 3)
            self._grid_layout.addWidget(
                self._make_preview(
                    title,
                    _array_to_qimage(_compose_overlay(augmented)),
                    "  |  ".join(operations) if operations else "本次未触发随机操作",
                ),
                row,
                column,
            )
            QApplication.processEvents()

        self._loading_label.hide()
        self._btn_refresh.setEnabled(True)
        self._source_combo.setEnabled(True)

    def _training_input_frame(self, path: Path) -> _PreviewFrame | None:
        if path not in self._frame_cache:
            loaded = _load_frame(path, self._project)
            if loaded is None:
                return None
            self._frame_cache[path] = loaded
        size = min(max(128, int(self._params.get("_imgsz", 640))), _MAX_PROCESS_SIZE)
        return _fit_frame(self._frame_cache[path], size, size, cover=False)

    def _clear_grid(self) -> None:
        while self._grid_layout.count() > 0:
            item = self._grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _make_preview(self, title: str, image: QImage, detail: str) -> QWidget:
        tile = QFrame()
        tile.setObjectName("augmentationPreviewTile")
        tile.setStyleSheet(
            "QFrame#augmentationPreviewTile {"
            f"background: {PALETTE['panel']}; border: 1px solid {PALETTE['line']};"
            "border-radius: 6px; }"
        )
        layout = QVBoxLayout(tile)
        layout.setContentsMargins(7, 7, 7, 7)
        layout.setSpacing(5)

        title_label = QLabel(title)
        title_label.setStyleSheet(text_style("section"))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        layout.addWidget(_PreviewImageLabel(image), 1)

        detail_label = QLabel(detail)
        detail_label.setStyleSheet(text_style("hint"))
        detail_label.setAlignment(Qt.AlignCenter)
        detail_label.setWordWrap(True)
        detail_label.setMinimumHeight(34)
        layout.addWidget(detail_label)
        return tile


def _load_frame(path: Path, project=None) -> _PreviewFrame | None:
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    qimage = reader.read()
    if qimage.isNull():
        return None
    image = _qimage_to_array(qimage)
    h, w = image.shape[:2]
    max_dim = max(h, w)
    if max_dim > _MAX_PROCESS_SIZE * 2:
        scale = (_MAX_PROCESS_SIZE * 2) / max_dim
        image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = image.shape[:2]

    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    masks: list[np.ndarray] = []
    count = 0
    if project is not None:
        try:
            image_annotation = load_annotation(project.label_path_for(path))
        except (OSError, ValueError, AttributeError):
            image_annotation = None
        if image_annotation:
            count = len(image_annotation.annotations)
            thickness = max(2, round(max(h, w) / 320))
            for ann in image_annotation.annotations:
                try:
                    color = QColor(project.config.get_class_color(ann.class_name))
                    rgba = (color.red(), color.green(), color.blue(), 235)
                except (AttributeError, TypeError):
                    rgba = (124, 92, 255, 235)
                if ann.polygon:
                    points = np.array(
                        [[round(x * (w - 1)), round(y * (h - 1))] for x, y in ann.polygon],
                        dtype=np.int32,
                    )
                    if len(points) >= 3:
                        cv2.polylines(overlay, [points], True, rgba, thickness, cv2.LINE_AA)
                        mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.fillPoly(mask, [points], 255)
                        masks.append(mask)
                elif ann.bbox:
                    cx, cy, bw, bh = ann.bbox
                    p1 = (round((cx - bw / 2) * w), round((cy - bh / 2) * h))
                    p2 = (round((cx + bw / 2) * w), round((cy + bh / 2) * h))
                    cv2.rectangle(overlay, p1, p2, rgba, thickness, cv2.LINE_AA)
                for kp in ann.keypoints:
                    center = (round(kp.x * (w - 1)), round(kp.y * (h - 1)))
                    cv2.circle(overlay, center, thickness + 2, rgba, -1, cv2.LINE_AA)
    return _PreviewFrame(image, overlay, masks, count)


def _qimage_to_array(image: QImage) -> np.ndarray:
    converted = image.convertToFormat(QImage.Format_RGB888)
    ptr = converted.bits()
    ptr.setsize(converted.byteCount())
    raw = np.frombuffer(ptr, dtype=np.uint8).reshape(converted.height(), converted.bytesPerLine())
    return raw[:, : converted.width() * 3].reshape(converted.height(), converted.width(), 3).copy()


def _array_to_qimage(image: np.ndarray) -> QImage:
    contiguous = np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))
    h, w = contiguous.shape[:2]
    return QImage(contiguous.data, w, h, contiguous.strides[0], QImage.Format_RGB888).copy()


def _fit_frame(frame: _PreviewFrame, width: int, height: int, *, cover: bool) -> _PreviewFrame:
    src_h, src_w = frame.image.shape[:2]
    scale = max(width / src_w, height / src_h) if cover else min(width / src_w, height / src_h)
    new_w = max(1, round(src_w * scale))
    new_h = max(1, round(src_h * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR

    image = cv2.resize(frame.image, (new_w, new_h), interpolation=interpolation)
    overlay = cv2.resize(frame.overlay, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    masks = [cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST) for mask in frame.masks]

    if cover:
        x = max(0, (new_w - width) // 2)
        y = max(0, (new_h - height) // 2)
        return _PreviewFrame(
            image[y:y + height, x:x + width].copy(),
            overlay[y:y + height, x:x + width].copy(),
            [mask[y:y + height, x:x + width].copy() for mask in masks],
            frame.annotation_count,
        )

    canvas = np.full((height, width, 3), _BORDER_COLOR, dtype=np.uint8)
    canvas_overlay = np.zeros((height, width, 4), dtype=np.uint8)
    offset_x = (width - new_w) // 2
    offset_y = (height - new_h) // 2
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = image
    canvas_overlay[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = overlay
    canvas_masks = []
    for mask in masks:
        target = np.zeros((height, width), dtype=np.uint8)
        target[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = mask
        canvas_masks.append(target)
    return _PreviewFrame(canvas, canvas_overlay, canvas_masks, frame.annotation_count)


def _augment_frame(
    source: _PreviewFrame,
    donors: Sequence[_PreviewFrame],
    params: dict,
    seed: int,
    *,
    include_mix: bool,
) -> tuple[_PreviewFrame, list[str]]:
    rng = random.Random(seed)
    frame = source.copy()
    operations: list[str] = []

    if include_mix and _trigger(rng, params.get("mosaic", 0)):
        frame = _mosaic(frame, donors, rng)
        operations.append("Mosaic")

    if include_mix and _trigger(rng, params.get("copy_paste", 0)):
        mode = str(params.get("copy_paste_mode", "flip"))
        frame, pasted = _copy_paste(frame, donors, rng, mode)
        operations.append("Copy-Paste" if pasted else "Copy-Paste(无可用轮廓)")

    if include_mix and _trigger(rng, params.get("mixup", 0)):
        frame, ratio = _mixup(frame, donors, rng)
        operations.append(f"MixUp {ratio:.2f}/{1 - ratio:.2f}")

    frame, geometric = _random_perspective(frame, params, rng)
    operations.extend(geometric)

    frame, hsv_text = _hsv(frame, params, rng)
    if hsv_text:
        operations.append(hsv_text)

    if _trigger(rng, params.get("flipud", 0)):
        frame = _flip_frame(frame, vertical=True)
        operations.append("上下翻转")
    if _trigger(rng, params.get("fliplr", 0)):
        frame = _flip_frame(frame, vertical=False)
        operations.append("左右翻转")

    auto_augment = str(params.get("auto_augment", "")).lower()
    if auto_augment and auto_augment != "none":
        frame, policy_ops = _classification_policy(frame, auto_augment, rng)
        operations.extend(policy_ops)

    if _trigger(rng, params.get("erasing", 0)):
        frame, erased = _random_erasing(frame, rng)
        operations.append(f"Erasing {erased:.0%}")

    return frame, operations


def _trigger(rng: random.Random, probability) -> bool:
    try:
        value = max(0.0, min(1.0, float(probability)))
    except (TypeError, ValueError):
        return False
    return value > 0 and rng.random() < value


def _mosaic(base: _PreviewFrame, donors: Sequence[_PreviewFrame], rng: random.Random) -> _PreviewFrame:
    h, w = base.image.shape[:2]
    output = np.full_like(base.image, _BORDER_COLOR)
    overlay = np.zeros_like(base.overlay)
    masks: list[np.ndarray] = []
    choices = [base] + [rng.choice(donors) for _ in range(3)]
    regions = [
        (0, 0, w // 2, h // 2),
        (w // 2, 0, w - w // 2, h // 2),
        (0, h // 2, w // 2, h - h // 2),
        (w // 2, h // 2, w - w // 2, h - h // 2),
    ]
    count = 0
    for item, (x, y, tile_w, tile_h) in zip(choices, regions):
        tile = _fit_frame(item, tile_w, tile_h, cover=True)
        output[y:y + tile_h, x:x + tile_w] = tile.image
        overlay[y:y + tile_h, x:x + tile_w] = _alpha_merge(
            overlay[y:y + tile_h, x:x + tile_w], tile.overlay
        )
        for mask in tile.masks:
            target = np.zeros((h, w), dtype=np.uint8)
            target[y:y + tile_h, x:x + tile_w] = mask
            masks.append(target)
        count += tile.annotation_count
    return _PreviewFrame(output, overlay, masks, count)


def _mixup(
    base: _PreviewFrame,
    donors: Sequence[_PreviewFrame],
    rng: random.Random,
) -> tuple[_PreviewFrame, float]:
    h, w = base.image.shape[:2]
    donor = _fit_frame(rng.choice(donors), w, h, cover=True)
    ratio = rng.betavariate(32.0, 32.0)
    image = np.clip(base.image.astype(np.float32) * ratio + donor.image * (1 - ratio), 0, 255).astype(np.uint8)
    return (
        _PreviewFrame(
            image,
            _alpha_merge(base.overlay, donor.overlay),
            base.masks + donor.masks,
            base.annotation_count + donor.annotation_count,
        ),
        ratio,
    )


def _copy_paste(
    base: _PreviewFrame,
    donors: Sequence[_PreviewFrame],
    rng: random.Random,
    mode: str,
) -> tuple[_PreviewFrame, bool]:
    h, w = base.image.shape[:2]
    donor = base.copy() if mode == "flip" else _fit_frame(rng.choice(donors), w, h, cover=True)
    if mode == "flip":
        donor = _flip_frame(donor, vertical=False)
    if not donor.masks:
        return base, False

    output = base.copy()
    candidates = list(donor.masks)
    rng.shuffle(candidates)
    selected = candidates[: max(1, min(3, len(candidates)))]
    combined = np.zeros((h, w), dtype=np.uint8)
    for mask in selected:
        combined = cv2.bitwise_or(combined, mask)
    active = combined > 0
    output.image[active] = donor.image[active]

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output.overlay, contours, -1, (246, 189, 96, 240), 2, cv2.LINE_AA)
    output.masks.extend(selected)
    output.annotation_count += len(selected)
    return output, True


def _random_perspective(
    frame: _PreviewFrame,
    params: dict,
    rng: random.Random,
) -> tuple[_PreviewFrame, list[str]]:
    h, w = frame.image.shape[:2]
    degrees = float(params.get("degrees", 0) or 0)
    translate = float(params.get("translate", 0) or 0)
    scale_gain = float(params.get("scale", 0) or 0)
    shear = float(params.get("shear", 0) or 0)
    perspective = float(params.get("perspective", 0) or 0)
    if not any((degrees, translate, scale_gain, shear, perspective)):
        return frame, []

    center = np.eye(3, dtype=np.float32)
    center[0, 2] = -w / 2
    center[1, 2] = -h / 2

    perspective_matrix = np.eye(3, dtype=np.float32)
    px = rng.uniform(-perspective, perspective)
    py = rng.uniform(-perspective, perspective)
    perspective_matrix[2, 0] = px
    perspective_matrix[2, 1] = py

    angle = rng.uniform(-degrees, degrees)
    scale = rng.uniform(max(0.01, 1 - scale_gain), 1 + scale_gain)
    rotate = np.eye(3, dtype=np.float32)
    rotate[:2] = cv2.getRotationMatrix2D((0, 0), angle, scale)

    shear_x = rng.uniform(-shear, shear)
    shear_y = rng.uniform(-shear, shear)
    shear_matrix = np.eye(3, dtype=np.float32)
    shear_matrix[0, 1] = math.tan(math.radians(shear_x))
    shear_matrix[1, 0] = math.tan(math.radians(shear_y))

    tx = rng.uniform(0.5 - translate, 0.5 + translate)
    ty = rng.uniform(0.5 - translate, 0.5 + translate)
    translation = np.eye(3, dtype=np.float32)
    translation[0, 2] = tx * w
    translation[1, 2] = ty * h
    matrix = translation @ shear_matrix @ rotate @ perspective_matrix @ center
    warped = _warp_frame(frame, matrix, w, h, use_perspective=bool(perspective))

    operations = []
    if degrees:
        operations.append(f"旋转 {angle:+.1f}°")
    if scale_gain:
        operations.append(f"缩放 {scale:.2f}x")
    if translate:
        operations.append(f"平移 {(tx - 0.5):+.0%}/{(ty - 0.5):+.0%}")
    if shear:
        operations.append(f"剪切 {shear_x:+.1f}°/{shear_y:+.1f}°")
    if perspective:
        operations.append(f"透视 {px:+.4f}/{py:+.4f}")
    return warped, operations


def _warp_frame(
    frame: _PreviewFrame,
    matrix: np.ndarray,
    width: int,
    height: int,
    *,
    use_perspective: bool = True,
) -> _PreviewFrame:
    if use_perspective:
        warp: Callable = cv2.warpPerspective
        transform = matrix
    else:
        warp = cv2.warpAffine
        transform = matrix[:2]
    image = warp(
        frame.image,
        transform,
        dsize=(width, height),
        flags=cv2.INTER_LINEAR,
        borderValue=_BORDER_COLOR,
    )
    overlay = warp(
        frame.overlay,
        transform,
        dsize=(width, height),
        flags=cv2.INTER_LINEAR,
        borderValue=(0, 0, 0, 0),
    )
    masks = [
        warp(mask, transform, dsize=(width, height), flags=cv2.INTER_NEAREST, borderValue=0)
        for mask in frame.masks
    ]
    return _PreviewFrame(image, overlay, masks, frame.annotation_count)


def _hsv(
    frame: _PreviewFrame,
    params: dict,
    rng: random.Random,
) -> tuple[_PreviewFrame, str]:
    gains = [float(params.get(key, 0) or 0) for key in ("hsv_h", "hsv_s", "hsv_v")]
    if not any(gains):
        return frame, ""
    sampled = [rng.uniform(-1, 1) * gain for gain in gains]
    hsv = cv2.cvtColor(frame.image, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    x = np.arange(256, dtype=np.float32)
    hue_lut = ((x + sampled[0] * 180) % 180).astype(np.uint8)
    saturation_lut = np.clip(x * (sampled[1] + 1), 0, 255).astype(np.uint8)
    value_lut = np.clip(x * (sampled[2] + 1), 0, 255).astype(np.uint8)
    saturation_lut[0] = 0
    adjusted = cv2.merge(
        (cv2.LUT(hue, hue_lut), cv2.LUT(saturation, saturation_lut), cv2.LUT(value, value_lut))
    )
    output = frame.copy()
    output.image = cv2.cvtColor(adjusted, cv2.COLOR_HSV2RGB)
    return output, f"HSV {sampled[0]:+.3f}/{sampled[1]:+.2f}/{sampled[2]:+.2f}"


def _flip_frame(frame: _PreviewFrame, *, vertical: bool) -> _PreviewFrame:
    axis = 0 if vertical else 1
    return _PreviewFrame(
        np.ascontiguousarray(np.flip(frame.image, axis=axis)),
        np.ascontiguousarray(np.flip(frame.overlay, axis=axis)),
        [np.ascontiguousarray(np.flip(mask, axis=axis)) for mask in frame.masks],
        frame.annotation_count,
    )


def _classification_policy(
    frame: _PreviewFrame,
    policy: str,
    rng: random.Random,
) -> tuple[_PreviewFrame, list[str]]:
    operations = ["亮度", "对比度", "饱和度", "旋转", "平移", "Solarize", "Posterize"]
    if policy == "augmix":
        branches = []
        branch_names = []
        for _ in range(3):
            branch = frame.copy()
            names = []
            for op in rng.sample(operations, 2):
                branch, name = _classification_operation(branch, op, rng)
                names.append(name)
            branches.append(branch.image.astype(np.float32))
            branch_names.extend(names)
        weights = np.array([rng.random() for _ in branches], dtype=np.float32)
        weights /= max(weights.sum(), 1e-6)
        mixed = sum(weight * branch for weight, branch in zip(weights, branches))
        blend = rng.betavariate(1.0, 1.0)
        output = frame.copy()
        output.image = np.clip(frame.image * (1 - blend) + mixed * blend, 0, 255).astype(np.uint8)
        return output, [f"AugMix {blend:.2f}", *branch_names[:3]]

    count = 2 if policy in {"randaugment", "autoaugment"} else 1
    output = frame
    names = []
    for op in rng.sample(operations, count):
        output, name = _classification_operation(output, op, rng)
        names.append(name)
    return output, [policy, *names]


def _classification_operation(
    frame: _PreviewFrame,
    operation: str,
    rng: random.Random,
) -> tuple[_PreviewFrame, str]:
    output = frame.copy()
    if operation == "亮度":
        factor = rng.uniform(0.65, 1.35)
        output.image = np.clip(output.image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return output, f"亮度 {factor:.2f}x"
    if operation == "对比度":
        factor = rng.uniform(0.65, 1.35)
        mean = output.image.mean(axis=(0, 1), keepdims=True)
        output.image = np.clip((output.image - mean) * factor + mean, 0, 255).astype(np.uint8)
        return output, f"对比度 {factor:.2f}x"
    if operation == "饱和度":
        factor = rng.uniform(0.5, 1.5)
        hsv = cv2.cvtColor(output.image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
        output.image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return output, f"饱和度 {factor:.2f}x"
    if operation == "旋转":
        angle = rng.uniform(-25, 25)
        h, w = output.image.shape[:2]
        matrix = np.eye(3, dtype=np.float32)
        matrix[:2] = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return _warp_frame(output, matrix, w, h, use_perspective=False), f"策略旋转 {angle:+.1f}°"
    if operation == "平移":
        h, w = output.image.shape[:2]
        dx = rng.uniform(-0.15, 0.15)
        dy = rng.uniform(-0.15, 0.15)
        matrix = np.array([[1, 0, dx * w], [0, 1, dy * h]], dtype=np.float32)
        affine = np.eye(3, dtype=np.float32)
        affine[:2] = matrix
        return _warp_frame(output, affine, w, h, use_perspective=False), f"策略平移 {dx:+.0%}/{dy:+.0%}"
    if operation == "Solarize":
        threshold = rng.randint(96, 192)
        output.image = np.where(output.image >= threshold, 255 - output.image, output.image).astype(np.uint8)
        return output, f"Solarize {threshold}"

    bits = rng.randint(4, 7)
    shift = 8 - bits
    output.image = ((output.image >> shift) << shift).astype(np.uint8)
    return output, f"Posterize {bits}bit"


def _random_erasing(frame: _PreviewFrame, rng: random.Random) -> tuple[_PreviewFrame, float]:
    h, w = frame.image.shape[:2]
    area_ratio = rng.uniform(0.02, 0.25)
    aspect = math.exp(rng.uniform(math.log(0.3), math.log(3.3)))
    erase_h = max(1, min(h, round(math.sqrt(area_ratio * h * w / aspect))))
    erase_w = max(1, min(w, round(math.sqrt(area_ratio * h * w * aspect))))
    x = rng.randrange(max(1, w - erase_w + 1))
    y = rng.randrange(max(1, h - erase_h + 1))
    noise_rng = np.random.default_rng(rng.randrange(2**32))
    output = frame.copy()
    output.image[y:y + erase_h, x:x + erase_w] = noise_rng.integers(
        0, 256, size=(erase_h, erase_w, 3), dtype=np.uint8
    )
    return output, (erase_h * erase_w) / (h * w)


def _alpha_merge(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    top_alpha = top[..., 3:4].astype(np.float32) / 255.0
    bottom_alpha = bottom[..., 3:4].astype(np.float32) / 255.0
    out_alpha = top_alpha + bottom_alpha * (1 - top_alpha)
    numerator = top[..., :3] * top_alpha + bottom[..., :3] * bottom_alpha * (1 - top_alpha)
    colors = np.divide(
        numerator,
        np.maximum(out_alpha, 1e-6),
        out=np.zeros_like(numerator),
        where=out_alpha > 0,
    )
    return np.concatenate((colors, out_alpha * 255), axis=2).astype(np.uint8)


def _compose_overlay(frame: _PreviewFrame) -> np.ndarray:
    alpha = frame.overlay[..., 3:4].astype(np.float32) / 255.0
    return np.clip(
        frame.image.astype(np.float32) * (1 - alpha) + frame.overlay[..., :3] * alpha,
        0,
        255,
    ).astype(np.uint8)


def _apply_augmentation(image: QImage, params: dict) -> QImage:
    """Backward-compatible helper for callers that preview a single QImage."""
    array = _qimage_to_array(image)
    frame = _PreviewFrame(array, np.zeros((*array.shape[:2], 4), dtype=np.uint8), [])
    size = min(max(array.shape[:2]), _MAX_PROCESS_SIZE)
    frame = _fit_frame(frame, size, size, cover=False)
    augmented, _operations = _augment_frame(
        frame,
        [frame],
        params,
        random.SystemRandom().randrange(2**31),
        include_mix=True,
    )
    return _array_to_qimage(_compose_overlay(augmented))
