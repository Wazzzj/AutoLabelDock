"""Inference adapter for Ultralytics-style prediction results."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from src.core.annotation import Annotation, Keypoint

logger = logging.getLogger(__name__)


class Predictor:
    """Wraps a YOLO model for inference, converting results to Annotations."""

    def __init__(self, model):
        """Initialize with a loaded ultralytics YOLO model instance."""
        self.model = model

    def release(self) -> None:
        """No-op release. Ultralytics models share the in-process runtime and
        don't require explicit GPU teardown; provided so the predictor satisfies
        the optional ``PredictorProtocol.release`` hook used by ModelController."""

    def recommended_imgsz(self) -> int | None:
        """Best-effort inference size stored in an Ultralytics model file."""
        for value in self._iter_imgsz_candidates():
            imgsz = self._coerce_imgsz(value)
            if imgsz is not None:
                return imgsz
        return None

    def _iter_imgsz_candidates(self):
        yolo = self.model
        inner_model = getattr(yolo, "model", None)
        ckpt = getattr(yolo, "ckpt", None)

        for container in (
            getattr(yolo, "overrides", None),
            getattr(yolo, "args", None),
            getattr(inner_model, "args", None),
            getattr(inner_model, "yaml", None),
        ):
            yield from self._imgsz_values_from_mapping(container)

        if isinstance(ckpt, dict):
            for key in ("train_args", "args", "model_args", "yaml"):
                yield from self._imgsz_values_from_mapping(ckpt.get(key))
            yield from self._imgsz_values_from_mapping(ckpt)

    @staticmethod
    def _imgsz_values_from_mapping(container):
        if not isinstance(container, dict):
            return
        for key in ("imgsz", "img_size", "image_size"):
            if key in container:
                yield container[key]

    @classmethod
    def _coerce_imgsz(cls, value) -> int | None:
        if value is None:
            return None
        if hasattr(value, "tolist"):
            try:
                value = value.tolist()
            except Exception:  # noqa: BLE001 - metadata can be library-specific
                return None
        if hasattr(value, "item") and not isinstance(value, (list, tuple, dict)):
            try:
                value = value.item()
            except Exception:  # noqa: BLE001 - metadata can be library-specific
                return None
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            try:
                value = float(value)
            except ValueError:
                return None
        if isinstance(value, (list, tuple)):
            candidates = [cls._coerce_imgsz(item) for item in value]
            candidates = [item for item in candidates if item is not None]
            return max(candidates) if candidates else None
        if isinstance(value, (int, float)):
            imgsz = int(value)
            if 32 <= imgsz <= 4096:
                return imgsz
        return None

    @staticmethod
    def _normalize_class_name(class_name: str) -> str:
        """Normalize class names for tolerant matching across model/project metadata."""
        return " ".join(str(class_name).split()).casefold()

    @staticmethod
    def _resolve_model_class_name(names, cls_id: int) -> str:
        """Resolve a class name from ultralytics model.names, tolerating list/dict variants."""
        if isinstance(names, dict):
            return str(names.get(cls_id, cls_id))
        if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
            return str(names[cls_id])
        return str(cls_id)

    @staticmethod
    def _rotate_closed(points: np.ndarray, start_index: int) -> np.ndarray:
        """从指定位置开始排列轮廓，并在末尾重复起点形成闭环。"""
        return np.concatenate(
            (points[start_index:], points[: start_index + 1]),
            axis=0,
        )

    @classmethod
    def _merge_hole_into_polygon(
        cls,
        outer: np.ndarray,
        hole: np.ndarray,
        outer_orientation: float,
    ) -> np.ndarray:
        """把内轮廓通过零宽度连接线合并进外轮廓。

        当前 Annotation 只有一个扁平 polygon 字段，不能直接保存
        “外轮廓 + 内孔”两个环。因此使用相反方向遍历内轮廓，并通过
        最近点连接，使最终单 polygon 在填充时仍然保留中间孔洞。
        """
        if len(outer) < 3 or len(hole) < 3:
            return outer

        hole_orientation = cv2.contourArea(
            hole.reshape(-1, 1, 2).astype(np.float32),
            oriented=True,
        )
        if (
            outer_orientation
            and hole_orientation
            and outer_orientation * hole_orientation > 0
        ):
            hole = hole[::-1].copy()

        delta = (
            outer[:, None, :].astype(np.float32)
            - hole[None, :, :].astype(np.float32)
        )
        distance_sq = np.sum(delta * delta, axis=2)
        outer_index, hole_index = np.unravel_index(
            int(np.argmin(distance_sq)),
            distance_sq.shape,
        )

        outer_path = cls._rotate_closed(outer, int(outer_index))
        hole_path = cls._rotate_closed(hole, int(hole_index))

        return np.concatenate(
            (
                outer_path,
                hole_path,
                outer[int(outer_index) : int(outer_index) + 1],
            ),
            axis=0,
        )

    @staticmethod
    def _approx_contour(contour: np.ndarray) -> np.ndarray:
        """轻微简化轮廓，同时尽量保留边缘细节。"""
        perimeter = cv2.arcLength(contour, True)
        epsilon = max(0.5, perimeter * 0.001)
        approximated = cv2.approxPolyDP(contour, epsilon, True)
        return approximated.reshape(-1, 2)

    @classmethod
    def _mask_to_normalized_polygons(
        cls,
        mask,
        width: int,
        height: int,
        min_area: float = 20.0,
    ) -> list[list[tuple[float, float]]]:
        """把像素级 mask 转为一个归一化 polygon，并保留内孔。

        一个 YOLO 实例理论上对应一个主要目标。模型 mask 偶尔会包含远离
        主体的细小、不连续噪声块。旧版代码把每个外部连通区域都生成成一条
        标注，因此会在图片边缘出现额外的“乱标”。

        这里仅保留面积最大的外部连通区域，再读取它的内部孔洞。
        """
        array = np.asarray(mask)
        while array.ndim > 2:
            array = array[0]

        if array.shape != (height, width):
            array = cv2.resize(
                array.astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )

        binary = (array > 0.5).astype(np.uint8)
        contours, hierarchy = cv2.findContours(
            binary,
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours or hierarchy is None:
            return []

        hierarchy = hierarchy[0]


        outer_indices = [
            index
            for index, contour in enumerate(contours)
            if hierarchy[index][3] == -1
            and abs(cv2.contourArea(contour)) >= min_area
        ]
        if not outer_indices:
            return []

        contour_index = max(
            outer_indices,
            key=lambda index: abs(cv2.contourArea(contours[index])),
        )
        contour = contours[contour_index]

        outer = cls._approx_contour(contour)
        if len(outer) < 3:
            return []

        outer_orientation = cv2.contourArea(
            outer.reshape(-1, 1, 2).astype(np.float32),
            oriented=True,
        )
        merged = outer


        child_index = int(hierarchy[contour_index][2])
        while child_index != -1:
            child = contours[child_index]
            if abs(cv2.contourArea(child)) >= min_area:
                hole = cls._approx_contour(child)
                if len(hole) >= 3:
                    merged = cls._merge_hole_into_polygon(
                        merged,
                        hole,
                        outer_orientation,
                    )
            child_index = int(hierarchy[child_index][0])

        normalized: list[tuple[float, float]] = []
        for x, y in merged:
            nx = max(0.0, min(1.0, float(x) / float(width)))
            ny = max(0.0, min(1.0, float(y) / float(height)))
            normalized.append((round(nx, 6), round(ny, 6)))

        return [normalized] if len(normalized) >= 3 else []

    @staticmethod
    def _polygon_bbox(
        polygon: list[tuple[float, float]],
        fallback: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        """根据 polygon 重新计算归一化 bbox；没有 polygon 时使用模型框。"""
        if len(polygon) < 3:
            return fallback

        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        return (
            round((x1 + x2) / 2.0, 6),
            round((y1 + y2) / 2.0, 6),
            round(x2 - x1, 6),
            round(y2 - y1, 6),
        )

    def predict_native(
        self,
        image_path: str | Path | list[str | Path],
        conf: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
        device: str | None = None,
        save: bool = False,
        project: str | Path | None = None,
        name: str | None = None,
        exist_ok: bool = False,
    ):
        """Run Ultralytics YOLO's native predict API and return raw results."""
        kwargs = {
            "source": (
                [str(path) for path in image_path]
                if isinstance(image_path, list)
                else str(image_path)
            ),
            "verbose": False,


            "retina_masks": True,
        }
        if conf is not None:
            kwargs["conf"] = conf
        if iou is not None:
            kwargs["iou"] = iou
        if imgsz:
            kwargs["imgsz"] = int(imgsz)
        if device:
            kwargs["device"] = str(device)
        if save:
            kwargs["save"] = True
        if project is not None:
            kwargs["project"] = str(project)
        if name:
            kwargs["name"] = str(name)
        if exist_ok:
            kwargs["exist_ok"] = True
        return self.model.predict(**kwargs)

    def predict(
        self,
        image_path: str | Path,
        conf: float = 0.5,
        iou: float = 0.45,
        project_classes: list[str] | None = None,
        class_match_mode: str = "class_id",
        kpt_labels: list[str] | None = None,
        imgsz: int | None = None,
        device: str | None = None,
        filter_to_project: bool = True,
    ) -> list[Annotation]:
        """Run inference and return list of Annotations."""
        annotations, _ = self._run(
            image_path, conf, iou, project_classes, class_match_mode, kpt_labels,
            imgsz, device, filter_to_project,
        )
        return annotations

    def predict_classify(
        self,
        image_path: str | Path,
        project_classes: list[str] | None = None,
        filter_to_project: bool = True,
        imgsz: int | None = None,
        device: str | None = None,
    ) -> tuple[str, float] | None:
        """Run classify inference. Return (class_name, confidence) or None.

        When ``filter_to_project`` is True (default) and ``project_classes`` is
        non-empty, predictions whose class is not in ``project_classes`` return
        None (legacy behavior). When False, the raw model class name is
        returned even if it is not in the project — caller handles registration.
        """
        results = self.predict_native(image_path, imgsz=imgsz, device=device)
        if not results:
            return None
        return self._parse_classify_result(
            results[0],
            image_path=image_path,
            project_classes=project_classes,
            filter_to_project=filter_to_project,
        )

    def predict_classify_batch(
        self,
        image_paths: list[str | Path],
        project_classes: list[str] | None = None,
        filter_to_project: bool = True,
        imgsz: int | None = None,
        device: str | None = None,
    ) -> list[tuple[str, float] | None]:
        """Run classify inference for multiple images in one model call."""
        if not image_paths:
            return []
        results = self.predict_native(image_paths, imgsz=imgsz, device=device)
        if not results:
            return [None] * len(image_paths)
        if len(results) != len(image_paths):
            logger.warning(
                "Classify batch returned %d results for %d sources; padding missing results with None",
                len(results),
                len(image_paths),
            )
        payloads = [
            self._parse_classify_result(
                result,
                image_path=image_path,
                project_classes=project_classes,
                filter_to_project=filter_to_project,
            )
            for image_path, result in zip(image_paths, results)
        ]
        if len(payloads) < len(image_paths):
            payloads.extend([None] * (len(image_paths) - len(payloads)))
        return payloads

    def _parse_classify_result(
        self,
        result,
        *,
        image_path: str | Path,
        project_classes: list[str] | None,
        filter_to_project: bool,
    ) -> tuple[str, float] | None:
        probs = getattr(result, "probs", None)
        if probs is None:
            return None
        cls_id = int(probs.top1)
        confidence = round(float(probs.top1conf.item()), 4)
        raw_name = self._resolve_model_class_name(self.model.names, cls_id)
        if not project_classes or not filter_to_project:
            return (raw_name, confidence)
        norm = self._normalize_class_name(raw_name)
        for cls in project_classes:
            if self._normalize_class_name(cls) == norm:
                return (cls, confidence)
        logger.warning(
            "Classify prediction '%s' not in project classes %s for %s",
            raw_name, project_classes, image_path,
        )
        return None

    def predict_with_size(
        self,
        image_path: str | Path,
        conf: float = 0.5,
        iou: float = 0.45,
        project_classes: list[str] | None = None,
        class_match_mode: str = "class_id",
        kpt_labels: list[str] | None = None,
        imgsz: int | None = None,
        device: str | None = None,
        filter_to_project: bool = True,
    ) -> tuple[list[Annotation], tuple[int, int]]:
        """Run inference and return annotations + image size (w, h)."""
        return self._run(
            image_path, conf, iou, project_classes, class_match_mode, kpt_labels,
            imgsz, device, filter_to_project,
        )

    def _run(
        self,
        image_path: str | Path,
        conf: float,
        iou: float,
        project_classes: list[str] | None,
        class_match_mode: str,
        kpt_labels: list[str] | None,
        imgsz: int | None,
        device: str | None,
        filter_to_project: bool,
    ) -> tuple[list[Annotation], tuple[int, int]]:
        results = self.predict_native(
            image_path,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
        )
        logger.debug("Predict: %s (conf=%.2f, iou=%.2f)", image_path, conf, iou)
        if not results:
            return [], (0, 0)

        result = results[0]
        h, w = result.orig_shape
        img_size = (w, h)
        names = self.model.names
        matched_annotations = []
        raw_annotations = []
        project_class_lookup: dict[str, tuple[str, int]] = {}
        if project_classes and class_match_mode == "class_name":
            project_class_lookup = {
                self._normalize_class_name(class_name): (class_name, idx)
                for idx, class_name in enumerate(project_classes)
            }

        boxes = result.boxes
        if boxes is None or len(boxes.cls) == 0:
            return [], img_size

        has_kpts = result.keypoints is not None



        mask_arrays: list[np.ndarray] = []
        masks = getattr(result, "masks", None)
        mask_data = getattr(masks, "data", None) if masks is not None else None
        if mask_data is not None:
            if hasattr(mask_data, "detach"):
                mask_data = mask_data.detach()
            if hasattr(mask_data, "cpu"):
                mask_data = mask_data.cpu()

            mask_array = np.asarray(mask_data)
            if mask_array.ndim == 2:
                mask_array = mask_array[None, ...]
            if mask_array.ndim == 3:
                mask_arrays = [
                    mask_array[index]
                    for index in range(mask_array.shape[0])
                ]

        for i in range(len(boxes.cls)):
            cls_id = int(boxes.cls[i].item())
            confidence = round(float(boxes.conf[i].item()), 4)
            raw_class_name = self._resolve_model_class_name(names, cls_id)
            project_match = None
            if project_classes and filter_to_project:
                if class_match_mode == "class_id":
                    if 0 <= cls_id < len(project_classes):
                        project_match = (project_classes[cls_id], cls_id)
                elif class_match_mode == "class_name":
                    project_match = project_class_lookup.get(self._normalize_class_name(raw_class_name))
                else:
                    raise ValueError(f"Unsupported class_match_mode: {class_match_mode}")
            if project_match is not None:
                class_name, resolved_id = project_match
            else:
                class_name = raw_class_name
                resolved_id = cls_id

            cx = round(float(boxes.xywhn[i][0].item()), 6)
            cy = round(float(boxes.xywhn[i][1].item()), 6)
            bw = round(float(boxes.xywhn[i][2].item()), 6)
            bh = round(float(boxes.xywhn[i][3].item()), 6)

            keypoints = []
            if has_kpts and result.keypoints.xyn is not None:
                kpts_xy = result.keypoints.xyn[i]
                kpts_conf = result.keypoints.conf[i] if result.keypoints.conf is not None else None
                for j in range(len(kpts_xy)):
                    kx = round(float(kpts_xy[j][0].item()), 6)
                    ky = round(float(kpts_xy[j][1].item()), 6)
                    kc = float(kpts_conf[j].item()) if kpts_conf is not None else 1.0
                    visible = 2 if kc > 0.5 else (1 if kc > 0 else 0)
                    label = kpt_labels[j] if kpt_labels and j < len(kpt_labels) else f"kp_{j}"
                    keypoints.append(Keypoint(x=kx, y=ky, visible=visible, label=label))

            polygons: list[list[tuple[float, float]]] = []
            if i < len(mask_arrays):
                polygons = self._mask_to_normalized_polygons(
                    mask_arrays[i],
                    w,
                    h,
                )



            if not polygons:
                polygons = [[]]

            fallback_bbox = (cx, cy, bw, bh)
            for polygon in polygons:
                annotation = Annotation(
                    class_name=class_name,
                    class_id=resolved_id,
                    bbox=self._polygon_bbox(polygon, fallback_bbox),
                    polygon=polygon,
                    keypoints=keypoints,
                    confidence=confidence,
                    confirmed=False,
                    source="auto",
                )
                raw_annotations.append(annotation)
                if (
                    not project_classes
                    or not filter_to_project
                    or project_match is not None
                ):
                    matched_annotations.append(annotation)

        if project_classes and filter_to_project and not matched_annotations and raw_annotations:
            # Avoid reporting "no objects" when detections exist but class metadata does not align.
            logger.warning(
                "Predict filtered out all %d detections for %s because model classes did not match "
                "project classes %s; returning raw detections instead",
                len(raw_annotations),
                image_path,
                project_classes,
            )
            annotations = raw_annotations
        else:
            annotations = matched_annotations if project_classes and filter_to_project else raw_annotations

        logger.debug(
            "Predict result: %d annotations (raw=%d, matched=%d)",
            len(annotations),
            len(raw_annotations),
            len(matched_annotations),
        )
        return annotations, img_size
