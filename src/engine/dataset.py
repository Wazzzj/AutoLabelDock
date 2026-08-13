"""Dataset preparation for YOLO training (train/val split, symlinks, data.yaml)."""
from __future__ import annotations

import logging
import random
import shutil
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import yaml

from src.core.annotation import ImageAnnotation
from src.core.class_mapping import ResolvedClassMap, resolve_detection_class_map
from src.core.label_io import load_annotation
from src.core.project import ProjectManager
from src.core.tags import TagFilter
from src.utils.fs import link_or_copy

logger = logging.getLogger(__name__)


def _resolved_annotation_class_name(annotation, classes: list[str]) -> str:
    class_name = annotation.class_name
    if not class_name:
        if 0 <= annotation.class_id < len(classes):
            class_name = classes[annotation.class_id]
        elif annotation.class_id >= 0:
            class_name = str(annotation.class_id)
    return class_name


def _is_trainable_annotation(annotation, task: str) -> bool:
    if task == "segment":
        return len(annotation.polygon) >= 3 or annotation.bbox is not None
    if task == "obb":
        return len(annotation.polygon) == 4 or (
            annotation.bbox is not None and not annotation.polygon
        )
    return annotation.bbox is not None


def count_selected_training_images(
    project: ProjectManager,
    task: str = "detect",
    tag_filter: TagFilter | None = None,
    status_filter: str | None = None,
    class_filter: str | None = None,
    data_folder: str | None = None,
) -> int:
    """Return the number of images that match the training data filters."""
    count = 0
    classes = project.config.classes
    for img_path in project.list_images(data_folder=data_folder):
        ia = load_annotation(project.label_path_for(img_path))
        if ia is None:
            continue
        if status_filter is not None and ia.status != status_filter:
            continue
        if tag_filter is not None and not tag_filter.matches(ia.tags):
            continue

        if task == "classify":
            if (
                ia.image_tags
                and ia.image_tags_confirmed
                and (class_filter is None or ia.image_tags[0] == class_filter)
            ):
                count += 1
            continue

        for ann in ia.annotations:
            if not ann.confirmed or not _is_trainable_annotation(ann, task):
                continue
            class_name = _resolved_annotation_class_name(ann, classes)
            if class_filter is None or class_name == class_filter:
                count += 1
                break
    return count


class DatasetPreparer:
    """Prepares a YOLO-compatible dataset from a project."""

    def __init__(self, project_manager: ProjectManager):
        self.pm = project_manager

    def prepare(
        self,
        output_dir: Path | str,
        task: str = "detect",
        val_ratio: float = 0.2,
        seed: int = 42,
        kpt_shape: list[int] | None = None,
        tag_filter: TagFilter | None = None,
        status_filter: str | None = None,
        class_filter: str | None = None,
        data_folder: str | None = None,
    ) -> Path:
        """Prepare dataset and return path to data.yaml.

        When ``tag_filter`` is provided and non-empty, only images whose
        per-image ``tags`` match the filter are considered (see
        ``src.core.tags.TagFilter``). Passing ``None`` or an empty filter
        is a no-op — original behavior is preserved.
        """
        output_dir = Path(output_dir)

        # Clean previous dataset to avoid stale symlinks / ultralytics .cache files
        if output_dir.exists():
            shutil.rmtree(output_dir)

        classes = self.pm.config.classes

        # Collect labeled images
        labeled: list[tuple[Path, ImageAnnotation]] = []
        for img_path in self.pm.list_images(data_folder=data_folder):
            label_path = self.pm.label_path_for(img_path)
            ia = load_annotation(label_path)
            if ia is None:
                continue
            if status_filter is not None and ia.status != status_filter:
                continue
            if tag_filter is not None and not tag_filter.matches(ia.tags):
                continue

            # Classification: check image_tags
            if task == "classify":
                if (
                    ia.image_tags
                    and ia.image_tags_confirmed
                    and (class_filter is None or ia.image_tags[0] == class_filter)
                ):
                    labeled.append((img_path, ia))
            # Detection/Segment/Pose/OBB: check confirmed annotations
            else:
                confirmed = [
                    a for a in ia.annotations
                    if a.confirmed and _is_trainable_annotation(a, task)
                ]
                if not confirmed:
                    continue
                normalized = self._normalize_detection_pose_or_segment_annotations(
                    ImageAnnotation(
                        image_path=ia.image_path,
                        image_size=ia.image_size,
                        annotations=confirmed,
                        image_tags=list(ia.image_tags),
                    ),
                    classes,
                )
                if class_filter is not None:
                    normalized = replace(
                        normalized,
                        annotations=[
                            a for a in normalized.annotations
                            if a.class_name == class_filter
                        ],
                    )
                if not normalized.annotations:
                    continue
                labeled.append((
                    img_path,
                    normalized,
                ))

        if not labeled:
            raise ValueError("没有找到已确认标注的图片，无法准备数据集")

        # Stratified split by primary class
        train_set, val_set = self._stratified_split(labeled, val_ratio, seed)

        if not train_set:
            raise ValueError("训练集为空，请减小验证集比例或增加标注数据")

        logger.info(
            "Dataset prepared: %d train, %d val (task=%s)",
            len(train_set), len(val_set), task,
        )

        if task == "classify":
            class_names = [class_filter] if class_filter else list(self.pm.config.classes)
            self._export_classify(output_dir, train_set, val_set, class_names)
            # Ultralytics' check_cls_dataset expects the dataset root directory
            # (it does `data_dir / "train"` directly). A data.yaml file is never
            # consulted for classify, so don't write one and return the root dir.
            return output_dir

        class_map = resolve_detection_class_map(
            [ia for _, ia in train_set + val_set],
            classes,
            only_confirmed=False,
        )
        self._export_detection_pose_or_segment(output_dir, train_set, val_set, class_map, task)

        # Generate data.yaml for detect/segment/pose/obb
        data_yaml_path = output_dir / "data.yaml"
        data = self._build_data_yaml(
            output_dir,
            class_map.names,
            task,
            kpt_shape,
            has_val=bool(val_set),
        )
        data_yaml_path.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")
        return data_yaml_path

    def _stratified_split(
        self,
        items: list[tuple[Path, ImageAnnotation]],
        val_ratio: float,
        seed: int,
    ) -> tuple[list[tuple[Path, ImageAnnotation]], list[tuple[Path, ImageAnnotation]]]:
        """Split items into train/val using stratified sampling by primary class."""
        if val_ratio <= 0:
            return items, []
        if val_ratio >= 1:
            return [], items

        by_class: dict[str, list[tuple[Path, ImageAnnotation]]] = defaultdict(list)
        for item in items:
            ia = item[1]
            # Classification: use image_tags[0]
            if ia.image_tags:
                primary_class = ia.image_tags[0]
            # Detection/Pose: use first annotation's class
            elif ia.annotations:
                primary_class = ia.annotations[0].class_name
            else:
                continue  # Skip items without class info
            by_class[primary_class].append(item)

        rng = random.Random(seed)
        train, val = [], []
        for cls_items in by_class.values():
            rng.shuffle(cls_items)
            n_val = max(1, round(len(cls_items) * val_ratio))
            if n_val >= len(cls_items):
                n_val = max(0, len(cls_items) - 1)
            val.extend(cls_items[:n_val])
            train.extend(cls_items[n_val:])

        return train, val

    def _normalize_detection_pose_or_segment_annotations(
        self,
        image_annotation: ImageAnnotation,
        classes: list[str],
    ) -> ImageAnnotation:
        normalized_annotations = []
        for annotation in image_annotation.annotations:
            class_name = _resolved_annotation_class_name(annotation, classes)
            normalized_annotations.append(replace(annotation, class_name=class_name))
        return replace(image_annotation, annotations=normalized_annotations)

    def _export_detection_pose_or_segment(
        self,
        output_dir: Path,
        train_set: list[tuple[Path, ImageAnnotation]],
        val_set: list[tuple[Path, ImageAnnotation]],
        class_map: ResolvedClassMap,
        task: str,
    ) -> None:
        """Export to the YOLO label representation for the selected task."""
        for split_name, split_data in [("train", train_set), ("val", val_set)]:
            if not split_data:
                continue
            img_dir = output_dir / split_name / "images"
            lbl_dir = output_dir / split_name / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            for img_path, ia in split_data:
                link = img_dir / img_path.name
                if not link.exists():
                    link_or_copy(img_path, link)

                lines = []
                for ann in ia.annotations:
                    cid = class_map.id_by_name[ann.class_name]
                    if task == "obb":
                        if len(ann.polygon) == 4:
                            polygon = ann.polygon
                        elif ann.bbox is not None and not ann.polygon:
                            cx, cy, w, h = ann.bbox
                            x1 = max(0.0, cx - w / 2)
                            y1 = max(0.0, cy - h / 2)
                            x2 = min(1.0, cx + w / 2)
                            y2 = min(1.0, cy + h / 2)
                            polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                        else:
                            continue
                        parts = [f"{cid}"]
                        for x, y in polygon:
                            parts.extend([f"{x:.6f}", f"{y:.6f}"])
                    elif task == "segment":
                        if ann.polygon and len(ann.polygon) >= 3:
                            polygon = ann.polygon
                        elif ann.bbox is not None:
                            cx, cy, w, h = ann.bbox
                            x1 = max(0.0, cx - w / 2)
                            y1 = max(0.0, cy - h / 2)
                            x2 = min(1.0, cx + w / 2)
                            y2 = min(1.0, cy + h / 2)
                            polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                        else:
                            continue
                        parts = [
                            f"{cid}",
                        ]
                        for x, y in polygon:
                            parts.extend([f"{x:.6f}", f"{y:.6f}"])
                    else:
                        if ann.bbox is None:
                            continue
                        cx, cy, w, h = ann.bbox
                        parts = [f"{cid}", f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
                    if task == "pose" and ann.keypoints:
                        for kp in ann.keypoints:
                            parts.extend([f"{kp.x:.6f}", f"{kp.y:.6f}", f"{kp.visible}"])
                    lines.append(" ".join(parts))
                (lbl_dir / (img_path.stem + ".txt")).write_text(
                    "\n".join(lines) + "\n" if lines else "", encoding="utf-8"
                )

    def _export_classify(
        self,
        output_dir: Path,
        train_set: list[tuple[Path, ImageAnnotation]],
        val_set: list[tuple[Path, ImageAnnotation]],
        class_names: list[str],
    ) -> None:
        """Export to YOLO classification directory structure.

        All project classes get a subdirectory in every non-empty split — even
        classes with no images — so the alphabetical class index ultralytics
        derives from the folder list stays stable across runs. Empty folders
        are tolerated via ``ImageFolder(allow_empty=True)`` on torchvision >=0.18.
        """
        all_classes = list(class_names)
        for split_name, split_data in [("train", train_set), ("val", val_set)]:
            if not split_data:
                continue
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for cls_name in all_classes:
                (split_dir / cls_name).mkdir(parents=True, exist_ok=True)
            for img_path, ia in split_data:
                if ia.image_tags:
                    cls_name = ia.image_tags[0]
                else:
                    cls_name = ia.annotations[0].class_name
                cls_dir = split_dir / cls_name
                cls_dir.mkdir(parents=True, exist_ok=True)
                link = cls_dir / img_path.name
                if not link.exists():
                    link_or_copy(img_path, link)

    def _build_data_yaml(
        self,
        output_dir: Path,
        classes: list[str],
        task: str,
        kpt_shape: list[int] | None,
        has_val: bool = True,
    ) -> dict:
        """Build data.yaml content dict for detect/segment/pose/obb tasks.

        Classification does not use data.yaml — ultralytics reads classify datasets
        directly from the directory structure.
        """
        data = {
            "path": str(output_dir.resolve()),
            "train": "train/images",
            "nc": len(classes),
            "names": classes,
        }
        if has_val:
            data["val"] = "val/images"
        if task == "pose" and kpt_shape:
            data["kpt_shape"] = kpt_shape
        return data
