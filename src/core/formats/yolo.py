"""YOLO format import/export (detection, segmentation, pose, and OBB)."""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import yaml

from src.core.annotation import Annotation, ImageAnnotation, Keypoint
from src.core.class_mapping import resolve_detection_class_map


def _iter_label_files(labels_dir: Path) -> list[Path]:
    """Return YOLO label files, excluding auxiliary metadata files."""
    return [
        txt_path
        for txt_path in sorted(labels_dir.rglob("*.txt"))
        if txt_path.name.lower() != "classes.txt"
    ]


def _find_classes_txt(labels_dir: Path) -> Path | None:
    """Search for classes.txt in the given dir or its parent."""
    for candidate in [
        labels_dir / "classes.txt",
        labels_dir.parent / "classes.txt",
    ]:
        if candidate.exists():
            return candidate
    return None


def _load_classes_txt(classes_txt: Path | str) -> list[str]:
    """Read class names from a YOLO classes.txt file."""
    classes: list[str] = []
    for raw_line in Path(classes_txt).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Some labeling tools write ``0:class_name`` instead of one name per line.
        prefix, separator, name = line.partition(":")
        if separator and prefix.strip().isdigit() and name.strip():
            class_id = int(prefix.strip())
            while len(classes) <= class_id:
                classes.append(str(len(classes)))
            classes[class_id] = name.strip()
        else:
            classes.append(line)
    return classes


def _parse_detection_fields(line: str, txt_path: Path, line_no: int) -> tuple[int, float, float, float, float]:
    """Parse the leading YOLO detection fields with context-rich errors."""
    parts = line.split()
    if len(parts) < 5:
        raise ValueError(f"{txt_path.name}:{line_no}: expected at least 5 fields, got {len(parts)}")
    try:
        return (
            int(parts[0]),
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
            float(parts[4]),
        )
    except ValueError as exc:
        raise ValueError(f"{txt_path.name}:{line_no}: {exc}") from exc


def _normalize_detection_export_annotations(
    image_annotations: list[ImageAnnotation],
    classes: list[str],
) -> list[ImageAnnotation]:
    normalized: list[ImageAnnotation] = []

    for image_annotation in image_annotations:
        normalized_annotations = []
        for annotation in image_annotation.annotations:
            class_name = annotation.class_name
            if not class_name:
                if 0 <= annotation.class_id < len(classes):
                    class_name = classes[annotation.class_id]
                elif annotation.class_id >= 0:
                    class_name = str(annotation.class_id)
            normalized_annotations.append(replace(annotation, class_name=class_name))
        normalized.append(replace(image_annotation, annotations=normalized_annotations))

    return normalized


def _export_label_path(labels_dir: Path, image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        rel = Path(path.name)
    else:
        parts = path.parts
        if parts and parts[0].casefold() == "images":
            rel = Path(*parts[1:]) if len(parts) > 1 else Path(path.name)
        else:
            rel = path
    return labels_dir / rel.with_suffix(".txt")


def _write_yolo_data_yaml(output_dir: Path, class_names: list[str]) -> None:
    data = {
        "path": ".",
        "train": "images",
        "val": "images",
        "nc": len(class_names),
        "names": class_names,
    }
    (output_dir / "data.yaml").write_text(
        yaml.dump(data, default_flow_style=False),
        encoding="utf-8",
    )


def export_yolo_detection(
    image_annotations: list[ImageAnnotation],
    output_dir: Path | str,
    classes: list[str],
    only_confirmed: bool = False,
) -> None:
    """Export annotations to YOLO detection format (txt + data.yaml)."""
    output_dir = Path(output_dir)
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    normalized_annotations = _normalize_detection_export_annotations(image_annotations, classes)
    class_map = resolve_detection_class_map(
        normalized_annotations,
        classes,
        only_confirmed=only_confirmed,
    )

    for ia in normalized_annotations:
        lines = []
        for ann in ia.annotations:
            if only_confirmed and not ann.confirmed:
                continue
            if ann.bbox is None:
                continue
            cid = class_map.id_by_name[ann.class_name]
            cx, cy, w, h = ann.bbox
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        label_path = _export_label_path(labels_dir, ia.image_path)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")

    _write_yolo_data_yaml(output_dir, class_map.names)


def export_yolo_segment(
    image_annotations: list[ImageAnnotation],
    output_dir: Path | str,
    classes: list[str],
    only_confirmed: bool = False,
) -> None:
    """Export annotations to YOLO segmentation format."""
    output_dir = Path(output_dir)
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    normalized_annotations = _normalize_detection_export_annotations(image_annotations, classes)
    class_map = resolve_detection_class_map(
        normalized_annotations,
        classes,
        only_confirmed=only_confirmed,
    )

    for ia in normalized_annotations:
        lines = []
        for ann in ia.annotations:
            if only_confirmed and not ann.confirmed:
                continue
            if len(ann.polygon) >= 3:
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

            cid = class_map.id_by_name[ann.class_name]
            parts = [str(cid)]
            for x, y in polygon:
                parts.extend([f"{x:.6f}", f"{y:.6f}"])
            lines.append(" ".join(parts))

        label_path = _export_label_path(labels_dir, ia.image_path)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")

    _write_yolo_data_yaml(output_dir, class_map.names)


def _bbox_polygon(bbox: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    cx, cy, w, h = bbox
    x1 = max(0.0, cx - w / 2)
    y1 = max(0.0, cy - h / 2)
    x2 = min(1.0, cx + w / 2)
    y2 = min(1.0, cy + h / 2)
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def export_yolo_obb(
    image_annotations: list[ImageAnnotation],
    output_dir: Path | str,
    classes: list[str],
    only_confirmed: bool = False,
) -> None:
    """Export four-point oriented boxes as ``class x1 y1 ... x4 y4``."""
    output_dir = Path(output_dir)
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    normalized_annotations = _normalize_detection_export_annotations(image_annotations, classes)
    class_map = resolve_detection_class_map(
        normalized_annotations,
        classes,
        only_confirmed=only_confirmed,
    )

    for ia in normalized_annotations:
        lines = []
        for ann in ia.annotations:
            if only_confirmed and not ann.confirmed:
                continue
            if len(ann.polygon) == 4:
                corners = ann.polygon
            elif ann.bbox is not None and not ann.polygon:
                corners = _bbox_polygon(ann.bbox)
            else:
                continue
            cid = class_map.id_by_name[ann.class_name]
            parts = [str(cid)]
            for x, y in corners:
                parts.extend([f"{x:.6f}", f"{y:.6f}"])
            lines.append(" ".join(parts))

        label_path = _export_label_path(labels_dir, ia.image_path)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")

    _write_yolo_data_yaml(output_dir, class_map.names)


def import_yolo_detection(
    labels_dir: Path | str,
    classes: list[str] | None = None,
    data_yaml: Path | str | None = None,
) -> list[ImageAnnotation]:
    """Import YOLO detection format. Provide classes or data_yaml."""
    labels_dir = Path(labels_dir)

    if classes is None and data_yaml:
        data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
        classes = data["names"]

    if classes is None:
        raise ValueError("Must provide classes or data_yaml")

    results = []
    for txt_path in _iter_label_files(labels_dir):
        annotations = []
        for line_no, line in enumerate(txt_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            cid, cx, cy, w, h = _parse_detection_fields(line.strip(), txt_path, line_no)
            annotations.append(Annotation(
                class_name=classes[cid] if cid < len(classes) else str(cid),
                class_id=cid,
                bbox=(cx, cy, w, h),
                confirmed=True,
                source="manual",
            ))
        results.append(ImageAnnotation(
            image_path=txt_path.stem,
            image_size=(0, 0),  # unknown without actual image
            annotations=annotations,
        ))
    return results


def export_yolo_pose(
    image_annotations: list[ImageAnnotation],
    output_dir: Path | str,
    classes: list[str],
    kpt_dim: int = 3,
    only_confirmed: bool = False,
) -> None:
    """Export annotations to YOLO pose format."""
    output_dir = Path(output_dir)
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    normalized_annotations = _normalize_detection_export_annotations(image_annotations, classes)
    class_map = resolve_detection_class_map(
        normalized_annotations,
        classes,
        only_confirmed=only_confirmed,
    )

    for ia in normalized_annotations:
        lines = []
        for ann in ia.annotations:
            if only_confirmed and not ann.confirmed:
                continue
            if ann.bbox is None:
                continue
            cid = class_map.id_by_name[ann.class_name]
            cx, cy, w, h = ann.bbox
            parts = [f"{cid}", f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
            for kp in ann.keypoints:
                parts.append(f"{kp.x:.6f}")
                parts.append(f"{kp.y:.6f}")
                if kpt_dim == 3:
                    parts.append(f"{kp.visible}")
            lines.append(" ".join(parts))
        label_path = _export_label_path(labels_dir, ia.image_path)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")

    _write_yolo_data_yaml(output_dir, class_map.names)


def export_yolo(
    image_annotations: list[ImageAnnotation],
    output_dir: Path | str,
    classes: list[str],
    only_confirmed: bool = False,
    task_type: str = "detect",
) -> None:
    """Export the YOLO representation matching the project task type."""
    if task_type == "segment":
        export_yolo_segment(
            image_annotations,
            output_dir,
            classes,
            only_confirmed=only_confirmed,
        )
        return
    if task_type == "pose":
        export_yolo_pose(
            image_annotations,
            output_dir,
            classes,
            only_confirmed=only_confirmed,
        )
        return
    if task_type == "obb":
        export_yolo_obb(
            image_annotations,
            output_dir,
            classes,
            only_confirmed=only_confirmed,
        )
        return
    export_yolo_detection(
        image_annotations,
        output_dir,
        classes,
        only_confirmed=only_confirmed,
    )


def import_yolo_pose(
    labels_dir: Path | str,
    classes: list[str],
    kpt_labels: list[str],
    kpt_dim: int = 3,
) -> list[ImageAnnotation]:
    """Import YOLO pose format."""
    labels_dir = Path(labels_dir)
    num_kpts = len(kpt_labels)

    results = []
    for txt_path in _iter_label_files(labels_dir):
        annotations = []
        for line_no, line in enumerate(txt_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            parts = line.strip().split()
            cid, cx, cy, w, h = _parse_detection_fields(line.strip(), txt_path, line_no)
            keypoints = []
            kp_start = 5
            try:
                for i in range(num_kpts):
                    offset = kp_start + i * kpt_dim
                    kx = float(parts[offset])
                    ky = float(parts[offset + 1])
                    vis = int(float(parts[offset + 2])) if kpt_dim == 3 else 2
                    keypoints.append(Keypoint(x=kx, y=ky, visible=vis, label=kpt_labels[i]))
            except (IndexError, ValueError) as exc:
                raise ValueError(f"{txt_path.name}:{line_no}: {exc}") from exc
            annotations.append(Annotation(
                class_name=classes[cid] if cid < len(classes) else str(cid),
                class_id=cid,
                bbox=(cx, cy, w, h),
                keypoints=keypoints,
                confirmed=True,
                source="manual",
            ))
        results.append(ImageAnnotation(
            image_path=txt_path.stem,
            image_size=(0, 0),
            annotations=annotations,
        ))
    return results


def import_yolo_obb(
    labels_dir: Path | str,
    classes: list[str],
) -> list[ImageAnnotation]:
    """Import normalized YOLO OBB labels with exactly four corner points."""
    labels_dir = Path(labels_dir)
    results = []
    for txt_path in _iter_label_files(labels_dir):
        annotations = []
        for line_no, line in enumerate(txt_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 9:
                raise ValueError(f"{txt_path.name}:{line_no}: expected 9 OBB fields, got {len(parts)}")
            try:
                cid = int(parts[0])
                values = [float(value) for value in parts[1:]]
            except ValueError as exc:
                raise ValueError(f"{txt_path.name}:{line_no}: {exc}") from exc
            if cid < 0:
                raise ValueError(f"{txt_path.name}:{line_no}: class id must be non-negative")
            if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
                raise ValueError(
                    f"{txt_path.name}:{line_no}: OBB coordinates must be finite and normalized to [0, 1]"
                )
            polygon = [(values[index], values[index + 1]) for index in range(0, 8, 2)]
            xs = [point[0] for point in polygon]
            ys = [point[1] for point in polygon]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            annotations.append(Annotation(
                class_name=classes[cid] if cid < len(classes) else str(cid),
                class_id=cid,
                bbox=((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1),
                polygon=polygon,
                confirmed=True,
                source="manual",
            ))
        results.append(ImageAnnotation(
            image_path=txt_path.stem,
            image_size=(0, 0),
            annotations=annotations,
        ))
    return results


def _find_data_yaml(labels_dir: Path) -> Path | None:
    """Search for data.yaml in the given dir, parent dir, or sibling paths."""
    for candidate in [
        labels_dir / "data.yaml",
        labels_dir.parent / "data.yaml",
    ]:
        if candidate.exists():
            return candidate
    return None


def _detect_yolo_format(labels_dir: Path, task_type: str | None = None) -> tuple[str, int]:
    """Detect whether YOLO labels are detection or pose by inspecting the first file.

    Returns ("detection", 0), ("obb", 0), or ("pose", num_keypoints).
    For pose, assumes kpt_dim=3 (x, y, visibility).
    """
    for txt_path in _iter_label_files(labels_dir):
        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        first_line = text.split("\n")[0].strip()
        parts = first_line.split()
        n = len(parts)
        if task_type == "obb":
            if n != 9:
                raise ValueError(f"{txt_path.name}: expected 9 OBB fields, got {n}")
            return "obb", 0
        if n <= 5:
            return "detection", 0
        # Assume kpt_dim=3: extra columns = num_keypoints * 3
        extra = n - 5
        if extra % 3 == 0:
            return "pose", extra // 3
        elif extra % 2 == 0:
            return "pose", extra // 2
        # Fallback: treat as detection
        return "detection", 0
    return "detection", 0


def import_yolo_auto(
    labels_dir: Path | str,
    classes: list[str] | None = None,
    data_yaml: Path | str | None = None,
    kpt_labels: list[str] | None = None,
    kpt_dim: int = 3,
    task_type: str | None = None,
) -> list[ImageAnnotation]:
    """Auto-detect YOLO format and import it using an optional task hint.

    Searches for data.yaml in the directory and its parent.
    Falls back to numeric class names if no classes are available.
    """
    labels_dir = Path(labels_dir)

    # Resolve classes from data.yaml if not provided
    if classes is None and data_yaml is None:
        data_yaml = _find_data_yaml(labels_dir)

    if classes is None and data_yaml is not None:
        data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8")) or {}
        classes = data.get("names")

    if classes is None:
        classes_txt = _find_classes_txt(labels_dir)
        if classes_txt is not None:
            classes = _load_classes_txt(classes_txt)

    # Detect format
    fmt, num_kpts = _detect_yolo_format(labels_dir, task_type=task_type)

    if fmt == "obb":
        if classes is None:
            classes = _infer_classes_from_files(labels_dir)
        return import_yolo_obb(labels_dir, classes)

    if fmt == "pose" and num_kpts > 0:
        if classes is None:
            # Infer max class id from files to generate fallback names
            classes = _infer_classes_from_files(labels_dir)
        if kpt_labels is None:
            kpt_labels = [f"kp_{i}" for i in range(num_kpts)]
        return import_yolo_pose(labels_dir, classes, kpt_labels, kpt_dim)
    else:
        if classes is None:
            classes = _infer_classes_from_files(labels_dir)
        return import_yolo_detection(labels_dir, classes)


def _infer_classes_from_files(labels_dir: Path) -> list[str]:
    """Scan all txt files to find max class_id and generate numeric class names."""
    max_id = -1
    for txt_path in _iter_label_files(labels_dir):
        for line_no, line in enumerate(txt_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            parts = line.strip().split()
            try:
                cid = int(parts[0])
            except ValueError as exc:
                raise ValueError(f"{txt_path.name}:{line_no}: {exc}") from exc
            if cid > max_id:
                max_id = cid
    if max_id < 0:
        return []
    return [str(i) for i in range(max_id + 1)]
