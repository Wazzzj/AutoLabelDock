"""iSAT instance-segmentation JSON format import/export."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.core.annotation import Annotation, ImageAnnotation


_BACKGROUND_CATEGORY = "__background__"


def _export_relative_image_path(image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return Path(path.name)
    parts = path.parts
    if parts and parts[0].casefold() == "images":
        return Path(*parts[1:]) if len(parts) > 1 else Path(path.name)
    return path


def _pixel_polygon(
    annotation: Annotation,
    width: int,
    height: int,
) -> list[tuple[float, float]]:
    if len(annotation.polygon) >= 3:
        return [
            (round(x * width, 6), round(y * height, 6))
            for x, y in annotation.polygon
        ]
    if annotation.bbox is None:
        return []
    cx, cy, bbox_width, bbox_height = annotation.bbox
    x1 = max(0.0, (cx - bbox_width / 2) * width)
    y1 = max(0.0, (cy - bbox_height / 2) * height)
    x2 = min(float(width), (cx + bbox_width / 2) * width)
    y2 = min(float(height), (cy + bbox_height / 2) * height)
    return [
        (round(x1, 6), round(y1, 6)),
        (round(x2, 6), round(y1, 6)),
        (round(x2, 6), round(y2, 6)),
        (round(x1, 6), round(y2, 6)),
    ]


def _pixel_bbox(
    annotation: Annotation,
    polygon: list[tuple[float, float]],
    width: int,
    height: int,
) -> list[float]:
    if annotation.bbox is not None:
        cx, cy, bbox_width, bbox_height = annotation.bbox
        return [
            round(max(0.0, (cx - bbox_width / 2) * width), 6),
            round(max(0.0, (cy - bbox_height / 2) * height), 6),
            round(min(float(width), (cx + bbox_width / 2) * width), 6),
            round(min(float(height), (cy + bbox_height / 2) * height), 6),
        ]
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def _polygon_area(polygon: list[tuple[float, float]]) -> float:
    area = 0.0
    for index, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(index + 1) % len(polygon)]
        area += x1 * y2 - x2 * y1
    return round(abs(area) / 2.0, 6)


def export_isat(
    image_annotations: list[ImageAnnotation],
    output_dir: Path | str,
    only_confirmed: bool = False,
) -> None:
    """Export one standard iSAT JSON annotation file per labeled image."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_annotation in image_annotations:
        width, height = image_annotation.image_size
        if width <= 0 or height <= 0:
            raise ValueError(
                f"iSAT 导出需要有效的图像尺寸: {image_annotation.image_path}"
            )

        relative_image = _export_relative_image_path(image_annotation.image_path)
        json_path = output_dir / relative_image.with_suffix(".json")
        json_path.parent.mkdir(parents=True, exist_ok=True)

        objects = []
        for annotation in image_annotation.annotations:
            if only_confirmed and not annotation.confirmed:
                continue
            polygon = _pixel_polygon(annotation, width, height)
            if len(polygon) < 3:
                continue
            group = len(objects) + 1
            objects.append({
                "category": annotation.class_name,
                "group": group,
                "segmentation": [[x, y] for x, y in polygon],
                "area": _polygon_area(polygon),
                "layer": group,
                "bbox": _pixel_bbox(annotation, polygon, width, height),
                "iscrowd": False,
                "note": "",
            })

        image_folder = (Path("images") / relative_image.parent).as_posix()
        data = {
            "info": {
                "description": "ISAT",
                "folder": image_folder,
                "name": relative_image.name,
                "width": width,
                "height": height,
                "depth": 3,
                "note": "",
            },
            "objects": objects,
        }
        json_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def is_isat_data(data: object) -> bool:
    """Return whether a decoded JSON value has the iSAT annotation structure."""
    if not isinstance(data, dict):
        return False
    info = data.get("info")
    objects = data.get("objects")
    if not isinstance(info, dict) or not isinstance(objects, list):
        return False
    return str(info.get("description", "")).strip().casefold() == "isat"


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _image_dimension(value: Any) -> int:
    number = _finite_float(value)
    if number is None or number <= 0:
        return 0
    return int(number)


def _polygon_from_segmentation(
    segmentation: object,
    width: int,
    height: int,
) -> list[tuple[float, float]]:
    if not isinstance(segmentation, list):
        return []

    raw_points: list[tuple[float, float]] = []
    if segmentation and all(not isinstance(item, (list, tuple)) for item in segmentation):
        if len(segmentation) % 2:
            return []
        for index in range(0, len(segmentation), 2):
            x = _finite_float(segmentation[index])
            y = _finite_float(segmentation[index + 1])
            if x is None or y is None:
                return []
            raw_points.append((x, y))
    else:
        for point in segmentation:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                return []
            x = _finite_float(point[0])
            y = _finite_float(point[1])
            if x is None or y is None:
                return []
            raw_points.append((x, y))

    if len(raw_points) < 3:
        return []
    return [(x / width, y / height) for x, y in raw_points]


def _bbox_from_isat(
    bbox: object,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    """Convert iSAT pixel ``[x1, y1, x2, y2]`` to normalized center form."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    values = [_finite_float(value) for value in bbox]
    if any(value is None for value in values):
        return None
    x1, y1, x2, y2 = (float(value) for value in values)
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    return (
        (left + right) / (2 * width),
        (top + bottom) / (2 * height),
        (right - left) / width,
        (bottom - top) / height,
    )


def _bbox_from_polygon(
    polygon: list[tuple[float, float]],
) -> tuple[float, float, float, float] | None:
    if not polygon:
        return None
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def import_isat_file(json_path: Path | str) -> ImageAnnotation | None:
    """Import one iSAT JSON file.

    Each iSAT object becomes one annotation containing both its polygon and
    bounding box, so the result can be used by segmentation and detection
    projects alike. Non-iSAT JSON files return ``None``.
    """
    json_path = Path(json_path)
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not is_isat_data(data):
        return None

    info = data["info"]
    width = _image_dimension(info.get("width"))
    height = _image_dimension(info.get("height"))
    if not width or not height:
        raise ValueError(f"iSAT 标注缺少有效的图像尺寸: {json_path}")

    raw_image_path = str(info.get("name") or json_path.with_suffix(".jpg").name)
    image_path = raw_image_path.replace("\\", "/").rsplit("/", 1)[-1]
    annotations: list[Annotation] = []

    for obj in data["objects"]:
        if not isinstance(obj, dict):
            continue
        class_name = str(obj.get("category", "")).strip()
        if not class_name or class_name.casefold() == _BACKGROUND_CATEGORY:
            continue

        polygon = _polygon_from_segmentation(
            obj.get("segmentation"),
            width,
            height,
        )
        bbox = _bbox_from_isat(obj.get("bbox"), width, height)
        if bbox is None:
            bbox = _bbox_from_polygon(polygon)
        if bbox is None and not polygon:
            continue

        annotation = Annotation(
            class_name=class_name,
            class_id=0,
            bbox=bbox,
            polygon=polygon,
            confirmed=True,
            source="isat",
        )
        annotation.clamp()
        annotations.append(annotation)

    return ImageAnnotation(
        image_path=image_path,
        image_size=(width, height),
        annotations=annotations,
    )


def import_isat(input_dir: Path | str) -> list[ImageAnnotation]:
    """Import all iSAT JSON annotations below a dataset directory."""
    input_dir = Path(input_dir)
    results: list[ImageAnnotation] = []
    for json_path in sorted(input_dir.rglob("*.json")):
        if json_path.name == "project.json":
            continue
        annotation = import_isat_file(json_path)
        if annotation is not None:
            results.append(annotation)
    return results
