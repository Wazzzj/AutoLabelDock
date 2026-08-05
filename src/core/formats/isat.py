"""iSAT instance-segmentation JSON format import."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.core.annotation import Annotation, ImageAnnotation


_BACKGROUND_CATEGORY = "__background__"


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
