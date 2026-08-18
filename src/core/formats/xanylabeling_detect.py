"""X-AnyLabeling JSON import for axis-aligned object detection."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

from src.core.annotation import Annotation, ImageAnnotation


def _relative_image_path(image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return Path(path.name)
    parts = path.parts
    if parts and parts[0].casefold() == "images":
        return Path(*parts[1:]) if len(parts) > 1 else Path(path.name)
    return path


def _json_path(output_dir: Path, image_path: str) -> Path:
    return output_dir / _relative_image_path(image_path).with_suffix(".json")


def _image_path(output_dir: Path, json_path: Path, image_path: str) -> str:
    image_abs = output_dir.parent / "images" / _relative_image_path(image_path)
    return os.path.relpath(image_abs, json_path.parent).replace("\\", "/")


def _rectangle_bbox(
    shape: dict,
    image_width: float,
    image_height: float,
) -> tuple[float, float, float, float] | None:
    """Convert X-AnyLabeling's two- or four-point rectangle to a bbox."""
    if str(shape.get("shape_type", "")).casefold() != "rectangle":
        return None

    points = shape.get("points", [])
    if not isinstance(points, list) or len(points) not in {2, 4}:
        return None

    try:
        coordinates = [(float(point[0]), float(point[1])) for point in points]
    except (TypeError, ValueError, IndexError):
        return None
    if not all(math.isfinite(value) for point in coordinates for value in point):
        return None

    xs = [point[0] for point in coordinates]
    ys = [point[1] for point in coordinates]
    x1 = max(0.0, min(float(image_width), min(xs)))
    y1 = max(0.0, min(float(image_height), min(ys)))
    x2 = max(0.0, min(float(image_width), max(xs)))
    y2 = max(0.0, min(float(image_height), max(ys)))
    if x2 <= x1 or y2 <= y1:
        return None

    return (
        (x1 + x2) / 2 / image_width,
        (y1 + y2) / 2 / image_height,
        (x2 - x1) / image_width,
        (y2 - y1) / image_height,
    )


def _shape_confidence(shape: dict) -> float:
    score = shape.get("score")
    if isinstance(score, (int, float)) and math.isfinite(float(score)):
        return max(0.0, min(1.0, float(score)))
    return 1.0


def import_xanylabeling_detect(input_dir: Path | str) -> list[ImageAnnotation]:
    """Import X-AnyLabeling ``rectangle`` shapes as detection boxes.

    X-AnyLabeling files can contain either Labelme-style two-point rectangles
    or four explicit corner points. Both representations become an
    axis-aligned internal bounding box. Empty JSON files are ignored.
    """
    input_dir = Path(input_dir)
    results: list[ImageAnnotation] = []

    for json_path in sorted(input_dir.rglob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue

        image_width = data.get("imageWidth", 0)
        image_height = data.get("imageHeight", 0)
        if not isinstance(image_width, (int, float)) or not isinstance(
            image_height, (int, float)
        ):
            continue
        if image_width <= 0 or image_height <= 0:
            continue

        annotations: list[Annotation] = []
        shapes = data.get("shapes", [])
        if not isinstance(shapes, list):
            continue
        for shape in shapes:
            if not isinstance(shape, dict):
                continue
            bbox = _rectangle_bbox(shape, image_width, image_height)
            if bbox is None:
                continue
            annotations.append(
                Annotation(
                    class_name=str(shape.get("label") or "unknown"),
                    class_id=0,
                    bbox=bbox,
                    confidence=_shape_confidence(shape),
                    confirmed=True,
                    source="manual",
                )
            )

        if not annotations:
            continue

        raw_image_path = data.get("imagePath") or (json_path.stem + ".jpg")
        image_path = str(raw_image_path).replace("\\", "/").rsplit("/", 1)[-1]
        results.append(
            ImageAnnotation(
                image_path=image_path,
                image_size=(int(image_width), int(image_height)),
                annotations=annotations,
            )
        )

    return results


def export_xanylabeling_detect(
    image_annotations: list[ImageAnnotation],
    output_dir: Path | str,
    only_confirmed: bool = False,
    task_type: str = "detect",
) -> None:
    """Export detection boxes as X-AnyLabeling four-point rectangles."""
    if task_type.strip().casefold() != "detect":
        raise ValueError("X-AnyLabeling Detect 导出仅适用于 detect 项目")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_annotation in image_annotations:
        image_width, image_height = image_annotation.image_size
        if image_width <= 0 or image_height <= 0:
            raise ValueError(
                "X-AnyLabeling Detect 导出需要有效的图像尺寸: "
                f"{image_annotation.image_path}"
            )

        shapes = []
        for annotation in image_annotation.annotations:
            if only_confirmed and not annotation.confirmed:
                continue
            if annotation.bbox is None:
                continue
            center_x, center_y, width, height = annotation.bbox
            x1 = (center_x - width / 2) * image_width
            y1 = (center_y - height / 2) * image_height
            x2 = (center_x + width / 2) * image_width
            y2 = (center_y + height / 2) * image_height
            shapes.append(
                {
                    "label": annotation.class_name,
                    "score": round(float(annotation.confidence), 6),
                    "points": [
                        [round(x1, 6), round(y1, 6)],
                        [round(x2, 6), round(y1, 6)],
                        [round(x2, 6), round(y2, 6)],
                        [round(x1, 6), round(y2, 6)],
                    ],
                    "group_id": None,
                    "description": "",
                    "difficult": False,
                    "shape_type": "rectangle",
                    "flags": {},
                    "attributes": {},
                    "kie_linking": [],
                }
            )

        output_path = _json_path(output_dir, image_annotation.image_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "2.5.4",
            "flags": {},
            "shapes": shapes,
            "imagePath": _image_path(
                output_dir,
                output_path,
                image_annotation.image_path,
            ),
            "imageData": None,
            "imageHeight": image_height,
            "imageWidth": image_width,
        }
        output_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
