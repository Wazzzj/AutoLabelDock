"""X-AnyLabeling JSON import/export for oriented bounding boxes."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

from src.core.annotation import Annotation, ImageAnnotation


def import_xanylabeling_obb(input_dir: Path | str) -> list[ImageAnnotation]:
    """Import X-AnyLabeling ``rotation`` shapes from JSON sidecar files."""
    input_dir = Path(input_dir)
    results: list[ImageAnnotation] = []
    for json_path in sorted(input_dir.rglob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
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
        for shape in data.get("shapes", []):
            if not isinstance(shape, dict) or shape.get("shape_type") != "rotation":
                continue
            points = shape.get("points", [])
            if not isinstance(points, list) or len(points) != 4:
                continue
            try:
                polygon = [
                    (float(point[0]) / image_width, float(point[1]) / image_height)
                    for point in points
                ]
            except (TypeError, ValueError, IndexError, ZeroDivisionError):
                continue
            xs = [point[0] for point in polygon]
            ys = [point[1] for point in polygon]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            annotations.append(
                Annotation(
                    class_name=str(shape.get("label") or "unknown"),
                    class_id=0,
                    bbox=((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1),
                    polygon=polygon,
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


def _obb_corners(annotation: Annotation) -> list[tuple[float, float]]:
    if len(annotation.polygon) == 4:
        return list(annotation.polygon)
    if annotation.bbox is None or annotation.polygon:
        return []
    center_x, center_y, width, height = annotation.bbox
    return [
        (center_x - width / 2, center_y - height / 2),
        (center_x + width / 2, center_y - height / 2),
        (center_x + width / 2, center_y + height / 2),
        (center_x - width / 2, center_y + height / 2),
    ]


def _direction(points: list[list[float]]) -> float:
    """Match X-AnyLabeling's first-edge angle normalized to [0, 2π)."""
    x1, y1 = points[0]
    x2, y2 = points[1]
    return math.atan2(y2 - y1, x2 - x1) % (2 * math.pi)


def export_xanylabeling_obb(
    image_annotations: list[ImageAnnotation],
    output_dir: Path | str,
    only_confirmed: bool = False,
    task_type: str = "obb",
) -> None:
    """Export one X-AnyLabeling ``rotation`` JSON per annotated image."""
    if task_type != "obb":
        raise ValueError("X-AnyLabeling OBB 导出仅适用于 OBB 项目")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_annotation in image_annotations:
        image_width, image_height = image_annotation.image_size
        if image_width <= 0 or image_height <= 0:
            raise ValueError(
                f"X-AnyLabeling OBB 导出需要有效的图像尺寸: "
                f"{image_annotation.image_path}"
            )

        shapes = []
        for annotation in image_annotation.annotations:
            if only_confirmed and not annotation.confirmed:
                continue
            corners = _obb_corners(annotation)
            if len(corners) != 4:
                continue
            points = [
                [round(x * image_width, 6), round(y * image_height, 6)]
                for x, y in corners
            ]
            shapes.append({
                "label": annotation.class_name,
                "shape_type": "rotation",
                "flags": {},
                "points": points,
                "group_id": None,
                "description": None,
                "difficult": False,
                "direction": _direction(points),
                "attributes": {},
            })

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
