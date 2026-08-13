"""Pascal VOC-style rotated bounding-box XML importer."""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

from src.core.annotation import Annotation, ImageAnnotation


def _required_float(parent: ET.Element, tag: str, xml_path: Path) -> float:
    value = parent.findtext(tag)
    if value is None:
        raise ValueError(f"{xml_path.name}: missing <{tag}>")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{xml_path.name}: invalid <{tag}> value {value!r}") from exc


def _rotated_corners(
    cx: float,
    cy: float,
    width: float,
    height: float,
    angle: float,
    image_width: float,
    image_height: float,
) -> list[tuple[float, float]]:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    corners = []
    for dx, dy in (
        (-width / 2, -height / 2),
        (width / 2, -height / 2),
        (width / 2, height / 2),
        (-width / 2, height / 2),
    ):
        x = (cx + dx * cos_a - dy * sin_a) / image_width
        y = (cy + dx * sin_a + dy * cos_a) / image_height
        corners.append((max(0.0, min(1.0, x)), max(0.0, min(1.0, y))))
    return corners


def import_voc_obb(
    source_dir: Path | str,
    classes: list[str] | None = None,
) -> list[ImageAnnotation]:
    """Import ``robndbox`` XML files used by roLabelImg and similar tools.

    Angles are interpreted as radians, matching the common VOC rotated-box XML
    convention. Standard axis-aligned ``bndbox`` objects are also accepted and
    converted to four-corner boxes.
    """
    source_dir = Path(source_dir)
    class_names = list(classes or [])
    results: list[ImageAnnotation] = []

    for xml_path in sorted(source_dir.rglob("*.xml")):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as exc:
            raise ValueError(f"{xml_path.name}: invalid XML: {exc}") from exc

        size = root.find("size")
        if size is None:
            raise ValueError(f"{xml_path.name}: missing <size>")
        image_width = _required_float(size, "width", xml_path)
        image_height = _required_float(size, "height", xml_path)
        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"{xml_path.name}: image size must be positive")

        image_name = (root.findtext("filename") or xml_path.stem).strip()
        # Sidecar XML files are matched by their own stem. Some labeling tools
        # leave a stale <filename> value after a batch rename (for example an
        # XML/image pair ending in ``_1`` while <filename> still ends in ``_0``).
        if Path(image_name).stem.casefold() != xml_path.stem.casefold():
            image_name = xml_path.stem
        annotations: list[Annotation] = []
        for obj in root.findall("object"):
            class_name = (obj.findtext("name") or "").strip()
            if not class_name:
                continue
            if class_name not in class_names:
                class_names.append(class_name)
            class_id = class_names.index(class_name)

            rotated = obj.find("robndbox")
            if rotated is not None:
                corners = _rotated_corners(
                    _required_float(rotated, "cx", xml_path),
                    _required_float(rotated, "cy", xml_path),
                    _required_float(rotated, "w", xml_path),
                    _required_float(rotated, "h", xml_path),
                    _required_float(rotated, "angle", xml_path),
                    image_width,
                    image_height,
                )
            else:
                box = obj.find("bndbox")
                if box is None:
                    continue
                x1 = max(0.0, min(1.0, _required_float(box, "xmin", xml_path) / image_width))
                y1 = max(0.0, min(1.0, _required_float(box, "ymin", xml_path) / image_height))
                x2 = max(0.0, min(1.0, _required_float(box, "xmax", xml_path) / image_width))
                y2 = max(0.0, min(1.0, _required_float(box, "ymax", xml_path) / image_height))
                corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

            xs = [point[0] for point in corners]
            ys = [point[1] for point in corners]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            annotations.append(Annotation(
                class_name=class_name,
                class_id=class_id,
                bbox=((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1),
                polygon=corners,
                confirmed=True,
                source="manual",
            ))

        results.append(ImageAnnotation(
            image_path=image_name,
            image_size=(int(image_width), int(image_height)),
            annotations=annotations,
        ))

    return results
