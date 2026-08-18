"""roLabelImg/Pascal VOC-style rotated bounding-box XML import and export."""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

from src.core.annotation import Annotation, ImageAnnotation


def _annotation_corners(annotation: Annotation) -> list[tuple[float, float]]:
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


def _robndbox_values(
    annotation: Annotation,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float, float] | None:
    corners = _annotation_corners(annotation)
    if len(corners) != 4:
        return None
    points = [(x * image_width, y * image_height) for x, y in corners]
    center_x = sum(point[0] for point in points) / 4
    center_y = sum(point[1] for point in points) / 4

    first = (points[1][0] - points[0][0], points[1][1] - points[0][1])
    last = (points[3][0] - points[0][0], points[3][1] - points[0][1])
    cross = first[0] * last[1] - first[1] * last[0]
    width_vector, height_vector = (first, last) if cross >= 0 else (last, first)
    width = math.hypot(*width_vector)
    height = math.hypot(*height_vector)
    if width <= 0 or height <= 0:
        return None
    angle = math.atan2(width_vector[1], width_vector[0]) % (2 * math.pi)
    return center_x, center_y, width, height, angle


def export_rolabelimg_obb(
    image_annotations: list[ImageAnnotation],
    output_dir: Path | str,
    only_confirmed: bool = False,
    task_type: str = "obb",
) -> None:
    """Export one roLabelImg ``robndbox`` XML file per annotated image."""
    if task_type != "obb":
        raise ValueError("roLabelImg OBB 导出仅适用于 OBB 项目")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_annotation in image_annotations:
        image_width, image_height = image_annotation.image_size
        if image_width <= 0 or image_height <= 0:
            raise ValueError(
                f"roLabelImg OBB 导出需要有效的图像尺寸: "
                f"{image_annotation.image_path}"
            )

        relative_image = Path(image_annotation.image_path)
        if relative_image.is_absolute():
            relative_image = Path(relative_image.name)
        elif relative_image.parts and relative_image.parts[0].casefold() == "images":
            relative_image = Path(*relative_image.parts[1:])
        output_path = output_dir / relative_image.with_suffix(".xml")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = relative_image.parent.name
        ET.SubElement(root, "filename").text = relative_image.name
        ET.SubElement(root, "path").text = image_annotation.image_path
        source = ET.SubElement(root, "source")
        ET.SubElement(source, "database").text = "Unknown"
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(image_width)
        ET.SubElement(size, "height").text = str(image_height)
        ET.SubElement(size, "depth").text = "3"
        ET.SubElement(root, "segmented").text = "0"

        for annotation in image_annotation.annotations:
            if only_confirmed and not annotation.confirmed:
                continue
            values = _robndbox_values(annotation, image_width, image_height)
            if values is None:
                continue
            center_x, center_y, width, height, angle = values
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "type").text = "robndbox"
            ET.SubElement(obj, "name").text = annotation.class_name
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            rotated = ET.SubElement(obj, "robndbox")
            for tag, value in (
                ("cx", center_x),
                ("cy", center_y),
                ("w", width),
                ("h", height),
                ("angle", angle),
            ):
                ET.SubElement(rotated, tag).text = f"{value:.6f}"

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(output_path, encoding="utf-8", xml_declaration=True)


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
