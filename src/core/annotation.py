"""Annotation data models for AutoLabel Dock."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class Keypoint:
    """A single keypoint with normalized coordinates."""

    x: float
    y: float
    visible: int  # 0=invisible, 1=occluded, 2=visible
    label: str

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "visible": self.visible, "label": self.label}

    @classmethod
    def from_dict(cls, d: dict) -> Keypoint:
        return cls(x=d["x"], y=d["y"], visible=d["visible"], label=d["label"])

    def clamp(self) -> None:
        """Clamp coordinates to [0, 1]."""
        self.x = max(0.0, min(1.0, self.x))
        self.y = max(0.0, min(1.0, self.y))


@dataclass
class Annotation:
    """A single annotation (bbox, polygon, keypoints, or a combination)."""

    class_name: str
    class_id: int
    bbox: tuple[float, float, float, float] | None = None  # (cx, cy, w, h) normalized
    polygon: list[tuple[float, float]] = field(default_factory=list)
    keypoints: list[Keypoint] = field(default_factory=list)
    confidence: float = 1.0
    confirmed: bool = True
    source: str = "manual"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "bbox": list(self.bbox) if self.bbox else None,
            "polygon": [[x, y] for x, y in self.polygon],
            "keypoints": [kp.to_dict() for kp in self.keypoints],
            "confidence": self.confidence,
            "confirmed": self.confirmed,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Annotation:
        bbox = tuple(d["bbox"]) if d.get("bbox") else None
        polygon = [tuple(point) for point in d.get("polygon", [])]
        keypoints = [Keypoint.from_dict(kp) for kp in d.get("keypoints", [])]
        return cls(
            id=d["id"],
            class_name=d["class_name"],
            class_id=d["class_id"],
            bbox=bbox,
            polygon=polygon,
            keypoints=keypoints,
            confidence=d.get("confidence", 1.0),
            confirmed=d.get("confirmed", True),
            source=d.get("source", "manual"),
        )

    def clamp(self) -> None:
        """Clamp bbox and keypoints to [0, 1] image bounds."""
        if self.bbox:
            cx, cy, w, h = self.bbox
            x1 = max(0.0, cx - w / 2)
            y1 = max(0.0, cy - h / 2)
            x2 = min(1.0, cx + w / 2)
            y2 = min(1.0, cy + h / 2)
            self.bbox = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
        if self.polygon:
            self.polygon = [
                (max(0.0, min(1.0, x)), max(0.0, min(1.0, y)))
                for x, y in self.polygon
            ]
        for kp in self.keypoints:
            kp.clamp()


def annotation_geometry(
    annotation: Annotation,
) -> tuple[float, float, float, float, float] | None:
    """Return normalized width, height, area and center coordinates.

    Polygon/OBB annotations use their actual polygon area (shoelace formula)
    and axis-aligned bounds for width, height and center. Plain detection boxes
    use their normalized bbox directly.
    """
    if annotation.polygon:
        points = annotation.polygon
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        area = abs(sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
        )) / 2.0
        return (
            width,
            height,
            area,
            (min(xs) + max(xs)) / 2,
            (min(ys) + max(ys)) / 2,
        )
    if annotation.bbox is not None:
        center_x, center_y, width, height = annotation.bbox
        return width, height, width * height, center_x, center_y
    return None


def annotation_center(annotation: Annotation) -> tuple[float, float] | None:
    """Return the normalized center used for ROI control.

    Detection boxes use their explicit center. OBB and instance-segmentation
    polygons use the shoelace area centroid, which follows the actual geometry
    instead of the center of its axis-aligned bounds.
    """
    if annotation.polygon and len(annotation.polygon) >= 3:
        points = annotation.polygon
        cross_sum = 0.0
        centroid_x_sum = 0.0
        centroid_y_sum = 0.0
        for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
            cross = x1 * y2 - x2 * y1
            cross_sum += cross
            centroid_x_sum += (x1 + x2) * cross
            centroid_y_sum += (y1 + y2) * cross
        if abs(cross_sum) > 1e-12:
            return (
                centroid_x_sum / (3.0 * cross_sum),
                centroid_y_sum / (3.0 * cross_sum),
            )
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    if annotation.bbox is not None:
        return annotation.bbox[0], annotation.bbox[1]
    return None


def annotation_pixel_geometry(
    annotation: Annotation,
    image_size: tuple[int, int],
) -> tuple[float, float, float, float, float] | None:
    """Return width, height, area and center coordinates in image pixels."""
    geometry = annotation_geometry(annotation)
    if geometry is None:
        return None
    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        return None
    width, height, area, center_x, center_y = geometry
    return (
        width * image_width,
        height * image_height,
        area * image_width * image_height,
        center_x * image_width,
        center_y * image_height,
    )


def annotation_area_text(
    annotation: Annotation,
    image_size: tuple[int, int] | None = None,
    *,
    include_pixels: bool = True,
) -> str:
    """Format an annotation area in pixels for canvas/preview labels.

    ``include_pixels`` is retained for call-site compatibility. Pixel area is
    now the canonical display and percentages are intentionally omitted.
    """
    if image_size is None:
        return ""
    geometry = annotation_pixel_geometry(annotation, image_size)
    if geometry is None:
        return ""
    pixel_area = round(geometry[2])
    return f"面积 {pixel_area} px²"


def annotation_display_label(
    annotation: Annotation,
    image_size: tuple[int, int] | None = None,
    *,
    include_pixels: bool = True,
) -> str:
    """Return the class and area label shared by canvas and preview."""
    area_text = annotation_area_text(
        annotation,
        image_size,
        include_pixels=include_pixels,
    )
    return (
        f"{annotation.class_name} | {area_text}"
        if area_text
        else annotation.class_name
    )


@dataclass
class ImageAnnotation:
    """All annotations for a single image."""

    image_path: str
    image_size: tuple[int, int]  # (width, height)
    annotations: list[Annotation] = field(default_factory=list)
    image_tags: list[str] = field(default_factory=list)
    image_tags_confirmed: bool = True
    image_tags_source: str = "manual"
    # Free-form user tags for dataset organization / filtering. Distinct from
    # image_tags (which stores the classify task's class label).
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "image_size": list(self.image_size),
            "image_tags": self.image_tags,
            "image_tags_confirmed": self.image_tags_confirmed,
            "image_tags_source": self.image_tags_source,
            "tags": self.tags,
            "annotations": [ann.to_dict() for ann in self.annotations],
        }

    @classmethod
    def from_dict(cls, d: dict) -> ImageAnnotation:
        return cls(
            image_path=d["image_path"],
            image_size=tuple(d["image_size"]),
            annotations=[Annotation.from_dict(a) for a in d.get("annotations", [])],
            image_tags=d.get("image_tags", []),
            image_tags_confirmed=d.get("image_tags_confirmed", True),
            image_tags_source=d.get("image_tags_source", "manual"),
            tags=list(d.get("tags", [])),
        )

    @property
    def confirmed_count(self) -> int:

        return sum(1 for a in self.annotations if a.confirmed)

    @property
    def unconfirmed_count(self) -> int:
        return sum(1 for a in self.annotations if not a.confirmed)

    @property
    def status(self) -> str:
        """Return 'unlabeled', 'confirmed', or 'pending'."""
        if self.image_tags:
            return "confirmed" if self.image_tags_confirmed else "pending"
        if not self.annotations:
            return "unlabeled"
        if all(a.confirmed for a in self.annotations):
            return "confirmed"
        return "pending"


def compute_iou(bbox1: tuple[float, float, float, float],
                bbox2: tuple[float, float, float, float]) -> float:
    """Compute IoU between two (cx, cy, w, h) normalized bboxes."""
    cx1, cy1, w1, h1 = bbox1
    cx2, cy2, w2, h2 = bbox2
    # Convert to x1, y1, x2, y2
    ax1, ay1, ax2, ay2 = cx1 - w1 / 2, cy1 - h1 / 2, cx1 + w1 / 2, cy1 + h1 / 2
    bx1, by1, bx2, by2 = cx2 - w2 / 2, cy2 - h2 / 2, cx2 + w2 / 2, cy2 + h2 / 2
    # Intersection
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def retain_highest_confidence_roi(
    annotations: list[Annotation],
    enabled: bool = False,
) -> list[Annotation]:
    """Keep only the highest-confidence predicted ROI when the rule is enabled.

    The original annotation objects are preserved, and ties keep the first ROI
    returned by the model so the result remains deterministic.
    """
    if not enabled or len(annotations) <= 1:
        return list(annotations)
    return [max(annotations, key=lambda annotation: annotation.confidence)]


def find_conflicts(
    existing: list[Annotation],
    predictions: list[Annotation],
    iou_threshold: float = 0.5,
) -> tuple[list[tuple[Annotation, Annotation]], list[Annotation]]:
    """Match predictions against confirmed same-class existing annotations by IoU.

    Returns (conflict_pairs, non_conflict_predictions).
    Each existing annotation is matched to at most one prediction (greedy, highest IoU first).
    """
    confirmed = [a for a in existing if a.confirmed and a.bbox]
    matched_existing: set[str] = set()
    conflicts: list[tuple[Annotation, Annotation]] = []
    non_conflicts: list[Annotation] = []

    for pred in predictions:
        if not pred.bbox:
            non_conflicts.append(pred)
            continue
        best_iou = 0.0
        best_match: Annotation | None = None
        for ex in confirmed:
            if ex.id in matched_existing:
                continue
            if ex.class_name != pred.class_name:
                continue
            iou = compute_iou(ex.bbox, pred.bbox)
            if iou > best_iou:
                best_iou = iou
                best_match = ex
        if best_match is not None and best_iou >= iou_threshold:
            matched_existing.add(best_match.id)
            conflicts.append((best_match, pred))
        else:
            non_conflicts.append(pred)

    return conflicts, non_conflicts
