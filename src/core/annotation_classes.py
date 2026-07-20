"""Helpers for deriving class names from saved annotations."""
from __future__ import annotations

from pathlib import Path

from src.core.annotation import ImageAnnotation
from src.core.label_io import load_annotation


def classes_in_annotation(annotation: ImageAnnotation | None) -> list[str]:
    """Return unique class names present in one annotation record."""
    if annotation is None:
        return []
    seen: set[str] = set()
    classes: list[str] = []
    for name in list(annotation.image_tags) + [
        ann.class_name for ann in annotation.annotations
    ]:
        if not name or name in seen:
            continue
        seen.add(name)
        classes.append(name)
    return classes


def merged_project_annotation_classes(project, images: list[Path] | None = None) -> list[str]:
    """Return project classes plus classes discovered in saved labels.

    Project class order is preserved. Classes that exist only in labels are
    appended alphabetically for stable filter menus.
    """
    result: list[str] = []
    seen: set[str] = set()
    for name in getattr(project.config, "classes", []):
        if not name or name in seen:
            continue
        seen.add(name)
        result.append(name)

    discovered: set[str] = set()
    for image_path in images if images is not None else project.list_images():
        annotation = load_annotation(project.label_path_for(image_path))
        for name in classes_in_annotation(annotation):
            if name not in seen:
                discovered.add(name)

    for name in sorted(discovered):
        seen.add(name)
        result.append(name)
    return result
