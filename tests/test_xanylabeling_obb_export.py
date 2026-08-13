import json
import math

import pytest

from src.controllers.project import ProjectController
from src.core.annotation import Annotation, ImageAnnotation
from src.core.formats import get_export_registry
from src.core.formats.xanylabeling_obb import export_xanylabeling_obb
from src.ui.icons import _SVGS


def test_xanylabeling_obb_export_writes_rotation_shape_and_relative_image(tmp_path):
    labels_dir = tmp_path / "labels"
    corners = [(0.1, 0.25), (0.8, 0.125), (0.9, 0.75), (0.2, 0.875)]

    export_xanylabeling_obb(
        [
            ImageAnnotation(
                image_path="images/batch_a/sample.jpg",
                image_size=(100, 80),
                annotations=[
                    Annotation(
                        class_name="label",
                        class_id=0,
                        polygon=corners,
                        confirmed=True,
                    )
                ],
            )
        ],
        labels_dir,
        only_confirmed=True,
    )

    data = json.loads(
        (labels_dir / "batch_a" / "sample.json").read_text(encoding="utf-8")
    )
    assert data["imagePath"] == "../../images/batch_a/sample.jpg"
    assert data["imageWidth"] == 100
    assert data["imageHeight"] == 80
    assert data["imageData"] is None
    assert len(data["shapes"]) == 1
    shape = data["shapes"][0]
    assert shape["label"] == "label"
    assert shape["shape_type"] == "rotation"
    assert shape["points"] == [[10.0, 20.0], [80.0, 10.0], [90.0, 60.0], [20.0, 70.0]]
    assert shape["description"] is None
    assert shape["direction"] == pytest.approx(
        math.atan2(-10.0, 70.0) % (2 * math.pi)
    )


def test_xanylabeling_obb_export_converts_bbox_and_filters_pending(tmp_path):
    export_xanylabeling_obb(
        [
            ImageAnnotation(
                image_path="images/sample.png",
                image_size=(200, 100),
                annotations=[
                    Annotation(
                        class_name="confirmed",
                        class_id=0,
                        bbox=(0.5, 0.5, 0.4, 0.2),
                        confirmed=True,
                    ),
                    Annotation(
                        class_name="pending",
                        class_id=1,
                        bbox=(0.5, 0.5, 0.2, 0.2),
                        confirmed=False,
                    ),
                ],
            )
        ],
        tmp_path,
        only_confirmed=True,
    )

    data = json.loads((tmp_path / "sample.json").read_text(encoding="utf-8"))
    assert [shape["label"] for shape in data["shapes"]] == ["confirmed"]
    assert data["shapes"][0]["points"] == [
        [60.0, 40.0],
        [140.0, 40.0],
        [140.0, 60.0],
        [60.0, 60.0],
    ]


def test_xanylabeling_obb_registry_rejects_non_obb_projects(tmp_path):
    registry = get_export_registry()
    exporter = registry.get("X-AnyLabeling-OBB")

    assert exporter is not None
    assert exporter.task_types == frozenset({"obb"})
    with pytest.raises(ValueError, match="不适用于 detect 项目"):
        registry.export(
            "X-AnyLabeling-OBB",
            [],
            tmp_path,
            task_type="detect",
        )


def test_xanylabeling_confirmed_exportability_requires_obb_geometry():
    image_annotation = ImageAnnotation(
        image_path="sample.jpg",
        image_size=(100, 100),
        annotations=[
            Annotation(
                class_name="label",
                class_id=0,
                polygon=[(0.1, 0.1), (0.8, 0.1), (0.5, 0.8)],
                confirmed=True,
            )
        ],
    )
    assert not ProjectController._has_format_exportable_annotations(
        image_annotation,
        "X-AnyLabeling-OBB",
        only_confirmed=True,
        task_type="obb",
    )


def test_export_icon_uses_upward_arrow():
    svg = _SVGS["export"]
    assert 'points="7 8 12 3 17 8"' in svg
    assert 'points="7 10 12 15 17 10"' not in svg
