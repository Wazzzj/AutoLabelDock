import json

import pytest

from src.core.formats import get_export_registry, get_import_registry
from src.controllers.project import ProjectController
from src.core.annotation import Annotation, ImageAnnotation
from src.core.formats.xanylabeling_detect import (
    export_xanylabeling_detect,
    import_xanylabeling_detect,
)


def _write_annotation(path, points, *, label="defect", score=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": "2.5.4",
                "shapes": [
                    {
                        "label": label,
                        "score": score,
                        "points": points,
                        "shape_type": "rectangle",
                    }
                ],
                "imagePath": f"../images/{path.stem}.png",
                "imageWidth": 200,
                "imageHeight": 100,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_xanylabeling_detect_imports_two_and_four_point_rectangles(tmp_path):
    _write_annotation(tmp_path / "jsons" / "two.json", [[20, 10], [120, 70]])
    _write_annotation(
        tmp_path / "jsons" / "four.json",
        [[30, 20], [150, 20], [150, 80], [30, 80]],
        score=0.8,
    )

    imported = import_xanylabeling_detect(tmp_path)

    assert [item.image_path for item in imported] == ["four.png", "two.png"]
    four, two = imported
    assert four.annotations[0].bbox == pytest.approx((0.45, 0.5, 0.6, 0.6))
    assert four.annotations[0].confidence == pytest.approx(0.8)
    assert two.annotations[0].bbox == pytest.approx((0.35, 0.4, 0.5, 0.6))


def test_xanylabeling_detect_ignores_empty_and_non_rectangle_json(tmp_path):
    (tmp_path / "empty.json").write_text(
        json.dumps(
            {
                "shapes": [],
                "imagePath": "empty.png",
                "imageWidth": 100,
                "imageHeight": 100,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "rotation.json").write_text(
        json.dumps(
            {
                "shapes": [
                    {
                        "label": "obb",
                        "shape_type": "rotation",
                        "points": [[1, 1], [9, 1], [9, 9], [1, 9]],
                    }
                ],
                "imagePath": "rotation.png",
                "imageWidth": 10,
                "imageHeight": 10,
            }
        ),
        encoding="utf-8",
    )

    assert import_xanylabeling_detect(tmp_path) == []


def test_xanylabeling_detect_is_registered_as_separate_import_format():
    info = get_import_registry().get("X-AnyLabeling-Detect")

    assert info is not None
    assert info.label == "X-AnyLabeling Detect (json)"


def test_xanylabeling_detect_export_writes_four_point_rectangle(tmp_path):
    export_xanylabeling_detect(
        [
            ImageAnnotation(
                image_path="images/batch/sample.png",
                image_size=(200, 100),
                annotations=[
                    Annotation(
                        class_name="defect",
                        class_id=0,
                        bbox=(0.5, 0.5, 0.4, 0.6),
                        confidence=0.75,
                    )
                ],
            )
        ],
        tmp_path / "labels",
        task_type="detect",
    )

    data = json.loads(
        (tmp_path / "labels" / "batch" / "sample.json").read_text(
            encoding="utf-8"
        )
    )
    assert data["imagePath"] == "../../images/batch/sample.png"
    assert data["imageWidth"] == 200
    assert data["imageHeight"] == 100
    assert data["shapes"] == [
        {
            "label": "defect",
            "score": 0.75,
            "points": [[60.0, 20.0], [140.0, 20.0], [140.0, 80.0], [60.0, 80.0]],
            "group_id": None,
            "description": "",
            "difficult": False,
            "shape_type": "rectangle",
            "flags": {},
            "attributes": {},
            "kie_linking": [],
        }
    ]


def test_xanylabeling_detect_export_is_detect_only_and_filters_pending(tmp_path):
    registry = get_export_registry()
    exporter = registry.get("X-AnyLabeling-Detect")
    assert exporter is not None
    assert exporter.task_types == frozenset({"detect"})

    with pytest.raises(ValueError):
        registry.export(
            "X-AnyLabeling-Detect",
            [],
            tmp_path,
            task_type="segment",
        )

    image_annotation = ImageAnnotation(
        image_path="sample.png",
        image_size=(100, 100),
        annotations=[
            Annotation(
                class_name="pending",
                class_id=0,
                bbox=(0.5, 0.5, 0.2, 0.2),
                confirmed=False,
            )
        ],
    )
    assert not ProjectController._has_format_exportable_annotations(
        image_annotation,
        "X-AnyLabeling-Detect",
        only_confirmed=True,
        task_type="detect",
    )
