import json

from src.core.annotation import Annotation, ImageAnnotation
from src.core.formats.labelme import export_labelme, import_labelme_file


def test_labelme_export_writes_valid_relative_image_path_and_polygon(tmp_path):
    labels_dir = tmp_path / "labels"

    export_labelme(
        [
            ImageAnnotation(
                image_path="images/batch_a/sample.jpg",
                image_size=(100, 80),
                annotations=[
                    Annotation(
                        class_name="part",
                        class_id=0,
                        polygon=[(0.1, 0.2), (0.8, 0.2), (0.7, 0.9)],
                        confirmed=True,
                    )
                ],
            )
        ],
        labels_dir,
        only_confirmed=True,
    )

    json_path = labels_dir / "batch_a" / "sample.json"
    assert json_path.exists()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["imagePath"] == "../../images/batch_a/sample.jpg"
    assert data["shapes"] == [
        {
            "label": "part",
            "shape_type": "polygon",
            "points": [[10.0, 16.0], [80.0, 16.0], [70.0, 72.0]],
            "group_id": None,
            "flags": {},
        }
    ]

    imported = import_labelme_file(json_path)
    assert imported is not None
    assert imported.annotations[0].class_name == "part"
    assert imported.annotations[0].polygon == [(0.1, 0.2), (0.8, 0.2), (0.7, 0.9)]


def test_labelme_export_segment_polygon_does_not_emit_bbox_rectangle(tmp_path):
    labels_dir = tmp_path / "labels"

    export_labelme(
        [
            ImageAnnotation(
                image_path="images/sample.jpg",
                image_size=(100, 80),
                annotations=[
                    Annotation(
                        class_name="glue",
                        class_id=0,
                        bbox=(0.45, 0.5, 0.7, 0.6),
                        polygon=[(0.1, 0.2), (0.8, 0.2), (0.7, 0.9)],
                        confirmed=True,
                    )
                ],
            )
        ],
        labels_dir,
        only_confirmed=True,
    )

    data = json.loads((labels_dir / "sample.json").read_text(encoding="utf-8"))
    assert [shape["shape_type"] for shape in data["shapes"]] == ["polygon"]


def test_labelme_import_reads_x_anylabeling_rotation_as_obb_polygon(tmp_path):
    json_path = tmp_path / "sample.json"
    json_path.write_text(
        json.dumps(
            {
                "version": "2.5.4",
                "shapes": [
                    {
                        "label": "label",
                        "shape_type": "rotation",
                        "points": [[10, 20], [80, 10], [90, 60], [20, 70]],
                        "group_id": None,
                        "direction": 0.1,
                    }
                ],
                "imagePath": "sample.jpeg",
                "imageWidth": 100,
                "imageHeight": 80,
            }
        ),
        encoding="utf-8",
    )

    imported = import_labelme_file(json_path)

    assert imported is not None
    assert len(imported.annotations) == 1
    annotation = imported.annotations[0]
    assert annotation.class_name == "label"
    assert annotation.confirmed is True
    assert annotation.polygon == [
        (0.1, 0.25),
        (0.8, 0.125),
        (0.9, 0.75),
        (0.2, 0.875),
    ]
    assert annotation.bbox == (0.5, 0.5, 0.8, 0.75)
