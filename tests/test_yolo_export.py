import yaml

from src.controllers.project import ProjectController
from src.core.annotation import Annotation, ImageAnnotation
from src.core.config import AppConfig
from src.core.formats import get_export_registry
from src.core.formats.yolo import (
    export_yolo_detection,
    export_yolo_segment,
    import_yolo_auto,
)
from src.core.label_io import save_annotation
from src.core.project import ProjectManager


def test_yolo_export_preserves_subdirectory_label_paths_and_data_yaml(tmp_path):
    export_yolo_detection(
        [
            ImageAnnotation(
                image_path="images/batch_a/sample.jpg",
                image_size=(100, 100),
                annotations=[
                    Annotation(
                        class_name="dog",
                        class_id=1,
                        bbox=(0.5, 0.5, 0.25, 0.25),
                        confirmed=True,
                    )
                ],
            )
        ],
        tmp_path,
        classes=["cat", "dog"],
        only_confirmed=True,
    )

    label_path = tmp_path / "labels" / "batch_a" / "sample.txt"
    assert label_path.exists()
    assert not (tmp_path / "labels" / "sample.txt").exists()
    assert label_path.read_text(encoding="utf-8").split()[0] == "0"

    data = yaml.safe_load((tmp_path / "data.yaml").read_text(encoding="utf-8"))
    assert data["path"] == "."
    assert data["train"] == "images"
    assert data["val"] == "images"
    assert data["nc"] == 1
    assert data["names"] == ["dog"]

    imported = import_yolo_auto(tmp_path / "labels")
    assert len(imported) == 1
    assert imported[0].annotations[0].class_name == "dog"


def test_yolo_confirmed_export_skips_images_without_confirmed_boxes(tmp_path):
    project = ProjectManager.create(
        tmp_path / "project",
        "project",
        classes=["cat", "dog"],
        task_type="detect",
    )
    confirmed_path = project.image_root() / "batch_a" / "confirmed.jpg"
    pending_path = project.image_root() / "batch_a" / "pending.jpg"
    unlabeled_path = project.image_root() / "batch_a" / "unlabeled.jpg"
    for path in (confirmed_path, pending_path, unlabeled_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake image bytes")

    save_annotation(
        ImageAnnotation(
            image_path=str(confirmed_path),
            image_size=(100, 100),
            annotations=[
                Annotation(
                    class_name="cat",
                    class_id=0,
                    bbox=(0.5, 0.5, 0.2, 0.2),
                    confirmed=True,
                )
            ],
        ),
        project.label_path_for(confirmed_path),
    )
    save_annotation(
        ImageAnnotation(
            image_path=str(pending_path),
            image_size=(100, 100),
            annotations=[
                Annotation(
                    class_name="dog",
                    class_id=1,
                    bbox=(0.5, 0.5, 0.2, 0.2),
                    confirmed=False,
                )
            ],
        ),
        project.label_path_for(pending_path),
    )

    controller = ProjectController(AppConfig(), tmp_path / "config.json", None)
    image_count, annotations, source_annotations = controller._prepare_format_export(
        project,
        tmp_path / "export",
        "",
        fmt="YOLO",
        only_confirmed=True,
    )

    assert image_count == 1
    assert len(annotations) == 1
    assert len(source_annotations) == 1
    assert annotations[0].image_path == "images/batch_a/confirmed.jpg"
    assert (tmp_path / "export" / "images" / "batch_a" / "confirmed.jpg").exists()
    assert not (tmp_path / "export" / "images" / "batch_a" / "pending.jpg").exists()
    assert not (tmp_path / "export" / "images" / "batch_a" / "unlabeled.jpg").exists()


def test_yolo_segment_export_writes_polygon_coordinates(tmp_path):
    export_yolo_segment(
        [
            ImageAnnotation(
                image_path="images/sample.jpg",
                image_size=(100, 100),
                annotations=[
                    Annotation(
                        class_name="object",
                        class_id=0,
                        bbox=(0.45, 0.55, 0.7, 0.7),
                        polygon=[(0.1, 0.2), (0.8, 0.3), (0.6, 0.9)],
                        confirmed=True,
                    )
                ],
            )
        ],
        tmp_path,
        classes=["object"],
    )

    fields = (tmp_path / "labels" / "sample.txt").read_text(encoding="utf-8").split()
    assert fields == [
        "0",
        "0.100000",
        "0.200000",
        "0.800000",
        "0.300000",
        "0.600000",
        "0.900000",
    ]


def test_yolo_registry_uses_segment_export_for_segment_project(tmp_path):
    annotations = [
        ImageAnnotation(
            image_path="images/sample.jpg",
            image_size=(100, 100),
            annotations=[
                Annotation(
                    class_name="object",
                    class_id=0,
                    polygon=[(0.1, 0.2), (0.8, 0.3), (0.6, 0.9)],
                    confirmed=True,
                )
            ],
        )
    ]

    get_export_registry().export(
        "YOLO",
        annotations,
        tmp_path,
        classes=["object"],
        task_type="segment",
    )

    assert len((tmp_path / "labels" / "sample.txt").read_text(encoding="utf-8").split()) == 7


def test_yolo_confirmed_segment_export_keeps_polygon_only_images(tmp_path):
    project = ProjectManager.create(
        tmp_path / "project",
        "project",
        classes=["object"],
        task_type="segment",
    )
    image_path = project.image_root() / "sample.jpg"
    image_path.write_bytes(b"fake image bytes")
    save_annotation(
        ImageAnnotation(
            image_path=str(image_path),
            image_size=(100, 100),
            annotations=[
                Annotation(
                    class_name="object",
                    class_id=0,
                    polygon=[(0.1, 0.2), (0.8, 0.3), (0.6, 0.9)],
                    confirmed=True,
                )
            ],
        ),
        project.label_path_for(image_path),
    )

    controller = ProjectController(AppConfig(), tmp_path / "config.json", None)
    image_count, annotations, _ = controller._prepare_format_export(
        project,
        tmp_path / "export",
        "",
        fmt="YOLO",
        only_confirmed=True,
    )

    assert image_count == 1
    assert len(annotations) == 1
