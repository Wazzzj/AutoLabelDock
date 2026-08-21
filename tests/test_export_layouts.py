import json

from src.controllers.project import ProjectController
from src.core.annotation import Annotation, ImageAnnotation
from src.core.config import AppConfig
from src.core.formats import get_export_registry
from src.core.label_io import save_annotation
from src.core.project import ProjectManager


def _controller(tmp_path):
    return ProjectController(AppConfig(), tmp_path / "app-config.json", None)


def test_sidecar_export_places_image_and_json_together(tmp_path):
    project = ProjectManager.create(
        tmp_path / "detect-project",
        "detect-project",
        classes=["defect"],
        task_type="detect",
    )
    image_path = project.image_root() / "batch_a" / "sample.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"image")
    save_annotation(
        ImageAnnotation(
            image_path=str(image_path),
            image_size=(100, 80),
            annotations=[
                Annotation("defect", 0, bbox=(0.5, 0.5, 0.4, 0.4))
            ],
        ),
        project.label_path_for(image_path),
    )

    output_root = tmp_path / "labelme-export"
    controller = _controller(tmp_path)
    image_count, annotations, _ = controller._prepare_format_export(
        project,
        output_root,
        "",
        fmt="labelme",
    )
    export_output = controller._format_export_output(output_root, "labelme")
    get_export_registry().export("labelme", annotations, export_output)

    image_output = output_root / "batch_a" / "sample.jpg"
    json_output = output_root / "batch_a" / "sample.json"
    assert image_count == 1
    assert image_output.exists()
    assert json_output.exists()
    assert not (output_root / "images").exists()
    assert not (output_root / "labels").exists()
    assert json.loads(json_output.read_text(encoding="utf-8"))["imagePath"] == "sample.jpg"


def test_format_output_directories_follow_native_layouts(tmp_path):
    controller = _controller(tmp_path)
    root = tmp_path / "export"

    for fmt in (
        "labelme",
        "X-AnyLabeling-Detect",
        "iSAT",
        "RoLabelImg-OBB",
        "X-AnyLabeling-OBB",
        "ImageFolder",
    ):
        assert controller._format_export_output(root, fmt) == root
    assert controller._format_export_output(root, "YOLO") == root
    assert controller._format_export_output(root, "COCO") == root / "annotations"
    assert controller._format_export_output(root, "CSV") == root / "labels"


def test_coco_export_uses_images_and_annotations_directories(tmp_path):
    project = ProjectManager.create(
        tmp_path / "coco-project",
        "coco-project",
        classes=["part"],
        task_type="detect",
    )
    image_path = project.image_root() / "sample.jpg"
    image_path.write_bytes(b"image")
    save_annotation(
        ImageAnnotation(
            image_path=str(image_path),
            image_size=(100, 80),
            annotations=[Annotation("part", 0, bbox=(0.5, 0.5, 0.2, 0.2))],
        ),
        project.label_path_for(image_path),
    )

    output_root = tmp_path / "coco-export"
    controller = _controller(tmp_path)
    _, annotations, _ = controller._prepare_format_export(
        project,
        output_root,
        "",
        fmt="COCO",
    )
    get_export_registry().export(
        "COCO",
        annotations,
        controller._format_export_output(output_root, "COCO"),
        classes=project.config.classes,
    )

    assert (output_root / "images" / "sample.jpg").exists()
    coco_path = output_root / "annotations" / "coco.json"
    assert coco_path.exists()
    assert json.loads(coco_path.read_text(encoding="utf-8"))["images"][0][
        "file_name"
    ] == "images/sample.jpg"
    assert not (output_root / "labels").exists()


def test_imagefolder_export_only_creates_class_directories(tmp_path):
    project = ProjectManager.create(
        tmp_path / "classify-project",
        "classify-project",
        classes=["NG"],
        task_type="classify",
    )
    image_path = project.image_root() / "sample.jpg"
    image_path.write_bytes(b"image")
    save_annotation(
        ImageAnnotation(
            image_path=str(image_path),
            image_size=(100, 80),
            image_tags=["NG"],
        ),
        project.label_path_for(image_path),
    )

    output_root = tmp_path / "imagefolder-export"
    controller = _controller(tmp_path)
    image_count, _, source_annotations = controller._prepare_format_export(
        project,
        output_root,
        "",
        fmt="ImageFolder",
    )
    get_export_registry().export(
        "ImageFolder",
        source_annotations,
        controller._format_export_output(output_root, "ImageFolder"),
        source_image_dir=project.image_root(),
        task_type="classify",
    )

    assert image_count == 1
    assert (output_root / "NG" / "sample.jpg").exists()
    assert not (output_root / "images").exists()
    assert not (output_root / "labels").exists()
