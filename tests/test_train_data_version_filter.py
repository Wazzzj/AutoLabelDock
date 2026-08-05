from src.core.annotation import Annotation, ImageAnnotation
from src.core.label_io import save_annotation
from src.core.project import ProjectManager
from src.engine.dataset import DatasetPreparer, count_selected_training_images


def _add_confirmed_image(project, folder: str, name: str):
    path = project.image_root() / folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image")
    save_annotation(
        ImageAnnotation(
            image_path=str(path),
            image_size=(100, 100),
            annotations=[
                Annotation(
                    class_name="part",
                    class_id=0,
                    bbox=(0.5, 0.5, 0.2, 0.2),
                    confirmed=True,
                )
            ],
        ),
        project.label_path_for(path),
    )
    return path


def test_training_count_filters_by_data_version(tmp_path):
    project = ProjectManager.create(
        tmp_path / "project", "demo", classes=["part"]
    )
    _add_confirmed_image(project, "v1", "first.jpg")
    _add_confirmed_image(project, "v2", "second.jpg")

    assert count_selected_training_images(project, data_folder="v1") == 1
    assert count_selected_training_images(project, data_folder="v2") == 1
    assert count_selected_training_images(project, data_folder="") == 2


def test_dataset_preparer_exports_only_selected_data_version(tmp_path):
    project = ProjectManager.create(
        tmp_path / "project", "demo", classes=["part"]
    )
    _add_confirmed_image(project, "v1", "first.jpg")
    _add_confirmed_image(project, "v2", "second.jpg")

    DatasetPreparer(project).prepare(
        tmp_path / "dataset",
        val_ratio=0,
        data_folder="v2",
    )

    exported = {
        path.name
        for path in (tmp_path / "dataset" / "train" / "images").iterdir()
    }
    assert exported == {"second.jpg"}
