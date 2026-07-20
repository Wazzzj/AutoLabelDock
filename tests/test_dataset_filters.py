import yaml

from src.core.annotation import Annotation, ImageAnnotation
from src.core.label_io import save_annotation
from src.core.project import ProjectManager
from src.core.tags import TagFilter
from src.engine.dataset import DatasetPreparer, count_selected_training_images


def test_detect_class_filter_exports_single_class_dataset(tmp_path):
    project = ProjectManager.create(
        tmp_path / "project",
        "project",
        classes=["cat", "dog"],
        task_type="detect",
    )
    image_path = project.image_root() / "sample.jpg"
    image_path.write_bytes(b"fake image bytes")

    save_annotation(
        ImageAnnotation(
            image_path=str(image_path),
            image_size=(100, 100),
            annotations=[
                Annotation(
                    class_name="cat",
                    class_id=0,
                    bbox=(0.2, 0.2, 0.1, 0.1),
                    confirmed=True,
                ),
                Annotation(
                    class_name="dog",
                    class_id=1,
                    bbox=(0.6, 0.6, 0.2, 0.2),
                    confirmed=True,
                ),
            ],
        ),
        project.label_path_for(image_path),
    )

    data_yaml = DatasetPreparer(project).prepare(
        tmp_path / "dataset",
        task="detect",
        val_ratio=0,
        status_filter="confirmed",
        class_filter="dog",
    )

    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    assert data["nc"] == 1
    assert data["names"] == ["dog"]

    label_text = (tmp_path / "dataset" / "train" / "labels" / "sample.txt").read_text(
        encoding="utf-8"
    )
    lines = [line for line in label_text.splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0].split()[0] == "0"


def test_count_selected_training_images_counts_images_not_annotations(tmp_path):
    project = ProjectManager.create(
        tmp_path / "project",
        "project",
        classes=["cat", "dog"],
        task_type="detect",
    )
    image_path = project.image_root() / "sample.jpg"
    image_path.write_bytes(b"fake image bytes")

    save_annotation(
        ImageAnnotation(
            image_path=str(image_path),
            image_size=(100, 100),
            tags=["train"],
            annotations=[
                Annotation(
                    class_name="dog",
                    class_id=1,
                    bbox=(0.2, 0.2, 0.1, 0.1),
                    confirmed=True,
                ),
                Annotation(
                    class_name="dog",
                    class_id=1,
                    bbox=(0.6, 0.6, 0.2, 0.2),
                    confirmed=True,
                ),
            ],
        ),
        project.label_path_for(image_path),
    )

    assert count_selected_training_images(
        project,
        task="detect",
        status_filter="confirmed",
        class_filter="dog",
        tag_filter=TagFilter(includes=("train",)),
    ) == 1


def test_count_selected_training_images_classify_uses_confirmed_image_tags(tmp_path):
    project = ProjectManager.create(
        tmp_path / "project",
        "project",
        classes=["cat", "dog"],
        task_type="classify",
    )
    cat_path = project.image_root() / "cat.jpg"
    dog_path = project.image_root() / "dog.jpg"
    pending_path = project.image_root() / "pending.jpg"
    for path in (cat_path, dog_path, pending_path):
        path.write_bytes(b"fake image bytes")

    save_annotation(
        ImageAnnotation(
            image_path=str(cat_path),
            image_size=(100, 100),
            image_tags=["cat"],
            image_tags_confirmed=True,
            tags=["train"],
        ),
        project.label_path_for(cat_path),
    )
    save_annotation(
        ImageAnnotation(
            image_path=str(dog_path),
            image_size=(100, 100),
            image_tags=["dog"],
            image_tags_confirmed=True,
            tags=["holdout"],
        ),
        project.label_path_for(dog_path),
    )
    save_annotation(
        ImageAnnotation(
            image_path=str(pending_path),
            image_size=(100, 100),
            image_tags=["cat"],
            image_tags_confirmed=False,
            tags=["train"],
        ),
        project.label_path_for(pending_path),
    )

    assert count_selected_training_images(
        project,
        task="classify",
        status_filter="confirmed",
        class_filter="cat",
        tag_filter=TagFilter(includes=("train",)),
    ) == 1
