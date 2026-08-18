from src.core.project import ProjectManager


def test_existing_label_does_not_block_image_import(tmp_path):
    project = ProjectManager.create(tmp_path / "project", "demo")
    project.create_data_folder("smg")

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_image = source_dir / "sample.jpg"
    source_label = source_dir / "sample.json"
    source_image.write_bytes(b"image")
    source_label.write_text('{"source": true}', encoding="utf-8")

    destination_image = project.image_root() / "smg" / "sample.jpg"
    destination_label = project.label_path_for(destination_image)
    destination_label.parent.mkdir(parents=True, exist_ok=True)
    destination_label.write_text('{"existing": true}', encoding="utf-8")

    imported, skipped = project.import_images_to_folder([source_dir], "smg")

    assert imported == [destination_image]
    assert skipped == []
    assert destination_image.read_bytes() == b"image"
    assert destination_label.read_text(encoding="utf-8") == '{"existing": true}'


def test_data_version_import_has_no_small_dataset_limit(tmp_path):
    project = ProjectManager.create(tmp_path / "project", "demo")
    project.create_data_folder("bulk")

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    expected_count = 360
    for index in range(expected_count):
        stem = f"image_{index:04d}"
        (source_dir / f"{stem}.jpg").write_bytes(b"image")

        destination_image = project.image_root() / "bulk" / f"{stem}.jpg"
        destination_label = project.label_path_for(destination_image)
        destination_label.parent.mkdir(parents=True, exist_ok=True)
        destination_label.write_text('{"existing": true}', encoding="utf-8")

    imported, skipped = project.import_images_to_folder([source_dir], "bulk")

    assert len(imported) == expected_count
    assert skipped == []
    assert len(project.list_images(data_folder="bulk")) == expected_count


def test_delete_data_version_only_removes_persistent_index(tmp_path):
    project = ProjectManager.create(tmp_path / "project", "demo")
    project.create_data_folder("version_a")
    image = project.image_root() / "version_a" / "sample.jpg"
    image.write_bytes(b"image")
    label = project.label_path_for(image)
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text('{"annotations": []}', encoding="utf-8")
    project.config.active_data_folder = "version_a"

    project.delete_data_folder("version_a")
    project.save()

    assert image.read_bytes() == b"image"
    assert label.exists()
    assert project.config.active_data_folder == ""
    assert "version_a" not in project.list_data_folders()
    assert image not in project.list_images()

    reopened = ProjectManager.open(project.project_dir)
    assert "version_a" not in reopened.list_data_folders()
    assert reopened.list_images(data_folder="version_a") == []
    assert reopened.list_images() == []


def test_creating_removed_data_version_restores_its_index(tmp_path):
    project = ProjectManager.create(tmp_path / "project", "demo")
    project.create_data_folder("version_a")
    image = project.image_root() / "version_a" / "sample.jpg"
    image.write_bytes(b"image")
    project.delete_data_folder("version_a")

    project.create_data_folder("version_a")

    assert "version_a" in project.list_data_folders()
    assert "version_a" not in project.config.excluded_data_folders
    assert project.list_images(data_folder="version_a") == [image]
    assert image in project.list_images()
