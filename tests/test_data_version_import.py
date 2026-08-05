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
