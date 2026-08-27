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


def test_importing_image_root_preserves_children_as_data_versions(tmp_path):
    project = ProjectManager.create(
        tmp_path / "project",
        "demo",
        image_dir=".",
    )
    source_root = tmp_path / "source"
    source_image = source_root / "version-a" / "nested" / "sample.jpg"
    source_image.parent.mkdir(parents=True)
    source_image.write_bytes(b"image")

    imported, skipped = project.import_images_to_folder([source_root], "")

    expected = project.project_dir / "version-a" / "nested" / "sample.jpg"
    assert imported == [expected]
    assert skipped == []
    assert expected.read_bytes() == b"image"
    assert project.list_data_folders() == ["version-a"]
    assert project.list_images(data_folder="version-a") == [expected]


def test_existing_root_sidecars_are_mirrored_without_removing_sources(tmp_path):
    project_root = tmp_path / "project"
    version = project_root / "version-a"
    version.mkdir(parents=True)
    image = version / "sample.jpg"
    sidecar = version / "sample.json"
    image.write_bytes(b"image")
    sidecar.write_text(
        '{"image_path":"sample.jpg","image_size":[10,10],'
        '"annotations":[],"image_tags":["ok"]}',
        encoding="utf-8",
    )
    project = ProjectManager.create(project_root, "demo", image_dir=".")
    project.config.label_dir = "labels"
    (project_root / "labels").mkdir()

    imported = project.import_image_sidecar_annotations()

    mirrored = project_root / "labels" / "version-a" / "sample.json"
    assert imported == 1
    assert sidecar.exists()
    assert mirrored.exists()
    assert '"image_path": "version-a/sample.jpg"' in mirrored.read_text(encoding="utf-8")


def test_new_projects_store_annotations_beside_images(tmp_path):
    project = ProjectManager.create(tmp_path / "project", "demo", image_dir=".")
    image = project.project_dir / "version-a" / "sample.jpg"
    image.parent.mkdir()
    image.write_bytes(b"image")

    assert project.config.label_dir == "."
    assert project.label_path_for(image) == image.with_suffix(".json")
    assert not (project.project_dir / "labels").exists()


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
