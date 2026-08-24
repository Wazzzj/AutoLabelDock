from pathlib import Path

import pytest

from src.core.project import ProjectManager


def test_set_image_directory_stores_project_subdirectory_as_relative(tmp_path):
    project = ProjectManager.create(tmp_path / "project", "project")
    old_image = project.image_root() / "old.jpg"
    old_image.write_bytes(b"old")
    selected = project.project_dir / "datasets" / "camera_a"
    selected.mkdir(parents=True)
    new_image = selected / "new.png"
    new_image.write_bytes(b"new")
    project.config.active_data_folder = "old-version"
    project.config.data_folders = ["old-version"]
    project.config.excluded_data_folders = ["removed-version"]

    resolved = project.set_image_directory(selected)

    assert resolved == selected.resolve()
    assert Path(project.config.image_dir) == Path("datasets") / "camera_a"
    assert project.image_root().resolve() == selected.resolve()
    assert project.config.active_data_folder == ""
    assert project.config.data_folders == []
    assert project.config.excluded_data_folders == []
    assert project.list_images() == [new_image]
    assert old_image.exists()

    reopened = ProjectManager.open(project.project_dir)
    assert reopened.image_root().resolve() == selected.resolve()


def test_set_image_directory_keeps_external_directory_absolute(tmp_path):
    project = ProjectManager.create(tmp_path / "project", "project")
    external = tmp_path / "external-images"
    external.mkdir()

    project.set_image_directory(external)

    assert Path(project.config.image_dir).is_absolute()
    assert project.image_root() == external.resolve()


def test_set_image_directory_rejects_missing_path(tmp_path):
    project = ProjectManager.create(tmp_path / "project", "project")

    with pytest.raises(FileNotFoundError):
        project.set_image_directory(tmp_path / "missing")
