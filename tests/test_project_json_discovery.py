from pathlib import Path

from src.controllers.project import ProjectController
from src.core.config import AppConfig
from src.core.project import ProjectManager
from src.core.project_scan import best_project_data_candidate


def test_root_json_moves_to_mirrored_labels_when_root_image_is_moved(tmp_path):
    project = ProjectManager.create(tmp_path, "demo")
    image_path = tmp_path / "sample.jpg"
    json_path = tmp_path / "sample.json"
    image_path.write_bytes(b"fake image")
    json_path.write_text('{"shapes": []}', encoding="utf-8")

    controller = ProjectController(AppConfig(), tmp_path / "config.json", None)

    moved_images, moved_labels = controller._move_root_images_to_image_dir(project)

    assert moved_images == 1
    assert moved_labels == 1
    assert (tmp_path / "images" / "sample.jpg").exists()
    assert (tmp_path / "images" / "sample.json").exists()
    assert not image_path.exists()
    assert not json_path.exists()


def test_project_scan_detects_root_json_for_images_dir(tmp_path):
    project = ProjectManager.create(tmp_path, "demo")
    (tmp_path / "images" / "sample.jpg").write_bytes(b"fake image")
    (tmp_path / "sample.json").write_text('{"shapes": []}', encoding="utf-8")

    detected = best_project_data_candidate(project)

    assert detected is not None
    assert detected.image_dir == "images"
    assert detected.label_dir == "."
    assert detected.image_count == 1
    assert detected.matched_json_count == 1
