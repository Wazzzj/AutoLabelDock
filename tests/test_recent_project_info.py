from src.app import WelcomePage
from src.core.project import ProjectManager


def test_recent_project_info_accepts_project_json_file_path(tmp_path):
    project = ProjectManager.create(tmp_path / "obb-project", "OBB", task_type="obb")

    info = WelcomePage._read_recent_project_info(None, str(project.project_dir / "project.json"))

    assert info["name"] == "OBB"
    assert info["task_type"] == "obb"
