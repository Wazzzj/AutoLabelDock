from src.core.backup import BackupManager
from src.core.project import ProjectManager


def test_backup_and_restore_sidecar_annotations(tmp_path):
    project = ProjectManager.create(tmp_path / "project", "demo", image_dir=".")
    version = project.project_dir / "version-a"
    version.mkdir()
    image = version / "sample.jpg"
    sidecar = version / "sample.json"
    image.write_bytes(b"image")
    sidecar.write_text('{"state":"original"}', encoding="utf-8")

    manager = BackupManager(project.project_dir)
    backup = manager.create_backup(project.config.label_dir)

    assert backup is not None
    assert (backup / "labels" / "version-a" / "sample.json").exists()
    assert manager.list_backups()[0]["label_count"] == 1

    sidecar.write_text('{"state":"changed"}', encoding="utf-8")
    assert manager.restore_backup(backup.name, project.config.label_dir)
    assert sidecar.read_text(encoding="utf-8") == '{"state":"original"}'
