from src.controllers.project import ProjectController
from src.core.label_io import load_annotation
from src.core.project import ProjectManager


def _write_voc_obb(path, filename: str, class_name: str) -> None:
    path.write_text(
        f"""<annotation>
  <filename>{filename}</filename>
  <size><width>200</width><height>100</height><depth>3</depth></size>
  <object><name>{class_name}</name><type>robndbox</type><robndbox>
    <cx>100</cx><cy>50</cy><w>80</w><h>20</h><angle>0.5</angle>
  </robndbox></object>
</annotation>""",
        encoding="utf-8",
    )


def test_obb_project_auto_imports_best_matching_root_sidecars(tmp_path):
    project = ProjectManager.create(
        tmp_path,
        "obb-project",
        task_type="obb",
        classes=[],
    )
    first_image = project.image_root() / "first.jpeg"
    second_image = project.image_root() / "second.jpeg"
    first_image.write_bytes(b"image")
    second_image.write_bytes(b"image")

    # Only one YOLO sidecar matches, while both XML sidecars match.
    (tmp_path / "classes.txt").write_text("0:txt_only\n", encoding="utf-8")
    (tmp_path / "first.txt").write_text(
        "0 0.1 0.2 0.7 0.1 0.8 0.6 0.2 0.7\n",
        encoding="utf-8",
    )
    _write_voc_obb(tmp_path / "first.xml", "first.jpeg", "xml_class")
    _write_voc_obb(tmp_path / "second.xml", "second.jpeg", "xml_class")

    imported_count, imported_format = ProjectController._import_discovered_obb_sidecars(project)

    assert imported_count == 2
    assert imported_format == "VOC-OBB"
    assert project.config.classes == ["xml_class"]
    assert load_annotation(project.label_path_for(first_image)).annotations[0].class_name == "xml_class"
    assert load_annotation(project.label_path_for(second_image)).annotations[0].class_name == "xml_class"

    # Reopening is idempotent and never overwrites existing internal labels.
    assert ProjectController._import_discovered_obb_sidecars(project) == (0, "")


def test_sidecar_auto_import_is_disabled_for_non_obb_projects(tmp_path):
    project = ProjectManager.create(tmp_path, "detect-project", task_type="detect")
    image_path = project.image_root() / "sample.jpeg"
    image_path.write_bytes(b"image")
    _write_voc_obb(tmp_path / "sample.xml", "sample.jpeg", "object")

    assert ProjectController._import_discovered_obb_sidecars(project) == (0, "")
    assert load_annotation(project.label_path_for(image_path)) is None
