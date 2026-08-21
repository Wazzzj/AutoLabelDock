import os
from collections import OrderedDict

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication, QMessageBox

from src.core.annotation import ImageAnnotation
from src.core.label_io import load_annotation, save_annotation
from src.core.project import ProjectManager
from src.ui.views.classify import ClassifyView
from src.utils.image import ImageCache


_APP = QApplication.instance() or QApplication([])


def _project_with_classified_images(tmp_path):
    project = ProjectManager.create(
        tmp_path / "classify-project",
        "classify-project",
        classes=["NG", "OK"],
        task_type="classify",
    )
    paths = []
    for index, class_name in enumerate(("NG", "OK", "OK")):
        path = project.image_root() / f"sample-{index}.png"
        image = QImage(24, 24, QImage.Format_RGB32)
        image.fill(Qt.white)
        assert image.save(str(path))
        save_annotation(
            ImageAnnotation(
                image_path=path.name,
                image_size=(24, 24),
                image_tags=[class_name],
                image_tags_confirmed=True,
            ),
            project.label_path_for(path),
        )
        paths.append(path)
    return project, paths


def _view_for(project, tmp_path):
    view = ClassifyView(
        ImageCache(),
        OrderedDict(),
        config_path=tmp_path / "app-config.json",
    )
    view.set_classes(project.config.classes)
    view.set_class_colors(
        {name: project.config.get_class_color(name) for name in project.config.classes}
    )
    view.set_project(project)
    _APP.processEvents()
    return view


def test_classify_workspace_shows_counts_and_batch_class_assignment(tmp_path):
    project, paths = _project_with_classified_images(tmp_path)
    view = _view_for(project, tmp_path)
    try:
        assert "共 3 张" in view._selection_summary.text()
        assert "已标注 3 张" in view._selection_summary.text()
        assert view._class_bar._buttons["NG"].text() == "1  NG  (1)"
        assert view._class_bar._buttons["OK"].text() == "2  OK  (2)"

        view._select_all_visible()
        assert len(view._grid.selectedItems()) == 3
        assert "已选 3 张" in view._selection_summary.text()
        assert view._delete_selected_btn.isEnabled()

        view._apply_class("NG")
        assert all(
            load_annotation(project.label_path_for(path)).image_tags == ["NG"]
            for path in paths
        )
        assert view._class_bar._buttons["NG"].text() == "1  NG  (3)"
        assert view._class_bar._buttons["OK"].text() == "2  OK  (0)"
    finally:
        view.cleanup()
        view.close()


def test_classify_workspace_deletes_all_selected_images(tmp_path, monkeypatch):
    project, paths = _project_with_classified_images(tmp_path)
    view = _view_for(project, tmp_path)
    try:
        view._grid.item(0).setSelected(True)
        view._grid.item(2).setSelected(True)
        monkeypatch.setattr(
            QMessageBox,
            "question",
            lambda *_args, **_kwargs: QMessageBox.Yes,
        )

        view._delete_selected_images()

        assert not paths[0].exists()
        assert paths[1].exists()
        assert not paths[2].exists()
        assert not project.label_path_for(paths[0]).exists()
        assert not project.label_path_for(paths[2]).exists()
        assert view._grid.count() == 1
        assert "共 1 张" in view._selection_summary.text()
    finally:
        view.cleanup()
        view.close()
