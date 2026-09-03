import os
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox

from src.core.project import ProjectManager
from src.ui.properties import AnnotationPanel
from src.ui.views.detect_pose import DetectPoseView
from src.utils.image import ImageCache


_APP = QApplication.instance() or QApplication([])


def _make_view(project: ProjectManager) -> DetectPoseView:
    view = DetectPoseView(ImageCache(), OrderedDict())
    view.resize(1280, 800)
    view._project = project
    view._refresh_data_folder_tree()
    view._left_pane.setVisible(True)
    view.show()
    _APP.processEvents()
    return view


def test_inspector_sections_stay_top_aligned_when_space_is_available():
    panel = AnnotationPanel()

    assert panel._outer_layout.alignment() & Qt.AlignTop
    panel.close()


def test_all_images_card_is_centered_and_root_menu_has_all_actions(tmp_path):
    project = ProjectManager.create(tmp_path / "project", "demo", task_type="detect")
    view = _make_view(project)

    all_item = view._data_tree.topLevelItem(0)
    card = view._data_tree.itemWidget(all_item, 0)
    viewport_center = view._data_tree.viewport().width() / 2
    card_center = card.geometry().center().x()
    assert abs(card_center - viewport_center) <= 1

    menu, actions = view._build_data_folder_context_menu("")
    assert [action.text() for action in menu.actions() if not action.isSeparator()] == [
        "导入图片...", "导入图片目录...", "删除",
    ]
    assert set(actions) == {"import_files", "import_directory", "delete_images"}
    view.close()


def test_delete_selected_data_folder_removes_only_its_images_and_labels(tmp_path):
    project = ProjectManager.create(tmp_path / "project", "demo", task_type="detect")
    project.create_data_folder("version-a")
    root_image = project.image_root() / "root.jpg"
    version_image = project.image_root() / "version-a" / "version.jpg"
    for image in (root_image, version_image):
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"image")
        label = project.label_path_for(image)
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("{}", encoding="utf-8")

    view = _make_view(project)
    version_item = view._data_tree.topLevelItem(0).child(0)
    view._data_tree.setCurrentItem(version_item)

    with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
        view._delete_selected_data_folder_images()

    assert root_image.exists()
    assert project.label_path_for(root_image).exists()
    assert not version_image.exists()
    assert not project.label_path_for(version_image).exists()
    view.close()


def test_delete_all_images_scope_removes_images_and_labels_from_every_version(tmp_path):
    project = ProjectManager.create(tmp_path / "project", "demo", task_type="detect")
    project.create_data_folder("version-a")
    root_image = project.image_root() / "root.jpg"
    version_image = project.image_root() / "version-a" / "version.jpg"
    for image in (root_image, version_image):
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"image")
        label = project.label_path_for(image)
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("{}", encoding="utf-8")

    view = _make_view(project)
    view._data_tree.setCurrentItem(view._data_tree.topLevelItem(0))

    with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
        view._delete_selected_data_folder_images()

    assert not root_image.exists()
    assert not project.label_path_for(root_image).exists()
    assert not version_image.exists()
    assert not project.label_path_for(version_image).exists()
    view.close()
