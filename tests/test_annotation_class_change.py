import os
import unittest
from collections import OrderedDict
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from src.core.annotation import Annotation, ImageAnnotation
from src.core.label_io import load_annotation, save_annotation
from src.core.project import ProjectManager
from src.ui.views.detect_pose import DetectPoseView
from src.utils.image import ImageCache


_APP = None


def _app():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


class _Config:
    classes = ["old", "new"]

    def get_class_id(self, class_name):
        return self.classes.index(class_name)

    def get_class_color(self, _class_name):
        return "#00ffff"


class _Project:
    config = _Config()

    @staticmethod
    def list_images():
        return []


class TestAnnotationClassChange(unittest.TestCase):
    def test_panel_change_updates_only_annotation_with_requested_id(self):
        self.app = _app()
        view = DetectPoseView(ImageCache(), OrderedDict())
        first = Annotation(class_name="old", class_id=0)
        same_name = Annotation(class_name="old", class_id=0)
        view._project = _Project()
        view._canvas.set_annotations([first, same_name])
        view._show_class_picker_at = Mock(return_value="new")
        view._push_undo = Mock()
        view._sync_annotations_to_panel = Mock()

        view._on_panel_annotation_class_change(first.id)

        self.assertEqual(first.class_name, "new")
        self.assertEqual(first.class_id, 1)
        self.assertEqual(same_name.class_name, "old")
        self.assertEqual(same_name.class_id, 0)
        view._push_undo.assert_called_once_with()
        view._sync_annotations_to_panel.assert_called_once_with()
        view.close()


def test_multi_image_selection_relabels_matching_annotations(tmp_path):
    _app()
    project = ProjectManager.create(
        tmp_path / "project",
        "project",
        classes=["old", "new", "keep"],
        task_type="detect",
    )
    first_path = project.image_root() / "first.jpg"
    second_path = project.image_root() / "second.jpg"
    first_path.write_bytes(b"image")
    second_path.write_bytes(b"image")

    first = ImageAnnotation(
        image_path=first_path.name,
        image_size=(100, 100),
        annotations=[
            Annotation(class_name="old", class_id=0, bbox=(0.3, 0.3, 0.2, 0.2)),
            Annotation(class_name="keep", class_id=2, bbox=(0.7, 0.7, 0.2, 0.2)),
        ],
    )
    second = ImageAnnotation(
        image_path=second_path.name,
        image_size=(100, 100),
        annotations=[
            Annotation(class_name="old", class_id=0, bbox=(0.5, 0.5, 0.2, 0.2)),
        ],
    )
    save_annotation(first, project.label_path_for(first_path))
    save_annotation(second, project.label_path_for(second_path))

    view = DetectPoseView(ImageCache(), OrderedDict())
    view._project = project
    view._current_image_path = first_path
    view._current_annotation = first
    view._prev_annotations_snapshot = view._stats_snapshot(first.annotations)
    view._canvas.set_annotations(list(first.annotations))
    view._file_list.set_image_paths([first_path, second_path])
    view._file_list.item(0).setCheckState(Qt.Checked)
    view._file_list.item(1).setCheckState(Qt.Checked)
    view._show_class_picker_at = Mock(return_value="new")

    view._on_panel_annotation_class_change(first.annotations[0].id)

    first_saved = load_annotation(project.label_path_for(first_path))
    second_saved = load_annotation(project.label_path_for(second_path))
    assert [ann.class_name for ann in first_saved.annotations] == ["new", "keep"]
    assert [ann.class_name for ann in second_saved.annotations] == ["new"]
    assert all(
        ann.class_id == 1
        for ia in (first_saved, second_saved)
        for ann in ia.annotations
        if ann.class_name == "new"
    )
    view.close()


if __name__ == "__main__":
    unittest.main()
