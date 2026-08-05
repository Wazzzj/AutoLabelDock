import os
import unittest
from collections import OrderedDict
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from src.core.annotation import Annotation
from src.ui.views.detect_pose import DetectPoseView
from src.utils.image import ImageCache


def _app():
    return QApplication.instance() or QApplication([])


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


if __name__ == "__main__":
    unittest.main()
