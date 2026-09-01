import os
from collections import OrderedDict
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox

from src.core.annotation import Annotation
from src.ui.class_picker import ClassPickerPopup, KeypointLabelPicker
from src.ui.properties import AnnotationPanel
from src.ui.views.detect_pose import DetectPoseView
from src.utils.image import ImageCache


_QT_APP = QApplication.instance() or QApplication([])


def test_escape_closes_class_picker_while_search_has_focus():
    picker = ClassPickerPopup(["person"], {"person": "#ff00ff"})
    QTimer.singleShot(0, lambda: QTest.keyClick(picker._input, Qt.Key_Escape))

    assert picker.exec_() == QDialog.Rejected
    picker.close()


def test_escape_closes_keypoint_picker_while_input_has_focus():
    picker = KeypointLabelPicker(["nose"], default_label="nose")
    QTimer.singleShot(0, lambda: QTest.keyClick(picker._input, Qt.Key_Escape))

    assert picker.exec_() == QDialog.Rejected
    picker.close()


def test_annotation_panel_clear_all_signal_emits():
    """清空入口已移至“更多”菜单；面板信号仍可触发（含确认流程上游）。"""
    panel = AnnotationPanel()
    emitted = []
    panel.clear_all_annotations_requested.connect(lambda: emitted.append(True))

    panel.set_annotations([
        Annotation(class_name="person", class_id=0, bbox=(0.5, 0.5, 0.4, 0.4))
    ])
    panel.clear_all_annotations_requested.emit()
    assert emitted == [True]
    panel.close()


def test_clear_all_annotations_requires_confirmation_and_records_undo():
    view = DetectPoseView(ImageCache(), OrderedDict())
    view._canvas.set_annotations([
        Annotation(class_name="person", class_id=0, bbox=(0.5, 0.5, 0.4, 0.4)),
        Annotation(class_name="person", class_id=0, bbox=(0.2, 0.2, 0.1, 0.1)),
    ])
    view._push_undo = Mock()
    view._sync_annotations_to_panel = Mock()

    with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
        view._on_clear_all_annotations()

    assert view._canvas.annotations == []
    view._push_undo.assert_called_once_with()
    view._sync_annotations_to_panel.assert_called_once_with()
    view.close()


def test_cancelled_picker_draw_state_returns_to_select_mode():
    view = DetectPoseView(ImageCache(), OrderedDict())
    view._set_tool("draw_keypoint")
    view._canvas._draw_start = (0.3, 0.4)

    view._clear_draw_state()

    assert view._canvas.tool_mode == "select"
    assert view._canvas._draw_start is None
    view.close()
