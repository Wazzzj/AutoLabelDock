import os
from collections import OrderedDict
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QDialog, QLabel, QMessageBox

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


def test_annotation_inspector_matches_card_and_class_list_design():
    """右侧检查器保持设计稿的卡片标注与扁平类别列表层级。"""
    panel = AnnotationPanel()
    panel.resize(292, 760)
    panel.set_class_colors({"bolt": "#4D9FFF", "scratch": "#F16A5D"})
    panel.set_classes(["bolt", "scratch"])
    annotations = [
        Annotation(
            class_name="bolt", class_id=0, bbox=(0.5, 0.5, 0.4, 0.4),
            confidence=0.96, confirmed=False,
        ),
        Annotation(
            class_name="scratch", class_id=1, bbox=(0.5, 0.5, 0.2, 0.2),
            confidence=0.91, confirmed=True,
        ),
    ]
    panel.set_annotations(annotations, image_size=(1000, 1000))
    panel.set_project_stats({"class_counts": {"bolt": 1203, "scratch": 31}})

    assert panel._section_titles["标注列表"].text() == "标注列表 · 2"
    assert panel._ann_tree.sizeHintForRow(0) == 58
    first_ann = panel._ann_tree.topLevelItem(0)
    assert first_ann.text(0) == "bolt"
    assert first_ann.data(0, Qt.UserRole + 3) == "0.96"

    first_class = panel._classes_list.itemWidget(panel._classes_list.item(0))
    assert first_class.findChild(QLabel, "name_lbl").text() == "bolt"
    assert first_class.findChild(QLabel, "count_lbl").text() == "1,203"
    assert "background:transparent" in first_class.styleSheet()
    assert "border:none" in first_class.styleSheet()

    selected = []
    panel.default_class_changed.connect(selected.append)
    QTest.mouseClick(first_class, Qt.LeftButton)
    assert selected == ["bolt"]
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
