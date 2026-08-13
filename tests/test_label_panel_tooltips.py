import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from src.ui.label_panel import LabelPanel


def test_history_and_save_tooltips_are_readable_chinese():
    app = QApplication.instance() or QApplication([])
    panel = LabelPanel()

    assert panel._btn_undo.toolTip() == "回撤 (Ctrl+Z)"
    assert panel._btn_redo.toolTip() == "前进 (Ctrl+Y)"
    assert panel._btn_save.toolTip() == "保存 (Ctrl+S)"

    panel.close()
    app.processEvents()
