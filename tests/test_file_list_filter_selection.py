import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from src.ui.file_list import FileListWidget


def _app():
    return QApplication.instance() or QApplication([])


def test_filtering_unchecks_hidden_file_list_items(tmp_path):
    _app()
    keep = tmp_path / "keep.jpg"
    drop = tmp_path / "drop.jpg"
    widget = FileListWidget()
    widget.set_image_paths([keep, drop])
    widget.set_status(keep, "confirmed")
    widget.set_status(drop, "pending")

    widget.check_visible_paths(True)
    widget.item(1).setSelected(True)

    widget.set_filter("confirmed")

    assert not widget.item(0).isHidden()
    assert widget.item(1).isHidden()
    assert widget.item(0).checkState() == Qt.Checked
    assert widget.item(1).checkState() == Qt.Unchecked
    assert not widget.item(1).isSelected()
    assert widget.get_selected_paths() == [keep]

