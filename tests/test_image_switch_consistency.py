import os
from collections import OrderedDict

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication

from src.core.annotation import Annotation, ImageAnnotation
from src.core.label_io import save_annotation
from src.core.project import ProjectManager
from src.ui.canvas import AnnotationCanvas
from src.ui.views.detect_pose import DetectPoseView


def _app():
    return QApplication.instance() or QApplication([])


_QT_APP = _app()


def test_replacing_annotations_cancels_drag_from_previous_image():
    app = _QT_APP
    canvas = AnnotationCanvas()
    canvas._dragging = True
    canvas._drag_type = "move"
    canvas._drag_ann_id = "old-annotation"
    canvas._drag_start_norm = (0.2, 0.2)
    canvas._drag_ann_snapshot = {"bbox": (0.2, 0.2, 0.1, 0.1)}

    canvas.set_annotations([
        Annotation("new", 0, bbox=(0.7, 0.7, 0.2, 0.2)),
    ])

    assert not canvas._dragging
    assert canvas._drag_ann_id is None
    assert canvas._drag_start_norm is None
    assert canvas._drag_ann_snapshot is None
    canvas.close()
    assert app is not None


class _ReentrantImageCache:
    def __init__(self):
        self.view = None
        self.first_path = None
        self.second_path = None
        self.triggered = False
        self.pixmaps = {}

    def get(self, path):
        path = type(self.first_path)(path)
        if path == self.first_path and not self.triggered:
            self.triggered = True
            self.view._on_image_selected(self.second_path)
        return self.pixmaps[path]


def test_reentrant_image_switch_keeps_latest_image_and_annotations_together(tmp_path):
    app = _QT_APP
    project = ProjectManager.create(
        tmp_path / "project", "project", classes=["first", "second"],
        task_type="detect",
    )
    first_path = project.image_root() / "first.jpg"
    second_path = project.image_root() / "second.jpg"
    first_path.write_bytes(b"placeholder")
    second_path.write_bytes(b"placeholder")

    first_annotation = Annotation(
        "first", 0, bbox=(0.2, 0.2, 0.1, 0.1), confirmed=False,
    )
    second_annotation = Annotation(
        "second", 1, bbox=(0.8, 0.8, 0.1, 0.1), confirmed=False,
    )
    save_annotation(
        ImageAnnotation(first_path.name, (100, 100), [first_annotation]),
        project.label_path_for(first_path),
    )
    save_annotation(
        ImageAnnotation(second_path.name, (100, 100), [second_annotation]),
        project.label_path_for(second_path),
    )

    cache = _ReentrantImageCache()
    cache.first_path = first_path
    cache.second_path = second_path
    cache.pixmaps = {
        first_path: QPixmap(100, 100),
        second_path: QPixmap(100, 100),
    }
    view = DetectPoseView(cache, OrderedDict())
    cache.view = view
    view._project = project

    view._on_image_selected(first_path)

    assert view._current_image_path == second_path
    assert [annotation.class_name for annotation in view._canvas.annotations] == ["second"]
    assert view._current_annotation.image_path == second_path.name
    view.close()
    assert app is not None
