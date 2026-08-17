import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QEvent, QPointF, Qt
from PyQt5.QtGui import QColor, QImage, QMouseEvent, QPainter, QPixmap
from PyQt5.QtTest import QSignalSpy
from PyQt5.QtWidgets import QApplication

from src.core.annotation import Annotation
from src.ui.canvas import AnnotationCanvas


def _app():
    return QApplication.instance() or QApplication([])


def test_polygon_hit_test_uses_mask_region_not_bbox_fill():
    _app()
    canvas = AnnotationCanvas()
    canvas.resize(200, 200)
    canvas.set_pixmap(QPixmap(100, 100))
    ann = Annotation(
        class_name="item",
        class_id=0,
        bbox=(0.5, 0.5, 0.8, 0.8),
        polygon=[
            (0.1, 0.1),
            (0.9, 0.1),
            (0.9, 0.9),
            (0.1, 0.9),
        ],
    )
    canvas.set_annotations([ann])

    assert canvas.hit_test(100, 100) == ann.id
    assert canvas.hit_test(100, 20) == ann.id


def test_polygon_hit_test_ignores_blank_bbox_area_outside_polygon():
    _app()
    canvas = AnnotationCanvas()
    canvas.resize(200, 200)
    canvas.set_pixmap(QPixmap(100, 100))
    ann = Annotation(
        class_name="item",
        class_id=0,
        bbox=(0.5, 0.5, 0.8, 0.8),
        polygon=[
            (0.1, 0.1),
            (0.3, 0.1),
            (0.3, 0.3),
            (0.1, 0.3),
        ],
    )
    canvas.set_annotations([ann])

    assert canvas.hit_test(100, 100) is None


def test_bbox_hit_test_still_works_for_detection_annotations():
    _app()
    canvas = AnnotationCanvas()
    canvas.resize(200, 200)
    canvas.set_pixmap(QPixmap(100, 100))
    ann = Annotation(
        class_name="item",
        class_id=0,
        bbox=(0.5, 0.5, 0.8, 0.8),
    )
    canvas.set_annotations([ann])

    assert canvas.hit_test(100, 100) == ann.id


def test_bbox_resize_can_shrink_to_one_image_pixel():
    _app()
    canvas = AnnotationCanvas()
    canvas.resize(200, 200)
    canvas.set_pixmap(QPixmap(1000, 1000))
    ann = Annotation(
        class_name="item",
        class_id=0,
        bbox=(0.5, 0.5, 0.2, 0.2),
    )
    canvas.set_annotations([ann])
    canvas.select_annotation(ann.id)

    canvas._drag_ann_id = ann.id
    canvas._drag_type = "resize_br"
    canvas._drag_start_norm = (0.6, 0.6)
    canvas._drag_ann_snapshot = ann.to_dict()

    canvas._handle_drag(0.0, 0.0)

    assert ann.bbox is not None
    assert abs(ann.bbox[2] - 0.001) < 1e-12
    assert abs(ann.bbox[3] - 0.001) < 1e-12


def test_annotation_canvas_label_background_is_fully_transparent():
    _app()
    canvas = AnnotationCanvas()
    canvas._image_w = 100
    canvas._image_h = 100
    image = QImage(160, 80, QImage.Format_ARGB32)
    background = QColor("#123456")
    image.fill(background)
    painter = QPainter(image)
    canvas._paint_label(
        painter,
        20,
        40,
        Annotation("defect", 0, bbox=(0.5, 0.5, 0.2, 0.2)),
        QColor("#ff0000"),
        False,
    )
    painter.end()

    assert image.pixelColor(20, 20) == background


def test_dragging_pending_box_syncs_panels_only_after_mouse_release():
    app = _app()
    canvas = AnnotationCanvas()
    canvas.resize(240, 160)
    canvas.set_pixmap(QPixmap(200, 100))
    annotation = Annotation(
        class_name="prediction",
        class_id=0,
        bbox=(0.5, 0.5, 0.3, 0.3),
        confirmed=False,
        source="auto",
    )
    canvas.set_annotations([annotation])
    canvas.show()
    app.processEvents()

    changed_spy = QSignalSpy(canvas.annotations_changed)
    modified_spy = QSignalSpy(canvas.annotation_modified)
    canvas._dragging = True
    canvas._drag_type = "move"
    canvas._drag_ann_id = annotation.id
    canvas._drag_start_norm = (0.5, 0.5)
    canvas._drag_ann_snapshot = annotation.to_dict()
    canvas._handle_drag(0.6, 0.55)

    assert annotation.confirmed is True
    assert len(changed_spy) == 0

    release = QMouseEvent(
        QEvent.MouseButtonRelease,
        QPointF(120, 100),
        Qt.LeftButton,
        Qt.NoButton,
        Qt.NoModifier,
    )
    canvas.mouseReleaseEvent(release)
    assert len(modified_spy) == 1
    assert modified_spy[0][0] == annotation.id
    assert len(changed_spy) == 0
    canvas.close()
