import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtGui import QPixmap
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
