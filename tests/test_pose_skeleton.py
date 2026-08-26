import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QRect
from PyQt5.QtGui import QColor, QImage, QPainter, QPixmap
from PyQt5.QtWidgets import QApplication

from src.core.annotation import (
    Annotation,
    COCO17_SKELETON_EDGES,
    ImageAnnotation,
    Keypoint,
    POSE_BOUNDING_BOX_RGB,
    keypoint_display_rgb,
    keypoint_skeleton_colored_segments,
    keypoint_skeleton_segments,
)
from src.ui.canvas import AnnotationCanvas
from src.ui.preview_panel import _draw_annotation_overlays


_QT_APP = QApplication.instance() or QApplication([])


def _keypoints(count: int) -> list[Keypoint]:
    return [
        Keypoint(x=index / max(1, count), y=0.5, visible=2, label=f"kp_{index}")
        for index in range(count)
    ]


def test_coco_17_keypoints_use_official_human_pose_skeleton():
    keypoints = _keypoints(17)

    segments = keypoint_skeleton_segments(keypoints)

    index_by_id = {id(keypoint): index for index, keypoint in enumerate(keypoints)}
    actual_edges = [
        (index_by_id[id(start)], index_by_id[id(end)])
        for start, end in segments
    ]
    assert actual_edges == list(COCO17_SKELETON_EDGES)


def test_custom_pose_connects_adjacent_keypoints_in_stored_order():
    keypoints = _keypoints(4)

    segments = keypoint_skeleton_segments(keypoints)

    assert segments == [
        (keypoints[0], keypoints[1]),
        (keypoints[1], keypoints[2]),
        (keypoints[2], keypoints[3]),
    ]


def test_skeleton_skips_edges_touching_invisible_keypoints():
    keypoints = _keypoints(4)
    keypoints[1].visible = 0

    assert keypoint_skeleton_segments(keypoints) == [
        (keypoints[2], keypoints[3]),
    ]


def test_coco_17_pose_uses_official_reference_colors():
    keypoints = _keypoints(17)

    segments = keypoint_skeleton_colored_segments(keypoints, (1, 2, 3))

    assert all(rgb == (224, 255, 41) for _, _, rgb in segments)
    assert keypoint_display_rgb(keypoints, 0, (1, 2, 3)) == (254, 100, 218)
    assert keypoint_display_rgb(keypoints, 5, (1, 2, 3)) == (254, 100, 218)
    assert keypoint_display_rgb(keypoints, 11, (1, 2, 3)) == (254, 100, 218)


def test_custom_pose_keeps_class_color():
    keypoints = _keypoints(4)
    fallback = (12, 34, 56)

    assert all(
        rgb == fallback
        for _, _, rgb in keypoint_skeleton_colored_segments(keypoints, fallback)
    )
    assert keypoint_display_rgb(keypoints, 0, fallback) == fallback


def _two_point_annotation() -> Annotation:
    return Annotation(
        class_name="person",
        class_id=0,
        keypoints=[
            Keypoint(0.2, 0.5, 2, "start"),
            Keypoint(0.8, 0.5, 2, "end"),
        ],
    )


def test_annotation_canvas_renders_line_between_pose_keypoints():
    canvas = AnnotationCanvas()
    canvas.resize(200, 200)
    pixmap = QPixmap(100, 100)
    pixmap.fill(QColor("white"))
    canvas.set_pixmap(pixmap)
    canvas.set_class_colors({"person": "#ff00ff"})
    canvas.set_annotations([_two_point_annotation()])

    rendered = QImage(200, 200, QImage.Format_ARGB32)
    rendered.fill(QColor("white"))
    painter = QPainter(rendered)
    canvas.render(painter)
    painter.end()

    assert rendered.pixelColor(100, 100) != QColor("white")
    canvas.close()


def test_preview_overlay_renders_line_between_pose_keypoints():
    rendered = QImage(100, 100, QImage.Format_ARGB32)
    rendered.fill(QColor("white"))
    painter = QPainter(rendered)
    _draw_annotation_overlays(
        painter,
        QRect(0, 0, 100, 100),
        ImageAnnotation("pose.jpg", (100, 100), [_two_point_annotation()]),
        {"person": "#ff00ff"},
    )
    painter.end()

    assert rendered.pixelColor(50, 50) != QColor("white")
    assert _QT_APP is not None


def test_occluded_keypoint_is_hollow_in_annotation_canvas():
    annotation = _two_point_annotation()
    annotation.keypoints = annotation.keypoints[:1]
    annotation.keypoints[0].visible = 1
    canvas = AnnotationCanvas()
    canvas.resize(200, 200)
    pixmap = QPixmap(100, 100)
    pixmap.fill(QColor("white"))
    canvas.set_pixmap(pixmap)
    canvas.set_class_colors({"person": "#ff00ff"})
    canvas.set_annotations([annotation])

    rendered = QImage(200, 200, QImage.Format_ARGB32)
    rendered.fill(QColor("white"))
    painter = QPainter(rendered)
    canvas.render(painter)
    painter.end()

    assert rendered.pixelColor(40, 100) == QColor("white")
    assert any(
        rendered.pixelColor(x, 100) != QColor("white")
        for x in range(43, 47)
    )
    canvas.close()


def test_occluded_keypoint_is_hollow_in_preview():
    annotation = _two_point_annotation()
    annotation.keypoints = annotation.keypoints[:1]
    annotation.keypoints[0].visible = 1
    rendered = QImage(100, 100, QImage.Format_ARGB32)
    rendered.fill(QColor("white"))
    painter = QPainter(rendered)
    _draw_annotation_overlays(
        painter,
        QRect(0, 0, 100, 100),
        ImageAnnotation("pose.jpg", (100, 100), [annotation]),
        {"person": "#ff00ff"},
    )
    painter.end()

    assert rendered.pixelColor(20, 50) == QColor("white")
    assert any(
        rendered.pixelColor(x, 50) == QColor("#ff00ff")
        for x in range(23, 26)
    )


def test_pose_bbox_uses_high_contrast_color_in_annotation_canvas():
    canvas = AnnotationCanvas()
    canvas.resize(200, 200)
    pixmap = QPixmap(100, 100)
    pixmap.fill(QColor("white"))
    canvas.set_pixmap(pixmap)
    canvas.set_class_colors({"person": "#f9e2af"})
    annotation = _two_point_annotation()
    annotation.bbox = (0.5, 0.5, 0.8, 0.8)
    canvas.set_annotations([annotation])

    rendered = QImage(200, 200, QImage.Format_ARGB32)
    rendered.fill(QColor("white"))
    painter = QPainter(rendered)
    canvas.render(painter)
    painter.end()

    assert rendered.pixelColor(20, 100) == QColor(*POSE_BOUNDING_BOX_RGB)
    canvas.close()


def test_pose_bbox_uses_high_contrast_color_in_preview():
    annotation = _two_point_annotation()
    annotation.bbox = (0.5, 0.5, 0.8, 0.8)
    rendered = QImage(100, 100, QImage.Format_ARGB32)
    rendered.fill(QColor("white"))
    painter = QPainter(rendered)
    _draw_annotation_overlays(
        painter,
        QRect(0, 0, 100, 100),
        ImageAnnotation("pose.jpg", (100, 100), [annotation]),
        {"person": "#f9e2af"},
    )
    painter.end()

    expected = QColor(*POSE_BOUNDING_BOX_RGB)
    assert any(rendered.pixelColor(x, 50) == expected for x in range(7, 13))
