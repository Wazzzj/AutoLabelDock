from collections import OrderedDict
import math

from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QMenu

from src.core.annotation import Annotation
from src.core.project import ProjectManager
from src.ui.canvas import AnnotationCanvas
from src.ui.views.detect_pose import DetectPoseView
from src.utils.image import ImageCache


_APP = QApplication.instance() or QApplication([])


def test_obb_project_keeps_polygon_and_oriented_box_as_separate_tools(tmp_path):
    project = ProjectManager.create(
        tmp_path / "obb-project",
        "obb-project",
        task_type="obb",
    )
    view = DetectPoseView(ImageCache(), OrderedDict())

    view.set_project(project)

    assert view._btn_polygon.text() == "多边形"
    assert view._btn_obb.text() == "旋转框"
    assert not view._btn_polygon.isHidden()
    assert not view._btn_obb.isHidden()

    view._activate_polygon_tool(False)
    assert view._canvas._polygon_point_limit is None
    assert view._canvas.tool_mode == "draw_polygon"
    assert view._btn_polygon.isChecked()
    assert not view._btn_obb.isChecked()

    view._activate_polygon_tool(True)
    assert view._canvas._polygon_point_limit is None
    assert view._canvas.tool_mode == "draw_obb"
    assert not view._btn_polygon.isChecked()
    assert view._btn_obb.isChecked()


def test_obb_context_menu_survives_case_variant_task_and_canvas_clear(tmp_path):
    project = ProjectManager.create(
        tmp_path / "obb-project-case",
        "obb-project-case",
        task_type="obb",
    )
    project.config.task_type = " OBB "
    view = DetectPoseView(ImageCache(), OrderedDict())

    view.set_project(project)
    view._canvas.clear()
    menu = QMenu()
    view._canvas._add_tool_mode_actions(menu)

    assert "旋转框" in [action.text() for action in menu.actions()]
    view.close()


def test_segment_project_only_shows_polygon_tool(tmp_path):
    project = ProjectManager.create(
        tmp_path / "segment-project",
        "segment-project",
        task_type="segment",
    )
    view = DetectPoseView(ImageCache(), OrderedDict())

    view.set_project(project)

    assert not view._btn_polygon.isHidden()
    assert view._btn_obb.isHidden()


def test_obb_context_menu_switches_to_drag_tool(tmp_path):
    project = ProjectManager.create(
        tmp_path / "obb-project",
        "obb-project",
        task_type="obb",
    )
    view = DetectPoseView(ImageCache(), OrderedDict())
    view.set_project(project)
    menu = QMenu()

    view._canvas._add_tool_mode_actions(menu)
    actions = {action.text(): action for action in menu.actions()}

    assert "旋转框" in actions
    actions["旋转框"].trigger()
    assert view._canvas.tool_mode == "draw_obb"
    assert view._canvas._polygon_point_limit is None
    assert view._btn_obb.isChecked()


def test_dragged_obb_is_created_as_four_point_polygon():
    canvas = AnnotationCanvas()
    canvas.set_pixmap(QPixmap(200, 100))
    canvas.set_tool_mode("draw_obb")
    canvas._draw_start = (0.7, 0.8)
    canvas._draw_current = (0.2, 0.3)

    annotation = canvas.create_obb_from_draw("object", 0)

    assert annotation is not None
    expected_bbox = (0.45, 0.55, 0.5, 0.5)
    for actual, expected in zip(annotation.bbox, expected_bbox):
        assert math.isclose(actual, expected, abs_tol=1e-9)
    assert annotation.polygon == [
        (0.2, 0.3),
        (0.7, 0.3),
        (0.7, 0.8),
        (0.2, 0.8),
    ]
    assert canvas.annotations == [annotation]


def test_obb_rotation_handle_rotates_all_four_points_together():
    canvas = AnnotationCanvas()
    canvas.resize(400, 240)
    canvas.set_pixmap(QPixmap(200, 100))
    canvas.set_obb_editing_enabled(True)
    annotation = Annotation(
        class_name="object",
        class_id=0,
        bbox=(0.5, 0.5, 0.4, 0.4),
        polygon=[(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)],
    )
    canvas.set_annotations([annotation])
    canvas.select_annotation(annotation.id)
    canvas.show()
    _APP.processEvents()
    assert not canvas.grab().isNull()
    _edge_point, handle = canvas._obb_rotation_handle_positions(annotation)
    assert canvas._hit_test_handle(*handle) == "rotate_obb"

    canvas.set_tool_mode("draw_obb")
    QTest.mousePress(
        canvas,
        Qt.LeftButton,
        pos=QPoint(round(handle[0]), round(handle[1])),
    )
    assert canvas._dragging
    assert canvas._drag_type == "rotate_obb"
    QTest.mouseRelease(
        canvas,
        Qt.LeftButton,
        pos=QPoint(round(handle[0]), round(handle[1])),
    )

    start = canvas.pixel_to_norm(*handle)
    center = (0.5, 0.5)
    start_vector = (
        (start[0] - center[0]) * canvas._image_w,
        (start[1] - center[1]) * canvas._image_h,
    )
    current = (
        center[0] - start_vector[1] / canvas._image_w,
        center[1] + start_vector[0] / canvas._image_h,
    )
    canvas._drag_ann_id = annotation.id
    canvas._drag_type = "rotate_obb"
    canvas._drag_start_norm = start
    canvas._drag_ann_snapshot = annotation.to_dict()

    canvas._handle_drag(*current)

    expected = [(0.6, 0.1), (0.6, 0.9), (0.4, 0.9), (0.4, 0.1)]
    for actual_point, expected_point in zip(annotation.polygon, expected):
        assert math.isclose(actual_point[0], expected_point[0], abs_tol=1e-9)
        assert math.isclose(actual_point[1], expected_point[1], abs_tol=1e-9)


def test_obb_corner_drag_resizes_rectangle_instead_of_moving_one_vertex():
    canvas = AnnotationCanvas()
    canvas.set_pixmap(QPixmap(200, 100))
    canvas.set_obb_editing_enabled(True)

    angle = math.radians(30)
    width_axis = (math.cos(angle), math.sin(angle))
    height_axis = (-math.sin(angle), math.cos(angle))
    center = (100.0, 50.0)

    def point(width_offset, height_offset):
        return (
            center[0] + width_axis[0] * width_offset + height_axis[0] * height_offset,
            center[1] + width_axis[1] * width_offset + height_axis[1] * height_offset,
        )

    pixel_polygon = [
        point(-30, -15),
        point(30, -15),
        point(30, 15),
        point(-30, 15),
    ]
    polygon = [(x / 200, y / 100) for x, y in pixel_polygon]
    xs = [item[0] for item in polygon]
    ys = [item[1] for item in polygon]
    annotation = Annotation(
        class_name="object",
        class_id=0,
        bbox=(
            (min(xs) + max(xs)) / 2,
            (min(ys) + max(ys)) / 2,
            max(xs) - min(xs),
            max(ys) - min(ys),
        ),
        polygon=polygon,
    )
    canvas.set_annotations([annotation])
    canvas.select_annotation(annotation.id)

    corner = canvas.norm_to_pixel(*annotation.polygon[0])
    assert canvas._hit_test_handle(*corner) == "resize_obb_0"

    anchor = pixel_polygon[2]
    target = (
        anchor[0] - width_axis[0] * 80 - height_axis[0] * 40,
        anchor[1] - width_axis[1] * 80 - height_axis[1] * 40,
    )
    canvas._drag_ann_id = annotation.id
    canvas._drag_type = "resize_obb_0"
    canvas._drag_start_norm = annotation.polygon[0]
    canvas._drag_ann_snapshot = annotation.to_dict()
    canvas._handle_drag(target[0] / 200, target[1] / 100)

    resized = [(x * 200, y * 100) for x, y in annotation.polygon]
    assert math.isclose(resized[0][0], target[0], abs_tol=1e-9)
    assert math.isclose(resized[0][1], target[1], abs_tol=1e-9)
    assert math.isclose(resized[2][0], anchor[0], abs_tol=1e-9)
    assert math.isclose(resized[2][1], anchor[1], abs_tol=1e-9)

    first_edge = (resized[1][0] - resized[0][0], resized[1][1] - resized[0][1])
    second_edge = (resized[3][0] - resized[0][0], resized[3][1] - resized[0][1])
    assert math.isclose(
        first_edge[0] * second_edge[0] + first_edge[1] * second_edge[1],
        0.0,
        abs_tol=1e-9,
    )
    assert math.isclose(resized[0][0] + resized[2][0], resized[1][0] + resized[3][0], abs_tol=1e-9)
    assert math.isclose(resized[0][1] + resized[2][1], resized[1][1] + resized[3][1], abs_tol=1e-9)


def test_four_point_polygon_without_obb_bbox_keeps_polygon_vertex_editing():
    canvas = AnnotationCanvas()
    canvas.set_pixmap(QPixmap(200, 100))
    canvas.set_obb_editing_enabled(True)
    annotation = Annotation(
        class_name="region",
        class_id=0,
        polygon=[(0.2, 0.2), (0.8, 0.2), (0.7, 0.8), (0.3, 0.7)],
    )
    canvas.set_annotations([annotation])
    canvas.select_annotation(annotation.id)

    corner = canvas.norm_to_pixel(*annotation.polygon[0])

    assert canvas._hit_test_handle(*corner) == "poly_vertex_0"
