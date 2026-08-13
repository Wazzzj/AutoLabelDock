import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QRect, Qt
from PyQt5.QtGui import QColor, QImage, QPainter
from PyQt5.QtWidgets import QApplication

from src.core.annotation import (
    Annotation,
    ImageAnnotation,
    annotation_area_text,
    annotation_center,
    annotation_display_label,
    annotation_geometry,
    annotation_pixel_geometry,
)
from src.ui.preview_panel import (
    NumericRange,
    PreviewAdvancedFilter,
    PreviewAdvancedFilterDialog,
    DetailPreviewCanvas,
    PreviewPanel,
    PreviewRoi,
    _annotation_control_result,
    _annotation_control_pen_style,
    _draw_annotation_overlays,
    _draw_annotation_label,
    _roi_contains_point,
    _image_control_result,
    _image_matches_annotation_filters,
)


def test_advanced_filter_requires_one_annotation_to_match_every_constraint():
    image_annotation = ImageAnnotation(
        image_path="sample.jpg",
        image_size=(100, 100),
        annotations=[
            Annotation(
                class_name="part",
                class_id=0,
                bbox=(0.2, 0.2, 0.1, 0.1),
                confidence=0.95,
            ),
            Annotation(
                class_name="part",
                class_id=0,
                bbox=(0.8, 0.8, 0.5, 0.5),
                confidence=0.20,
            ),
        ],
    )
    advanced_filter = PreviewAdvancedFilter(
        width=NumericRange(40, 60),
        confidence=NumericRange(0.9, 1.0),
    )

    assert _image_matches_annotation_filters(image_annotation, "part")
    assert not _image_control_result(
        image_annotation,
        {"part": advanced_filter},
    )


def test_polygon_filter_uses_polygon_area_and_bounds_center():
    annotation = Annotation(
        class_name="obb",
        class_id=0,
        bbox=(0.5, 0.5, 0.8, 0.8),
        polygon=[(0.2, 0.5), (0.5, 0.2), (0.8, 0.5), (0.5, 0.8)],
    )

    width, height, area, center_x, center_y = annotation_geometry(annotation)

    assert round(width, 6) == 0.6
    assert round(height, 6) == 0.6
    assert round(area, 6) == 0.18
    assert center_x == 0.5
    assert center_y == 0.5
    pixel_geometry = annotation_pixel_geometry(annotation, (1000, 500))
    assert pixel_geometry is not None
    assert tuple(round(value, 6) for value in pixel_geometry) == (
        600.0,
        300.0,
        90000.0,
        500.0,
        250.0,
    )
    assert annotation_area_text(annotation, (1000, 500)) == "面积 90000 px²"
    assert annotation_display_label(annotation, (1000, 500)) == (
        "obb | 面积 90000 px²"
    )


def test_bbox_area_text_uses_box_area_and_image_pixels():
    annotation = Annotation(
        class_name="connector",
        class_id=0,
        bbox=(0.5, 0.5, 0.2, 0.3),
    )

    assert annotation_area_text(annotation, (1000, 500)) == (
        "面积 30000 px²"
    )
    assert annotation_display_label(
        annotation,
        (1000, 500),
        include_pixels=False,
    ) == "connector | 面积 30000 px²"


def test_detection_control_counts_enabled_box_properties():
    advanced_filter = PreviewAdvancedFilter(
        area=NumericRange(0.1, 0.5),
        center_x=NumericRange(0.2, 0.8),
    )

    assert advanced_filter.active_count() == 2


def _qapp():
    return QApplication.instance() or QApplication([])


def test_range_fields_stay_editable_and_editing_enables_condition():
    app = _qapp()
    dialog = PreviewAdvancedFilterDialog({}, ["part"])
    enabled, minimum, maximum = dialog._range_controls["width"]

    assert not enabled.isChecked()
    assert minimum.isEnabled()
    assert maximum.isEnabled()

    minimum.setValue(12.0)

    assert enabled.isChecked()
    assert minimum.suffix() == " px"
    assert maximum.maximum() == 1_000_000.0
    assert dialog.value()["part"].width == NumericRange(12.0, 1_000_000.0)

    enabled.setChecked(False)

    assert minimum.isEnabled()
    assert maximum.isEnabled()
    assert "part" not in dialog.value()
    assert app is not None


def test_reset_leaves_all_conditions_inactive():
    app = _qapp()
    dialog = PreviewAdvancedFilterDialog(
        {"part": PreviewAdvancedFilter(width=NumericRange(12, 80))},
        ["part"],
    )

    dialog._reset_all()

    assert dialog.value() == {}
    assert app is not None


def test_pixel_filter_limits_use_image_dimensions_without_percentage_cap():
    app = _qapp()
    dialog = PreviewAdvancedFilterDialog(
        {},
        ["part"],
        pixel_limits={
            "width": 4096,
            "height": 2160,
            "area": 8_847_360,
            "center_x": 4096,
            "center_y": 2160,
        },
    )

    assert dialog._range_controls["width"][2].maximum() == 4096
    assert dialog._range_controls["height"][2].maximum() == 2160
    assert dialog._range_controls["area"][2].maximum() == 8_847_360
    assert dialog._range_controls["center_x"][2].maximum() == 4096
    assert dialog._range_controls["center_y"][2].maximum() == 2160
    assert app is not None


def test_detection_box_control_is_per_class_and_unconfigured_classes_are_neutral():
    image_size = (1000, 500)
    small = Annotation("scratch", 0, bbox=(0.5, 0.5, 0.1, 0.2), confidence=0.9)
    large = Annotation("scratch", 0, bbox=(0.5, 0.5, 0.4, 0.2), confidence=0.9)
    other = Annotation("dent", 1, bbox=(0.5, 0.5, 0.9, 0.9), confidence=0.1)
    rules = {
        "scratch": PreviewAdvancedFilter(width=NumericRange(80, 200)),
    }

    assert _annotation_control_result(small, image_size, rules) is True
    assert _annotation_control_result(large, image_size, rules) is False
    assert _annotation_control_result(other, image_size, rules) is None
    assert _annotation_control_result(small, image_size, rules, enabled=False) is None


def test_image_control_is_ng_when_any_controlled_detection_box_fails():
    annotation = ImageAnnotation(
        image_path="sample.jpg",
        image_size=(1000, 500),
        annotations=[
            Annotation("scratch", 0, bbox=(0.3, 0.5, 0.1, 0.2)),
            Annotation("scratch", 0, bbox=(0.7, 0.5, 0.4, 0.2)),
            Annotation("dent", 1, bbox=(0.5, 0.5, 0.9, 0.9)),
        ],
    )
    rules = {"scratch": PreviewAdvancedFilter(width=NumericRange(80, 200))}

    assert _image_control_result(annotation, rules) is False
    assert _image_control_result(annotation, rules, enabled=False) is None


def test_detection_control_uses_solid_for_ok_and_dashed_for_ng_boxes():
    image_size = (1000, 500)
    passed = Annotation("scratch", 0, bbox=(0.5, 0.5, 0.1, 0.2))
    failed = Annotation("scratch", 0, bbox=(0.5, 0.5, 0.4, 0.2))
    rules = {"scratch": PreviewAdvancedFilter(width=NumericRange(80, 200))}

    assert _annotation_control_pen_style(passed, image_size, rules, True) == Qt.SolidLine
    assert _annotation_control_pen_style(failed, image_size, rules, True) == Qt.DashLine


def test_control_dialog_keeps_independent_rules_for_each_defect_class():
    app = _qapp()
    dialog = PreviewAdvancedFilterDialog({}, ["scratch", "dent"])
    dialog._range_controls["width"][1].setValue(20)
    dialog._range_controls["width"][2].setValue(80)
    dialog._class_combo.setCurrentText("dent")
    dialog._range_controls["area"][1].setValue(100)
    dialog._range_controls["area"][2].setValue(1000)

    rules = dialog.value()

    assert rules["scratch"].width == NumericRange(20, 80)
    assert rules["dent"].area == NumericRange(100, 1000)
    assert app is not None


def test_preview_toolbar_restores_tag_filter_and_control_defaults_off():
    app = _qapp()
    panel = PreviewPanel()
    toolbar_widgets = [
        panel._toolbar.widgetForAction(action)
        for action in panel._toolbar.actions()
    ]

    assert panel._tag_filter_bar in toolbar_widgets
    assert toolbar_widgets.index(panel._tag_filter_bar) < toolbar_widgets.index(panel._control_btn)
    assert panel._control_enabled is False
    assert panel._control_rules == {}
    assert panel._control_btn.text() == "标注卡控（未开启）"
    panel.cleanup()
    assert app is not None


def test_preview_polygon_center_is_fully_transparent():
    app = _qapp()
    image = QImage(100, 100, QImage.Format_ARGB32)
    background = QColor("#123456")
    image.fill(background)
    annotation = ImageAnnotation(
        image_path="sample.jpg",
        image_size=(100, 100),
        annotations=[
            Annotation(
                "defect",
                0,
                polygon=[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)],
            )
        ],
    )
    painter = QPainter(image)
    _draw_annotation_overlays(
        painter,
        QRect(0, 0, 100, 100),
        annotation,
        {"defect": "#ff0000"},
    )
    painter.end()

    assert image.pixelColor(50, 50) == background
    assert app is not None


def test_rectangle_circle_and_polygon_roi_membership():
    rectangle = PreviewRoi("rectangle", ((0.1, 0.1), (0.5, 0.5)))
    circle = PreviewRoi("ellipse", ((0.2, 0.2), (0.8, 0.8)))
    polygon = PreviewRoi("polygon", ((0.1, 0.1), (0.9, 0.1), (0.5, 0.9)))

    assert _roi_contains_point(rectangle, (0.3, 0.3))
    assert not _roi_contains_point(rectangle, (0.7, 0.3))
    assert _roi_contains_point(circle, (0.5, 0.5))
    assert not _roi_contains_point(circle, (0.2, 0.2))
    assert _roi_contains_point(polygon, (0.5, 0.4))
    assert not _roi_contains_point(polygon, (0.1, 0.8))


def test_roi_outside_detection_is_dashed_and_excluded_from_control_result():
    image_size = (1000, 500)
    inside = Annotation("scratch", 0, bbox=(0.25, 0.5, 0.1, 0.2))
    outside_failed = Annotation("scratch", 0, bbox=(0.75, 0.5, 0.4, 0.2))
    roi = PreviewRoi("rectangle", ((0.0, 0.0), (0.5, 1.0)))
    rules = {"scratch": PreviewAdvancedFilter(width=NumericRange(80, 200))}
    annotation = ImageAnnotation(
        "sample.jpg",
        image_size,
        annotations=[inside, outside_failed],
    )

    assert _annotation_control_result(inside, image_size, rules, True, [roi]) is True
    assert _annotation_control_result(outside_failed, image_size, rules, True, [roi]) is None
    assert _annotation_control_pen_style(outside_failed, image_size, rules, True, [roi]) == Qt.DashLine
    assert _image_control_result(annotation, rules, True, [roi]) is True


def test_preview_roi_is_session_only_and_not_added_to_annotations():
    app = _qapp()
    panel = PreviewPanel()
    roi = PreviewRoi("rectangle", ((0.1, 0.1), (0.9, 0.9)))

    panel._on_preview_rois_changed([roi])

    assert panel._preview_rois == [roi]
    assert panel._control_enabled is True
    assert not hasattr(Annotation("defect", 0), "roi")
    panel.cleanup()
    assert app is not None


def test_global_roi_alone_controls_all_detection_classes():
    image_size = (1000, 500)
    inside = Annotation("scratch", 0, bbox=(0.25, 0.5, 0.1, 0.2))
    outside = Annotation("dent", 1, bbox=(0.75, 0.5, 0.1, 0.2))
    roi = PreviewRoi("rectangle", ((0.0, 0.0), (0.5, 1.0)))
    annotation = ImageAnnotation(
        "sample.jpg",
        image_size,
        annotations=[inside, outside],
    )

    assert _annotation_control_result(inside, image_size, {}, True, [roi]) is True
    assert _annotation_control_result(outside, image_size, {}, True, [roi]) is None
    assert _annotation_control_pen_style(outside, image_size, {}, True, [roi]) == Qt.DashLine
    assert _image_control_result(annotation, {}, True, [roi]) is True


def test_global_roi_with_only_outside_detections_is_ok():
    roi = PreviewRoi("ellipse", ((0.1, 0.1), (0.5, 0.9)))
    annotation = ImageAnnotation(
        "sample.jpg",
        (1000, 500),
        annotations=[Annotation("scratch", 0, bbox=(0.8, 0.5, 0.1, 0.2))],
    )

    assert _image_control_result(annotation, {}, True, [roi]) is True


def test_circle_roi_tool_constrains_drag_to_equal_displayed_width_and_height():
    app = _qapp()
    canvas = DetailPreviewCanvas(None, ImageAnnotation("sample.jpg", (1000, 500)), {})
    canvas.resize(1000, 500)
    canvas.set_roi_tool("ellipse")

    start = (0.1, 0.1)
    end = canvas._constrain_roi_end(start, (0.5, 0.5))

    displayed_width = abs(end[0] - start[0]) * canvas.sizeHint().width()
    displayed_height = abs(end[1] - start[1]) * canvas.sizeHint().height()
    assert round(displayed_width, 6) == round(displayed_height, 6)
    assert app is not None


def test_roi_can_move_and_is_clamped_inside_image():
    rectangle = PreviewRoi("rectangle", ((0.2, 0.3), (0.6, 0.7)))

    moved = DetailPreviewCanvas._move_roi(rectangle, 0.1, -0.2)
    clamped = DetailPreviewCanvas._move_roi(rectangle, 0.8, -0.8)

    assert tuple((round(x, 6), round(y, 6)) for x, y in moved.points) == (
        (0.3, 0.1),
        (0.7, 0.5),
    )
    assert tuple((round(x, 6), round(y, 6)) for x, y in clamped.points) == (
        (0.6, 0.0),
        (1.0, 0.4),
    )


def test_obb_uses_four_corner_centroid_for_global_roi_control():
    obb = Annotation(
        "oriented",
        0,
        bbox=(0.5, 0.5, 0.6, 0.4),
        polygon=[(0.2, 0.4), (0.6, 0.2), (0.8, 0.6), (0.4, 0.8)],
    )
    inside_roi = PreviewRoi("ellipse", ((0.35, 0.35), (0.65, 0.65)))
    outside_roi = PreviewRoi("rectangle", ((0.0, 0.0), (0.3, 0.3)))

    center = annotation_center(obb)

    assert center is not None
    assert tuple(round(value, 6) for value in center) == (0.5, 0.5)
    assert _annotation_control_pen_style(obb, (1000, 500), {}, True, [inside_roi]) == Qt.SolidLine
    assert _annotation_control_pen_style(obb, (1000, 500), {}, True, [outside_roi]) == Qt.DashLine


def test_instance_segment_uses_polygon_centroid_instead_of_bounds_center():
    # The extra mass on the left moves the real centroid away from the
    # axis-aligned bounds center, exercising segmentation-specific ROI logic.
    segment = Annotation(
        "surface",
        0,
        polygon=[
            (0.1, 0.1),
            (0.9, 0.1),
            (0.9, 0.3),
            (0.3, 0.3),
            (0.3, 0.9),
            (0.1, 0.9),
        ],
    )
    centroid_roi = PreviewRoi("rectangle", ((0.3, 0.3), (0.42, 0.42)))
    bounds_center_roi = PreviewRoi("rectangle", ((0.45, 0.45), (0.55, 0.55)))

    center = annotation_center(segment)

    assert center is not None
    assert tuple(round(value, 6) for value in center) == (0.371429, 0.371429)
    assert _annotation_control_pen_style(segment, (1000, 500), {}, True, [centroid_roi]) == Qt.SolidLine
    assert _annotation_control_pen_style(segment, (1000, 500), {}, True, [bounds_center_roi]) == Qt.DashLine


def test_polygon_roi_can_finish_and_connect_with_enter():
    app = _qapp()
    canvas = DetailPreviewCanvas(None, ImageAnnotation("sample.jpg", (1000, 500)), {})
    canvas.set_roi_tool("polygon")
    canvas._roi_polygon_points = [(0.1, 0.1), (0.8, 0.1), (0.5, 0.8)]

    assert canvas._finish_roi_polygon()
    assert canvas._rois == [PreviewRoi(
        "polygon",
        ((0.1, 0.1), (0.8, 0.1), (0.5, 0.8)),
    )]
    assert canvas._roi_polygon_points == []
    assert app is not None


def test_polygon_roi_detects_click_near_first_point_for_closing():
    app = _qapp()
    canvas = DetailPreviewCanvas(None, ImageAnnotation("sample.jpg", (1000, 500)), {})
    canvas.resize(1000, 500)
    canvas.set_roi_tool("polygon")
    canvas._roi_polygon_points = [(0.1, 0.1), (0.8, 0.1), (0.5, 0.8)]

    assert canvas._near_first_polygon_point((0.105, 0.105))
    assert not canvas._near_first_polygon_point((0.2, 0.2))
    assert app is not None


def test_preview_annotation_label_background_is_fully_transparent():
    app = _qapp()
    image = QImage(160, 80, QImage.Format_ARGB32)
    background = QColor("#123456")
    image.fill(background)
    painter = QPainter(image)
    _draw_annotation_label(
        painter,
        QRect(0, 0, 160, 80),
        (20, 40),
        "defect",
        QColor("#ff0000"),
        1.0,
    )
    painter.end()

    # Padding pixels around the glyphs stay unchanged because no label
    # background rectangle is painted.
    assert image.pixelColor(20, 25) == background
    assert app is not None
