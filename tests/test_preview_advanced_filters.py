import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from src.core.annotation import (
    Annotation,
    ImageAnnotation,
    annotation_area_text,
    annotation_display_label,
    annotation_geometry,
)
from src.core.tags import TagFilter
from src.ui.preview_panel import (
    NumericRange,
    PreviewAdvancedFilter,
    PreviewAdvancedFilterDialog,
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
        width=NumericRange(0.4, 0.6),
        confidence=NumericRange(0.9, 1.0),
    )

    assert not _image_matches_annotation_filters(
        image_annotation,
        "part",
        advanced_filter,
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
    assert annotation_area_text(annotation, (1000, 500)) == "面积 18.00% (90000 px²)"
    assert annotation_display_label(annotation, (1000, 500)) == (
        "obb | 面积 18.00% (90000 px²)"
    )


def test_bbox_area_text_uses_box_area_and_image_pixels():
    annotation = Annotation(
        class_name="connector",
        class_id=0,
        bbox=(0.5, 0.5, 0.2, 0.3),
    )

    assert annotation_area_text(annotation, (1000, 500)) == (
        "面积 6.00% (30000 px²)"
    )
    assert annotation_display_label(
        annotation,
        (1000, 500),
        include_pixels=False,
    ) == "connector | 面积 6.00%"


def test_advanced_filter_count_treats_tag_selection_as_one_condition():
    advanced_filter = PreviewAdvancedFilter(
        tag_filter=TagFilter(includes=("hard", "train"), mode="and"),
        area=NumericRange(0.1, 0.5),
        center_x=NumericRange(0.2, 0.8),
    )

    assert advanced_filter.active_count() == 3


def _qapp():
    return QApplication.instance() or QApplication([])


def test_range_fields_stay_editable_and_editing_enables_condition():
    app = _qapp()
    dialog = PreviewAdvancedFilterDialog(PreviewAdvancedFilter(), [])
    enabled, minimum, maximum = dialog._range_controls["width"]

    assert not enabled.isChecked()
    assert minimum.isEnabled()
    assert maximum.isEnabled()

    minimum.setValue(12.0)

    assert enabled.isChecked()
    assert dialog.value().width == NumericRange(0.12, 1.0)

    enabled.setChecked(False)

    assert minimum.isEnabled()
    assert maximum.isEnabled()
    assert not dialog.value().width.is_active()
    assert app is not None


def test_reset_leaves_all_conditions_inactive():
    app = _qapp()
    dialog = PreviewAdvancedFilterDialog(
        PreviewAdvancedFilter(width=NumericRange(0.12, 0.8)),
        [],
    )

    dialog._reset_all()

    assert dialog.value().active_count() == 0
    assert app is not None
