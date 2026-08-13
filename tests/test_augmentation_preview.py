import numpy as np

from src.ui.augmentation_preview import (
    _PreviewFrame,
    _augment_frame,
    _compose_overlay,
)


def _solid_frame(color, *, mask=False):
    image = np.full((96, 96, 3), color, dtype=np.uint8)
    overlay = np.zeros((96, 96, 4), dtype=np.uint8)
    masks = []
    if mask:
        object_mask = np.zeros((96, 96), dtype=np.uint8)
        object_mask[20:50, 10:40] = 255
        masks.append(object_mask)
        overlay[20:50, 10:12] = (255, 255, 0, 255)
    return _PreviewFrame(image, overlay, masks, int(mask))


def test_preview_applies_geometric_color_and_flip_params_deterministically():
    source = _solid_frame((80, 120, 180), mask=True)
    params = {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 12.0,
        "translate": 0.1,
        "scale": 0.3,
        "shear": 4.0,
        "perspective": 0.0005,
        "flipud": 0.0,
        "fliplr": 1.0,
    }

    first, first_ops = _augment_frame(source, [source], params, 1234, include_mix=False)
    second, second_ops = _augment_frame(source, [source], params, 1234, include_mix=False)

    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.overlay, second.overlay)
    assert first_ops == second_ops
    assert first.image.shape == source.image.shape
    assert not np.array_equal(first.image, source.image)
    assert any(item.startswith("旋转 ") for item in first_ops)
    assert any(item.startswith("缩放 ") for item in first_ops)
    assert any(item.startswith("平移 ") for item in first_ops)
    assert any(item.startswith("剪切 ") for item in first_ops)
    assert any(item.startswith("透视 ") for item in first_ops)
    assert "左右翻转" in first_ops


def test_full_training_sample_uses_multiple_images_for_mosaic():
    source = _solid_frame((255, 0, 0))
    donors = [
        _solid_frame((0, 255, 0)),
        _solid_frame((0, 0, 255)),
        _solid_frame((255, 255, 0)),
    ]
    params = {
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
    }

    augmented, operations = _augment_frame(source, donors, params, 7, include_mix=True)
    colors = np.unique(augmented.image.reshape(-1, 3), axis=0)

    assert operations == ["Mosaic"]
    assert len(colors) >= 2


def test_annotation_overlay_is_rendered_separately_from_training_pixels():
    frame = _solid_frame((30, 40, 50), mask=True)

    composed = _compose_overlay(frame)

    assert np.array_equal(frame.image[20, 10], np.array([30, 40, 50]))
    assert np.array_equal(composed[20, 10], np.array([255, 255, 0]))
