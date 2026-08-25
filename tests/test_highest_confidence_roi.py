from src.controllers.model import ModelController
from src.core.annotation import Annotation, retain_highest_confidence_roi
from src.ui.model_panel import ModelPanel


def _qt_app():
    from PyQt5.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _annotation(confidence: float) -> Annotation:
    return Annotation(
        class_name="defect",
        class_id=0,
        bbox=(0.5, 0.5, 0.2, 0.2),
        confidence=confidence,
        confirmed=False,
        source="auto",
    )


def test_roi_rule_is_optional_and_disabled_by_default():
    annotations = [_annotation(0.2), _annotation(0.9)]

    assert retain_highest_confidence_roi(annotations) == annotations


def test_model_panel_persists_optional_roi_rule():
    app = _qt_app()
    panel = ModelPanel()

    assert not panel.should_retain_highest_confidence_roi()

    panel.apply_panel_settings({"retain_highest_confidence_roi": True})

    assert panel.should_retain_highest_confidence_roi()
    assert panel.get_panel_settings()["retain_highest_confidence_roi"] is True
    panel.close()
    assert app is not None


def test_roi_rule_keeps_only_the_highest_confidence_annotation():
    low = _annotation(0.2)
    highest = _annotation(0.9)
    medium = _annotation(0.6)

    assert retain_highest_confidence_roi(
        [low, highest, medium], enabled=True,
    ) == [highest]


def test_roi_rule_keeps_first_annotation_when_confidence_ties():
    first = _annotation(0.9)
    second = _annotation(0.9)

    assert retain_highest_confidence_roi(
        [first, second], enabled=True,
    ) == [first]


class _FakeBoxes:
    def __init__(self, confidences):
        self.conf = list(confidences)

    def __len__(self):
        return len(self.conf)


class _FakeNativeResult:
    def __init__(self, confidences):
        self.boxes = _FakeBoxes(confidences)
        self.obb = None

    def __getitem__(self, index):
        return _FakeNativeResult(self.boxes.conf[index])


def test_native_roi_rule_slices_the_complete_result_to_the_best_box():
    result = _FakeNativeResult([0.4, 0.95, 0.8])

    filtered = ModelController._retain_highest_confidence_native_roi(
        result, enabled=True,
    )

    assert filtered.boxes.conf == [0.95]


def test_native_roi_rule_returns_original_result_when_disabled():
    result = _FakeNativeResult([0.4, 0.95])

    assert ModelController._retain_highest_confidence_native_roi(
        result, enabled=False,
    ) is result
