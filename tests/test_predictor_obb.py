from types import SimpleNamespace

import numpy as np

from src.engine.predictor import Predictor


class _FakeObbPredictor(Predictor):
    def predict_native(self, *args, **kwargs):
        obb = SimpleNamespace(
            cls=np.array([0.0]),
            conf=np.array([0.91]),
            xyxyxyxyn=np.array([
                [[0.1, 0.2], [0.7, 0.1], [0.8, 0.6], [0.2, 0.7]],
            ]),
        )
        return [SimpleNamespace(
            orig_shape=(100, 200),
            obb=obb,
            boxes=None,
            keypoints=None,
            masks=None,
        )]


def test_predictor_converts_ultralytics_obb_to_four_point_annotation():
    predictor = _FakeObbPredictor(SimpleNamespace(names={0: "label"}))

    annotations, image_size = predictor.predict_with_size(
        "sample.jpg",
        project_classes=["label"],
    )

    assert image_size == (200, 100)
    assert len(annotations) == 1
    assert annotations[0].polygon == [
        (0.1, 0.2), (0.7, 0.1), (0.8, 0.6), (0.2, 0.7),
    ]
    assert annotations[0].bbox == (0.45, 0.4, 0.7, 0.6)
