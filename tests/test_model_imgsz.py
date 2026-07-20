from src.engine.predictor import Predictor


class _Obj:
    pass


def test_recommended_imgsz_reads_yolo_overrides_first():
    yolo = _Obj()
    yolo.overrides = {"imgsz": "512"}
    inner = _Obj()
    inner.args = {"imgsz": 640}
    yolo.model = inner

    assert Predictor(yolo).recommended_imgsz() == 512


def test_recommended_imgsz_reads_checkpoint_train_args():
    yolo = _Obj()
    yolo.ckpt = {"train_args": {"imgsz": 768}}

    assert Predictor(yolo).recommended_imgsz() == 768


def test_recommended_imgsz_uses_larger_side_for_rectangular_metadata():
    yolo = _Obj()
    inner = _Obj()
    inner.args = {"imgsz": [640, 960]}
    yolo.model = inner

    assert Predictor(yolo).recommended_imgsz() == 960


def test_recommended_imgsz_ignores_invalid_metadata():
    yolo = _Obj()
    yolo.overrides = {"imgsz": "bad"}
    yolo.ckpt = {"train_args": {"imgsz": 4}}

    assert Predictor(yolo).recommended_imgsz() is None
