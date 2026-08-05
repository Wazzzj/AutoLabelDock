from src.engine.trainer import TrainConfig
from src.core.train_templates import extract_task_params


def test_single_cls_is_passed_to_train_args_only_when_enabled():
    disabled = TrainConfig(data_yaml="data.yaml", model="model.pt", task="detect")
    assert "single_cls" not in disabled.to_train_args()

    enabled = TrainConfig(
        data_yaml="data.yaml",
        model="model.pt",
        task="detect",
        single_cls=True,
    )
    assert enabled.to_train_args()["single_cls"] is True


def test_detect_augmentation_values_are_passed_for_detection_tasks():
    config = TrainConfig(
        data_yaml="data.yaml",
        model="model.pt",
        task="segment",
        include_detect_params=False,
        mosaic=0.0,
        mixup=0.0,
    )

    args = config.to_train_args()

    assert args["mosaic"] == 0.0
    assert args["mixup"] == 0.0


def test_segmentation_params_are_passed_only_for_segmentation_tasks():
    segment = TrainConfig(
        data_yaml="data.yaml",
        model="model.pt",
        task="segment",
        mask_ratio=2,
        overlap_mask=False,
        copy_paste=0.35,
        copy_paste_mode="mixup",
    )
    detect = TrainConfig(
        data_yaml="data.yaml",
        model="model.pt",
        task="detect",
        mask_ratio=2,
    )

    assert segment.to_train_args()["mask_ratio"] == 2
    assert segment.to_train_args()["overlap_mask"] is False
    assert segment.to_train_args()["copy_paste"] == 0.35
    assert segment.to_train_args()["copy_paste_mode"] == "mixup"
    assert "mask_ratio" not in detect.to_train_args()
    assert "overlap_mask" not in detect.to_train_args()
    assert "copy_paste" not in detect.to_train_args()
    assert "copy_paste_mode" not in detect.to_train_args()
    assert extract_task_params(segment)["mask_ratio"] == 2
    assert extract_task_params(segment)["overlap_mask"] is False
    assert extract_task_params(segment)["copy_paste"] == 0.35
    assert extract_task_params(segment)["copy_paste_mode"] == "mixup"
    assert "mask_ratio" not in extract_task_params(detect)


def test_task_specific_params_do_not_leak_when_hidden_flags_are_enabled():
    classify = TrainConfig(
        data_yaml="data.yaml",
        model="model.pt",
        task="classify",
        include_detect_params=True,
        include_classify_params=True,
        include_pose_params=True,
    ).to_train_args()
    detect = TrainConfig(
        data_yaml="data.yaml",
        model="model.pt",
        task="detect",
        include_classify_params=True,
        include_pose_params=True,
    ).to_train_args()

    assert "mosaic" not in classify
    assert "pose" not in classify
    assert "erasing" in classify
    assert "erasing" not in detect
    assert "pose" not in detect
