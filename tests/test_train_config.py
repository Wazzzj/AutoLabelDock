from src.engine.trainer import TrainConfig


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
        copy_paste=0.0,
    )

    args = config.to_train_args()

    assert args["mosaic"] == 0.0
    assert args["mixup"] == 0.0
    assert args["copy_paste"] == 0.0
