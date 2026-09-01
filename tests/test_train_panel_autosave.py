from PyQt5.QtCore import QEvent
from PyQt5.QtWidgets import QApplication

from src.core.config import AppConfig
from src.ui.train_panel import TrainPanel


_APP = None


def _qapp():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


def test_train_panel_saves_settings_when_input_loses_focus(tmp_path):
    _qapp()
    config_path = tmp_path / "config.json"
    app_config = AppConfig()
    panel = TrainPanel(app_config=app_config, config_path=config_path)

    panel._epochs_spin.setValue(321)
    panel.eventFilter(panel._epochs_spin, QEvent(QEvent.FocusOut))

    saved = AppConfig.load(config_path)
    assert saved.last_train_config["epochs"] == 321


def test_train_panel_saves_segment_params_and_builds_segment_config(tmp_path):
    _qapp()
    config_path = tmp_path / "config.json"
    app_config = AppConfig()
    panel = TrainPanel(app_config=app_config, config_path=config_path)

    panel._task_combo.setCurrentText("segment")
    panel._mask_ratio_spin.setValue(2)
    panel._overlap_mask_check.setChecked(False)
    panel._copy_paste_spin.setValue(0.35)
    panel._copy_paste_mode_combo.setCurrentText("mixup")
    panel.eventFilter(panel._mask_ratio_spin, QEvent(QEvent.FocusOut))

    saved = AppConfig.load(config_path)
    config = panel.get_train_config("data.yaml", "model.pt", log=False)
    assert saved.last_train_config["mask_ratio"] == 2
    assert saved.last_train_config["overlap_mask"] is False
    assert saved.last_train_config["copy_paste"] == 0.35
    assert saved.last_train_config["copy_paste_mode"] == "mixup"
    args = config.to_train_args()
    assert {
        key: args[key]
        for key in ("mask_ratio", "overlap_mask", "copy_paste", "copy_paste_mode")
    } == {
        "mask_ratio": 2,
        "overlap_mask": False,
        "copy_paste": 0.35,
        "copy_paste_mode": "mixup",
    }


def test_train_panel_shows_only_task_relevant_parameter_groups():
    _qapp()
    panel = TrainPanel()
    groups = {
        "detect_geo": panel._detect_geo_group,
        "detect_mix": panel._detect_mix_group,
        "segment": panel._segment_group,
        "classify": panel._classify_aug_group,
        "pose": panel._pose_group,
    }
    expected_visible = {
        "detect": {"detect_geo", "detect_mix"},
        "segment": {"detect_geo", "detect_mix", "segment"},
        "classify": {"classify"},
        "pose": {"detect_geo", "detect_mix", "pose"},
    }

    for task, visible_names in expected_visible.items():
        panel._task_combo.setCurrentText(task)
        actual_visible = {
            name for name, group in groups.items()
            if not group.isHidden()
        }
        assert actual_visible == visible_names
