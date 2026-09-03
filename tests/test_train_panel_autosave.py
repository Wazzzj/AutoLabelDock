from unittest.mock import patch

from PyQt5.QtCore import QEvent
from PyQt5.QtWidgets import QApplication, QFileDialog

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


def test_train_panel_uses_a_styled_task_popup_and_shows_selected_model_path(tmp_path):
    _qapp()
    panel = TrainPanel()
    selected_model = tmp_path / "models" / "best.pt"
    selected_model.parent.mkdir()
    selected_model.write_bytes(b"model")
    panel.show()
    _qapp().processEvents()

    assert panel._task_combo.view().objectName() == "trainTaskPopup"
    assert "background" in panel._task_combo.view().styleSheet()
    assert not hasattr(panel, "_btn_browse_model")
    assert panel._btn_choose_existing_model.text() == "选择已有模型"
    assert panel._selected_model_path.isReadOnly()
    assert not panel._model_combo.isVisible()

    with patch.object(
        QFileDialog, "getOpenFileName", return_value=(str(selected_model), "")
    ):
        panel._choose_existing_model()

    resolved_path = str(selected_model.resolve())
    assert panel._selected_model_path.text() == resolved_path
    assert panel._resolve_model_path() == resolved_path
    panel.close()


def test_train_panel_restores_a_local_model_path_named_like_an_official_model(tmp_path):
    _qapp()
    selected_model = tmp_path / "checkpoint" / "yolov8n.pt"
    selected_model.parent.mkdir()
    selected_model.write_bytes(b"model")
    panel = TrainPanel()

    panel.apply_template_params({"model": str(selected_model)})

    resolved_path = str(selected_model.resolve())
    assert panel._selected_model_path.text() == resolved_path
    assert panel.get_train_config("data.yaml", log=False).model == resolved_path
