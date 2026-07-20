from PyQt5.QtCore import QEvent
from PyQt5.QtWidgets import QApplication

from src.core.config import AppConfig
from src.ui.train_panel import TrainPanel


def _qapp():
    return QApplication.instance() or QApplication([])


def test_train_panel_saves_settings_when_input_loses_focus(tmp_path):
    _qapp()
    config_path = tmp_path / "config.json"
    app_config = AppConfig()
    panel = TrainPanel(app_config=app_config, config_path=config_path)

    panel._epochs_spin.setValue(321)
    panel.eventFilter(panel._epochs_spin, QEvent(QEvent.FocusOut))

    saved = AppConfig.load(config_path)
    assert saved.last_train_config["epochs"] == 321
