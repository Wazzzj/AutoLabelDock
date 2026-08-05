import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from src.core.config import AppConfig
from src.ui.train_panel import TrainPanel


def _qapp():
    return QApplication.instance() or QApplication([])


def test_train_data_version_selection_is_saved_and_restored(tmp_path):
    app = _qapp()
    config_path = tmp_path / "config.json"
    app_config = AppConfig()
    panel = TrainPanel(app_config=app_config, config_path=config_path)
    panel.set_available_data_folders(
        ["v1", "v2"],
        default_folder="v2",
        preserve_selection=False,
    )
    assert panel.get_data_folder_filter() == "v2"

    panel._data_folder_filter_combo.setCurrentIndex(
        panel._data_folder_filter_combo.findData("v1")
    )
    panel.save_last_train_settings()

    restored_config = AppConfig.load(config_path)
    restored = TrainPanel(app_config=restored_config, config_path=config_path)
    restored.set_available_data_folders(
        ["v1", "v2"],
        default_folder="v2",
        preserve_selection=False,
    )

    assert restored.get_data_folder_filter() == "v1"
    assert restored.get_train_config(
        "", model="model.pt", log=False
    ).dataset_data_folder == "v1"
    assert app is not None
