"""AutoLabel Dock — entry point."""

import os
import sys

# Keep user-installed binary wheels out of a Conda/venv process. This must run
# before importing NumPy, OpenCV, Qt, PyTorch, or any application module.
os.environ["PYTHONNOUSERSITE"] = "1"
os.environ["MPLBACKEND"] = "Agg"

from src.utils.runtime_env import disable_user_site_packages

disable_user_site_packages()


def main() -> int:
    # Imports stay inside main(). Windows DataLoader workers execute this file
    # as __mp_main__; keeping the top level GUI-free prevents every worker from
    # loading and later tearing down Qt DLLs.
    from PyQt5.QtWidgets import QApplication

    from src.app import MainWindow
    from src.core.encoding_utils import configure_utf8_environment
    from src.ui.icons import app_icon
    from src.ui.theme import apply_theme
    from src.utils.logging_config import setup_logging
    from src.utils.runtime_env import configure_headless_matplotlib

    configure_utf8_environment()
    setup_logging()
    configure_headless_matplotlib()
    app = QApplication(sys.argv)
    app.setWindowIcon(app_icon())

    # 内部同时设置中文字体和主题
    apply_theme(app)

    window = MainWindow()
    window.show()

    return app.exec_()


def run_training_process() -> int:
    """Dispatch the PyInstaller executable's Qt-free training child mode."""
    from src.engine.train_process import main as train_process_main

    return train_process_main(sys.argv[2:])


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    if len(sys.argv) > 1 and sys.argv[1] == "--train-process":
        raise SystemExit(run_training_process())
    raise SystemExit(main())
