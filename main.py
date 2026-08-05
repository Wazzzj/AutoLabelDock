"""AutoLabel Dock — entry point."""

import os
import sys

# Ultralytics generates result plots from the background training QThread.
# GUI Matplotlib backends (notably PyCharm's backend_interagg) are not
# thread-safe and can corrupt the Windows native heap when training finishes.
# Agg keeps all saved training plots without creating GUI objects.
os.environ["MPLBACKEND"] = "Agg"

from PyQt5.QtWidgets import QApplication

from src.app import MainWindow
from src.core.encoding_utils import configure_utf8_environment
from src.ui.icons import app_icon
from src.ui.theme import apply_theme
from src.utils.logging_config import setup_logging


def main() -> int:
    configure_utf8_environment()
    setup_logging()

    app = QApplication(sys.argv)
    app.setWindowIcon(app_icon())

    # 内部同时设置中文字体和主题
    apply_theme(app)

    window = MainWindow()
    window.show()

    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
