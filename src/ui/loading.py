"""Small reusable loading indicator backed by configured static resources."""
from __future__ import annotations

from PyQt5.QtCore import Qt, QTimer, QRectF, QByteArray
from PyQt5.QtGui import QPainter, QPixmap, QTransform
from PyQt5.QtSvg import QSvgRenderer
from PyQt5.QtWidgets import QLabel, QWidget, QHBoxLayout

from src.core.resources import LOADING_SVG
from src.ui.theme import text_style


class SpinningIcon(QLabel):
    """A QLabel that continuously rotates the bundled loading SVG."""

    def __init__(self, size: int = 22, parent=None):
        super().__init__(parent)
        self._size = size
        self._angle = 0
        self._base = self._render_base_pixmap(size)
        self.setFixedSize(size, size)
        self._timer = QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._tick)
        self._paint_current()

    def start(self) -> None:
        self._timer.start()
        self.show()

    def stop(self) -> None:
        self._timer.stop()
        self.hide()

    def _tick(self) -> None:
        self._angle = (self._angle + 30) % 360
        self._paint_current()

    def _render_base_pixmap(self, size: int) -> QPixmap:
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)

        if not LOADING_SVG.exists():
            return pixmap
        renderer = QSvgRenderer(QByteArray(LOADING_SVG.read_bytes()))
        painter = QPainter(pixmap)
        renderer.render(painter, QRectF(0, 0, size, size))
        painter.end()
        return pixmap

    def _paint_current(self) -> None:
        target = QPixmap(self._size, self._size)
        target.fill(Qt.transparent)
        painter = QPainter(target)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        transform = QTransform()
        transform.translate(self._size / 2, self._size / 2)
        transform.rotate(self._angle)
        transform.translate(-self._size / 2, -self._size / 2)
        painter.setTransform(transform)
        painter.drawPixmap(0, 0, self._base)
        painter.end()
        self.setPixmap(target)


class LoadingRow(QWidget):
    """Inline spinner plus status text."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.spinner = SpinningIcon(18)
        self.label = QLabel(text)
        self.label.setStyleSheet(text_style("hint"))
        layout.addWidget(self.spinner)
        layout.addWidget(self.label, 1)
        self.setVisible(False)

    def start(self, text: str) -> None:
        self.label.setText(text)
        self.setVisible(True)
        self.spinner.start()

    def stop(self) -> None:
        self.spinner.stop()
        self.setVisible(False)
