"""Scheme-A application shell widgets: nav rail, top bar, task chips, palette.

自绘"统一工作台"壳层（对应 ui_redesign/a-*.html 原型）：
- NavRail    左侧 56px 图标导航栏（徽标计数 + 闭环进度环 + 设置入口）
- LoopRing   闭环进度环：绿色弧=已确认占比，训练中呼吸脉冲
- TopBar     顶部上下文条：项目切换器 + ⌘K 命令入口 + 图片数/设备徽标
- TaskChip   底部常驻任务条组件（训练/批量标注进度，跨页面可见）

组件只负责展示与交互信号，业务编排仍在 MainWindow。
"""
from __future__ import annotations

from typing import Callable

from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPoint
from PyQt5.QtGui import QColor, QFontMetrics, QPainter, QPen, QFont
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.ui.icons import icon
from src.ui.theme import PALETTE

_RAIL_WIDTH = 56
_TOPBAR_HEIGHT = 44


# ────────────────────────────────────────────────────────────
# 闭环进度环
# ────────────────────────────────────────────────────────────
class LoopRing(QWidget):
    """导航栏底部的闭环进度环（签名元素）。

    绿色弧 = 已确认图片占比；黄色弧 = 待确认占比；
    训练进行时外圈呼吸脉冲，环心显示百分比。
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedSize(40, 40)
        self.setCursor(Qt.PointingHandCursor)
        self._confirmed_ratio = 0.0
        self._pending_ratio = 0.0
        self._training = False
        self._pulse_phase = 0.0
        self._tooltip_text = "暂无项目"
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self.setToolTip(self._tooltip_text)

    def set_stats(self, confirmed: int, pending: int, total: int) -> None:
        self._confirmed_ratio = (confirmed / total) if total else 0.0
        self._pending_ratio = (pending / total) if total else 0.0
        pct = round(self._confirmed_ratio * 100)
        self._tooltip_text = (
            f"闭环进度\n已确认 {confirmed}/{total} · {pct}%"
            + (f"\n待确认 {pending}" if pending else "")
            + ("\n● 训练进行中" if self._training else "")
        )
        self.setToolTip(self._tooltip_text)
        self.update()

    def set_training(self, active: bool) -> None:
        if self._training == active:
            return
        self._training = active
        if active:
            self._timer.start(70)
        else:
            self._timer.stop()
            self.update()

    def _tick(self) -> None:
        self._pulse_phase = (self._pulse_phase + 0.06) % 1.0
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        from PyQt5.QtCore import QRectF

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        cx = self.width() / 2
        cy = self.height() / 2
        radius = 15.0
        ring = QRectF(cx - radius, cy - radius, radius * 2, radius * 2)

        # 呼吸脉冲（训练中）
        if self._training:
            phase = self._pulse_phase
            alpha = int(120 * (1 - phase))
            spread = 2 + phase * 5
            accent = QColor(PALETTE["primary"])
            accent.setAlpha(alpha)
            painter.setPen(QPen(accent, 1.2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QRectF(cx - radius - spread, cy - radius - spread,
                                       (radius + spread) * 2, (radius + spread) * 2))
            painter.setPen(QPen(QColor(PALETTE["primary"]), 1.2))
            painter.drawEllipse(QRectF(cx - radius - 3.5, cy - radius - 3.5,
                                       (radius + 3.5) * 2, (radius + 3.5) * 2))

        # 底环
        painter.setPen(QPen(QColor(PALETTE["line"]), 3.2))
        painter.drawEllipse(ring)

        # 已确认弧（绿，从顶部顺时针）
        span_ok = int(-16 * 360 * self._confirmed_ratio)
        if span_ok:
            pen = QPen(QColor(PALETTE["success"]), 3.2)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.drawArc(ring, 90 * 16, span_ok)
        # 待确认弧（黄，紧随其后）
        span_pending = int(-16 * 360 * self._pending_ratio)
        if span_pending:
            pen = QPen(QColor(PALETTE["warning"]), 3.2)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.drawArc(ring, int(90 * 16 + span_ok), span_pending)

        # 环心百分比
        painter.setPen(QColor(PALETTE["text"]))
        painter.setFont(QFont(self.font().family(), 7, QFont.DemiBold))
        painter.drawText(self.rect(), Qt.AlignCenter, f"{round(self._confirmed_ratio * 100)}%")
        painter.end()


# ────────────────────────────────────────────────────────────
# 左侧导航栏
# ────────────────────────────────────────────────────────────
class NavRail(QWidget):
    """56px 图标导航栏。page_requested(key) 驱动页面栈切换。"""

    page_requested = pyqtSignal(str)
    settings_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("NavRail")
        self.setFixedWidth(_RAIL_WIDTH)
        self.setStyleSheet(
            "#NavRail{background:%s;border:none;border-right:1px solid %s;}"
            "#NavRailButton{background:transparent;border:1px solid transparent;"
            "border-radius:10px;padding:0;}"
            "#NavRailButton:hover{background:%s;}"
            "#NavRailButton:checked{background:%s;"
            "border:1px solid %s;}"
            % (PALETTE["bg_deep"], PALETTE["line"],
               PALETTE["panel"], PALETTE["primary_soft"], PALETTE["line_strong"])
        )
        self._buttons: dict[str, QToolButton] = {}
        self._icons: dict[str, str] = {}
        self._badges: dict[str, QLabel] = {}
        self._enabled_keys: set[str] = set()
        self._current: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 10, 0, 12)
        layout.setSpacing(2)
        layout.setAlignment(Qt.AlignHCenter)

        logo = QLabel(self)
        logo.setPixmap(NavRail._aperture_pixmap(30))
        logo.setAlignment(Qt.AlignCenter)
        logo.setContentsMargins(0, 0, 0, 8)
        layout.addWidget(logo)

        self.add_page("home", "welcome", "主页")
        self.add_page("label", "label_page", "标注")
        self.add_page("preview", "eye", "预览")
        self.add_page("train", "train_tab", "训练")
        self.add_page("models", "model_tab", "模型")
        self.add_page("tools", "script_tab", "小工具")

        layout.addStretch(1)

        self._loop_ring = LoopRing(self)
        layout.addWidget(self._loop_ring, 0, Qt.AlignHCenter)
        ring_spacer = QLabel(self)
        ring_spacer.setFixedHeight(6)
        layout.addWidget(ring_spacer)

        settings_btn = QToolButton(self)
        settings_btn.setObjectName("NavRailButton")
        settings_btn.setFixedSize(42, 42)
        settings_btn.setIcon(icon("settings", PALETTE["text_subtle"], 20))
        settings_btn.setToolTip("全局设置（菜单：文件 / 编辑）")
        settings_btn.clicked.connect(self.settings_requested.emit)
        layout.addWidget(settings_btn, 0, Qt.AlignHCenter)

        self.set_current("home")

    @staticmethod
    def _aperture_pixmap(size: int):
        """光圈标志：外环(暗) + 内环(主色) + 中心点。"""
        from PyQt5.QtCore import QRectF
        from PyQt5.QtGui import QPixmap

        pm = QPixmap(size, size)
        pm.fill(Qt.transparent)
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.Antialiasing)
        cx = cy = size / 2
        painter.setPen(QPen(QColor(PALETTE["line"]), max(1, int(size * 0.06))))
        painter.drawEllipse(QRectF(cx - size * 0.42, cy - size * 0.42,
                                   size * 0.84, size * 0.84))
        painter.setPen(QPen(QColor(PALETTE["primary"]), max(1, int(size * 0.06))))
        painter.drawEllipse(QRectF(cx - size * 0.25, cy - size * 0.25,
                                   size * 0.5, size * 0.5))
        painter.setBrush(QColor(PALETTE["primary"]))
        painter.setPen(Qt.NoPen)
        r = size * 0.08
        painter.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))
        painter.end()
        return pm

    def add_page(self, key: str, icon_name: str, title: str) -> None:
        btn = QToolButton(self)
        btn.setObjectName("NavRailButton")
        btn.setCheckable(True)
        btn.setFixedSize(42, 42)
        btn.setIcon(icon(icon_name, PALETTE["text_subtle"], 20))
        self._icons[key] = icon_name
        btn.setToolTip(title)
        btn.clicked.connect(lambda _=False, k=key: self.page_requested.emit(k))
        self._buttons[key] = btn
        self.layout().addWidget(btn, 0, Qt.AlignHCenter)

    def pages(self) -> list[str]:
        return list(self._buttons.keys())

    def set_enabled_pages(self, keys: list[str]) -> None:
        """导航按钮始终可点；未打开项目时的页面切换由 MainWindow 给出提示。"""
        self._enabled_keys = set(self._buttons.keys())
        for btn in self._buttons.values():
            btn.setEnabled(True)

    def set_current(self, key: str) -> None:
        self._current = key
        for k, btn in self._buttons.items():
            checked = k == key
            btn.setChecked(checked)
            color = PALETTE["primary"] if checked else PALETTE["text_subtle"]
            btn.setIcon(icon(self._icons[k], color, 20))

    def current(self) -> str | None:
        return self._current

    def set_badge(self, key: str, text: str | None, tone: str = "red") -> None:
        """tone: red | blue | dim。text=None 隐藏徽标。"""
        if key not in self._buttons:
            return
        badge = self._badges.get(key)
        if text is None or not text:
            if badge is not None:
                badge.hide()
            return
        if badge is None:
            badge = QLabel(self._buttons[key])
            badge.setObjectName("NavRailBadge")
            self._badges[key] = badge
        color = {"red": PALETTE["danger"], "blue": PALETTE["primary"],
                 "dim": PALETTE["line_strong"]}.get(tone, PALETTE["danger"])
        badge.setText(text)
        badge.setStyleSheet(
            f"background:{color};color:#FFFFFF;font-size:9px;font-weight:600;"
            "border-radius:7px;padding:0 4px;min-width:14px;min-height:14px;"
            "border:none;"
        )
        badge.adjustSize()
        badge.move(self._buttons[key].width() - badge.width() - 3, 3)
        badge.show()

    def loop_ring(self) -> LoopRing:
        return self._loop_ring


# ────────────────────────────────────────────────────────────
# 顶部上下文条
# ────────────────────────────────────────────────────────────
class TopBar(QWidget):
    """44px 顶部上下文条：右侧设备徽标（绿灯=可用）与版本号。"""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("TopBar")
        self.setFixedHeight(_TOPBAR_HEIGHT)
        self.setStyleSheet(
            "#TopBar{background:transparent;border:none;}"
            "#TopChip{background:%s;border:1px solid %s;border-radius:7px;"
            "padding:4px 10px;color:%s;font-size:11.5px;}"
            "#ProjectBadge{background:%s;border-radius:6px;color:%s;"
            "font-size:10px;font-weight:700;}"
            "#ProjectName{color:%s;font-size:13px;font-weight:700;}"
            "#TaskTypeChip{background:%s;border-radius:5px;color:%s;"
            "padding:3px 7px;font-size:10px;font-family:Menlo,'SF Mono',monospace;}"
            % (PALETTE["bg"], PALETTE["line"], PALETTE["text_muted"],
               PALETTE["primary_soft"], PALETTE["primary"], PALETTE["text"],
               PALETTE["panel_alt"], PALETTE["primary"])
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(8)

        self._project_context = QWidget(self)
        context_layout = QHBoxLayout(self._project_context)
        context_layout.setContentsMargins(0, 0, 0, 0)
        context_layout.setSpacing(7)
        self._project_badge = QLabel(self._project_context)
        self._project_badge.setObjectName("ProjectBadge")
        self._project_badge.setFixedSize(26, 26)
        self._project_badge.setAlignment(Qt.AlignCenter)
        self._project_name_label = QLabel(self._project_context)
        self._project_name_label.setObjectName("ProjectName")
        self._project_name_label.setMinimumWidth(0)
        self._project_name_label.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Preferred
        )
        self._task_type_chip = QLabel(self._project_context)
        self._task_type_chip.setObjectName("TaskTypeChip")
        context_layout.addWidget(self._project_badge)
        context_layout.addWidget(self._project_name_label, 1)
        context_layout.addWidget(self._task_type_chip)
        self._project_context.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Preferred
        )
        self._project_name_full = ""
        self._project_context.hide()
        layout.addWidget(self._project_context, 1)

        self._device_chip = QLabel(self)
        self._device_chip.setObjectName("TopChip")
        self.set_device("设备识别中…", active=False)
        layout.addWidget(self._device_chip)

        self._version_chip = QLabel(self)
        self._version_chip.setObjectName("TopChip")
        layout.addWidget(self._version_chip)

    @staticmethod
    def _device_markup(text: str, active: bool) -> str:
        dot = PALETTE["success"] if active else PALETTE["line_strong"]
        return f"<span style='color:{dot};'>●</span>&nbsp;&nbsp;{text}"

    def set_device(self, text: str, active: bool = True) -> None:
        self._device_chip.setText(self._device_markup(text, active))

    def set_version(self, text: str) -> None:
        self._version_chip.setText(text)

    def set_project_context(self, name: str, task_type: str) -> None:
        """Show the current project identity in the shell's left context slot."""
        clean_name = str(name or "").strip()
        if not clean_name:
            self.clear_project_context()
            return
        self._project_badge.setText(clean_name[:2])
        self._project_badge.setToolTip(clean_name)
        self._project_name_full = clean_name
        self._project_name_label.setText(clean_name)
        self._project_name_label.setToolTip(clean_name)
        self._task_type_chip.setText(str(task_type or "").strip().casefold())
        self._project_context.show()
        if self.isVisible():
            QTimer.singleShot(0, self._elide_project_name)

    def clear_project_context(self) -> None:
        """Hide and reset the current-project identity in the shell."""
        self._project_name_full = ""
        self._project_badge.clear()
        self._project_badge.setToolTip("")
        self._project_name_label.clear()
        self._project_name_label.setToolTip("")
        self._task_type_chip.clear()
        self._project_context.hide()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._project_context.isVisible():
            QTimer.singleShot(0, self._elide_project_name)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._project_context.isVisible():
            QTimer.singleShot(0, self._elide_project_name)

    def _elide_project_name(self) -> None:
        if not self._project_name_full:
            return
        available_width = self._project_name_label.contentsRect().width()
        if available_width <= 0:
            return
        text = QFontMetrics(self._project_name_label.font()).elidedText(
            self._project_name_full, Qt.ElideRight, available_width
        )
        self._project_name_label.setText(text)


# ────────────────────────────────────────────────────────────
# 底部常驻任务条组件
# ────────────────────────────────────────────────────────────
class TaskChip(QWidget):
    """状态栏内的常驻任务：图标 + 标题 + 细进度条 + 计数文本。"""

    def __init__(self, title: str, icon_name: str, color: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._color = color
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 10, 0)
        layout.setSpacing(6)

        self._icon_label = QLabel(self)
        self._icon_label.setPixmap(icon(icon_name, color, 13).pixmap(13, 13))
        layout.addWidget(self._icon_label)

        self._title_label = QLabel(title, self)
        self._title_label.setObjectName("TaskChipTitle")
        layout.addWidget(self._title_label)

        self._bar = QProgressBar(self)
        self._bar.setObjectName("TaskChipBar")
        self._bar.setFixedWidth(88)
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        layout.addWidget(self._bar)

        self._text_label = QLabel("", self)
        self._text_label.setObjectName("TaskChipText")
        layout.addWidget(self._text_label)

        self.hide()

    def start(self, title: str | None = None) -> None:
        if title:
            self._title_label.setText(title)
        self._bar.setRange(0, 0)  # busy 指示
        self.show()

    def update_progress(self, current: int, total: int, extra: str = "") -> None:
        self._bar.setRange(0, max(1, total))
        self._bar.setValue(current)
        text = f"{current}/{total}"
        if extra:
            text += f" · {extra}"
        self._text_label.setText(text)
        self.show()

    def set_text(self, text: str) -> None:
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._text_label.setText(text)
        self.show()

    def finish(self, message: str = "") -> None:
        if message:
            self._text_label.setText(message)
            QTimer.singleShot(4000, self.hide)
        else:
            self.hide()


# ────────────────────────────────────────────────────────────
