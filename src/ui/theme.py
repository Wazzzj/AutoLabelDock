"""Scheme-A workbench theme (solid-color) shared style helpers for PyQt5.

主题：对齐 ui_redesign/a-*.html 原型 —— 石墨藏蓝面板 + 电光蓝主按钮（实色，无玻璃/渐变）。
- 背景：纯色 #0B0F16；面板 #101623；边框 #1E2838 / #2C3A50；文字 #E8EEF7。
- 主按钮文字深色 #04121F（原型 primary）。圆角 8、基础字号 13px。
- 所有 PALETTE 值保持 QColor 可解析的 hex 字符串。
"""
from __future__ import annotations
from PyQt5.QtGui import QColor, QFont, QFontDatabase, QPalette
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

UI_FONT_FAMILY = ""


def apply_application_font(app: QApplication) -> str:
    """选择系统中可用的中文字体并应用到整个程序。"""

    global UI_FONT_FAMILY

    available_fonts = set(QFontDatabase().families())

    candidates = [
        "Microsoft YaHei UI",
        "Microsoft YaHei",
        "微软雅黑",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "PingFang SC",
        "SimSun",
        "宋体",
    ]

    UI_FONT_FAMILY = next(
        (name for name in candidates if name in available_fonts),
        app.font().family(),
    )

    font = QFont(UI_FONT_FAMILY)
    font.setPointSize(10)
    font.setStyleStrategy(QFont.PreferAntialias)

    app.setFont(font)
    return UI_FONT_FAMILY


# 配色与 ui_redesign/a-*.html 设计稿一致（石墨藏蓝 / 钢青面板 / 电光蓝）
PALETTE = {
    "bg": "#0B0F16",
    "bg_deep": "#070A10",
    "canvas": "#0A0F16",
    "panel": "#101623",
    "panel_alt": "#141C2C",
    "panel_raised": "#1B2334",
    "line": "#1E2838",
    "line_strong": "#2C3A50",
    "text": "#E8EEF7",
    "text_muted": "#9FB0C8",
    "text_subtle": "#5D6E88",
    "ink": "#FFFFFF",
    "primary": "#4D9FFF",
    "primary_hover": "#66ADFF",
    "primary_pressed": "#3B85E6",
    "primary_soft": "#1E3A5F",
    "selection": "#4D9FFF",
    "selection_text": "#FFFFFF",
    "success": "#3ECF8E",
    "success_soft": "#112724",
    "danger": "#F16A5D",
    "danger_soft": "#3B2025",
    "warning": "#F5B83D",
    "warning_soft": "#3D2D16",
    "violet": "#A78BFA",
    "teal": "#22D3EE",
}

# Backwards-compatible alias for older modules/tests that still describe the
# palette as Catppuccin. New code should use PALETTE directly.
MOCHA = {
    "base": PALETTE["bg"],
    "mantle": PALETTE["panel"],
    "crust": PALETTE["bg_deep"],
    "surface0": PALETTE["panel_alt"],
    "surface1": PALETTE["line"],
    "surface2": PALETTE["line_strong"],
    "overlay0": PALETTE["text_subtle"],
    "text": PALETTE["text"],
    "subtext0": PALETTE["text_muted"],
    "green": PALETTE["success"],
    "blue": PALETTE["primary"],
    "red": PALETTE["danger"],
    "peach": PALETTE["warning"],
    "yellow": PALETTE["warning"],
    "mauve": PALETTE["violet"],
    "teal": PALETTE["teal"],
    "sky": PALETTE["primary_hover"],
    "lavender": PALETTE["violet"],
}


def _rgba(hex_color: str, alpha: float) -> str:
    """Convert a #RRGGBB hex color into an rgba() QSS string."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return hex_color
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _refresh_style(widget) -> None:
    style = widget.style()
    if style is None:
        return
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def set_widget_role(widget, role: str):
    """Attach a QSS role dynamic property and refresh the widget style."""
    widget.setProperty("role", role)
    if hasattr(widget, "setCursor") and role not in {"passive", "list-item"}:
        widget.setCursor(Qt.PointingHandCursor)
    _refresh_style(widget)
    return widget


def set_button_role(button, role: str):
    """Mark a button as a named action role for global QSS styling."""
    return set_widget_role(button, role)


def set_surface(widget, surface: str):
    """Mark a widget as a named surface for global QSS styling."""
    widget.setProperty("surface", surface)
    _refresh_style(widget)
    return widget


def text_style(role: str = "muted") -> str:
    styles = {
        "display": (
            f"color: {PALETTE['text']}; font-size: 32px; font-weight: 700;"
        ),
        "title": (
            f"color: {PALETTE['text']}; font-size: 18px; font-weight: 650;"
        ),
        "section": (
            f"color: {PALETTE['text_muted']}; font-size: 13px; font-weight: 650;"
        ),
        "body": f"color: {PALETTE['text']}; font-size: 13px;",
        "small": f"color: {PALETTE['text']}; font-size: 12px;",
        "muted": f"color: {PALETTE['text_muted']}; font-size: 12.5px;",
        "hint": f"color: {PALETTE['text_subtle']}; font-size: 12px;",
        "success": f"color: {PALETTE['success']}; font-size: 13px; font-weight: 650;",
        "warning": f"color: {PALETTE['warning']}; font-size: 13px; font-weight: 650;",
        "error": f"color: {PALETTE['danger']}; font-size: 12px;",
    }
    return styles.get(role, styles["muted"])


def chip_style(active: bool = False) -> str:
    bg = PALETTE["primary_soft"] if active else PALETTE["panel_alt"]
    fg = PALETTE["primary"] if active else PALETTE["text"]
    border = PALETTE["primary"] if active else PALETTE["line"]
    weight = "700" if active else "500"
    return (
        "QToolButton {"
        f" background-color: {bg};"
        f" color: {fg};"
        f" border: 1px solid {border};"
        " border-radius: 13px;"
        " padding: 4px 10px;"
        " font-size: 12px;"
        f" font-weight: {weight};"
        "}"
        f"QToolButton:hover {{ background-color: {PALETTE['primary_soft']};"
        f" border-color: {PALETTE['primary']}; }}"
    )


# 主按钮前景色（原型 .btn.primary 深色文字）
_PRIMARY_FG = "#04121F"

STYLESHEET = f"""
QWidget {{
    background-color: transparent;
    color: {PALETTE['text']};
    font-size: 13px;
    selection-background-color: {PALETTE['selection']};
    selection-color: {PALETTE['selection_text']};
}}

QMainWindow, QDialog {{
    background-color: {PALETTE['bg']};
}}

QWidget[surface="panel"] {{
    background-color: {PALETTE['panel']};
    border: 1px solid {PALETTE['line']};
    border-radius: 8px;
}}

QTabWidget::pane {{
    border: none;
    background-color: {PALETTE['bg']};
}}

QTabBar {{
    background-color: {PALETTE['panel']};
}}

QTabBar::tab {{
    background-color: transparent;
    color: {PALETTE['text_muted']};
    padding: 9px 18px;
    border: none;
    margin-right: 1px;
    min-height: 24px;
}}

QTabBar::tab:selected {{
    background-color: {PALETTE['primary_soft']};
    color: {PALETTE['text']};
    border-bottom: 2px solid {PALETTE['primary']};
}}

QTabBar::tab:hover {{
    color: {PALETTE['text']};
    background-color: {PALETTE['panel_alt']};
}}

QToolBar {{
    background-color: {PALETTE['panel']};
    border: 1px solid {PALETTE['line']};
    border-radius: 8px;
    spacing: 6px;
    padding: 8px 10px;
}}

QToolBar::separator {{
    background-color: {PALETTE['line_strong']};
    width: 1px;
    margin: 4px 7px;
}}

QPushButton {{
    background-color: {PALETTE['panel_alt']};
    color: {PALETTE['text']};
    border: 1px solid {PALETTE['line_strong']};
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 13px;
    font-weight: 600;
    min-height: 16px;
}}

QPushButton:hover {{
    border-color: {PALETTE['primary']};
    color: {PALETTE['primary']};
}}

QPushButton:focus {{
    border-color: {PALETTE['primary']};
}}

QPushButton:pressed {{
    background-color: {PALETTE['panel_raised']};
}}

QPushButton:checked {{
    background-color: {PALETTE['primary_soft']};
    color: {PALETTE['primary']};
    border-color: {PALETTE['primary']};
}}

QPushButton:disabled {{
    color: {PALETTE['text_subtle']};
    background-color: {PALETTE['panel']};
    border-color: {PALETTE['line']};
}}

QPushButton[role="primary"] {{
    background-color: {PALETTE['primary']};
    color: {_PRIMARY_FG};
    border-color: {PALETTE['primary']};
    font-weight: 700;
}}

QPushButton[role="primary"]:hover {{
    background-color: {PALETTE['primary_hover']};
    border-color: {PALETTE['primary_hover']};
    color: {_PRIMARY_FG};
}}

QPushButton[role="primary"]:pressed {{
    background-color: {PALETTE['primary_pressed']};
    border-color: {PALETTE['primary_pressed']};
    color: {_PRIMARY_FG};
}}

QPushButton[role="danger"] {{
    background-color: {PALETTE['danger_soft']};
    color: {PALETTE['danger']};
    border-color: {PALETTE['danger']};
}}

QPushButton[role="danger"]:hover {{
    background-color: {PALETTE['danger']};
    color: {PALETTE['bg']};
}}

QPushButton[role="success"] {{
    background-color: {PALETTE['success_soft']};
    color: {PALETTE['success']};
    border-color: {PALETTE['success']};
}}

QPushButton[role="secondary"] {{
    background-color: {PALETTE['panel_alt']};
    color: {PALETTE['text']};
}}

QPushButton[role="icon"] {{
    min-width: 26px;
    max-width: 32px;
    padding: 4px;
}}

QPushButton[role="icon-danger"] {{
    min-width: 24px;
    max-width: 28px;
    padding: 3px;
    color: {PALETTE['danger']};
}}

QPushButton[role="list-item"] {{
    text-align: left;
    padding: 6px 8px;
    border-color: transparent;
    background-color: transparent;
    font-weight: 500;
}}

QPushButton[role="list-item"]:checked {{
    background-color: {PALETTE['primary_soft']};
    border-color: {PALETTE['primary']};
    color: {PALETTE['primary']};
}}

QPushButton[role="primary"]:disabled,
QPushButton[role="danger"]:disabled,
QPushButton[role="success"]:disabled,
QPushButton[role="secondary"]:disabled {{
    color: {PALETTE['text_subtle']};
    background-color: {PALETTE['panel']};
    border-color: {PALETTE['line']};
}}

QToolButton {{
    background-color: transparent;
    color: {PALETTE['text_muted']};
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 4px 8px;
}}

QToolButton:hover {{
    background-color: {PALETTE['panel_alt']};
    color: {PALETTE['text']};
    border-color: {PALETTE['line']};
}}

QToolButton:checked {{
    background-color: {PALETTE['primary_soft']};
    color: {PALETTE['primary']};
    border-color: {PALETTE['primary']};
}}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {PALETTE['bg']};
    color: {PALETTE['text']};
    border: 1px solid {PALETTE['line']};
    border-radius: 6px;
    padding: 5px 8px;
    font-size: 12.5px;
    selection-background-color: {PALETTE['selection']};
    selection-color: {PALETTE['selection_text']};
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {PALETTE['primary']};
}}

QComboBox::drop-down {{
    border: none;
    padding-right: 8px;
}}

QTextEdit, QPlainTextEdit {{
    background-color: {PALETTE['bg']};
    color: {PALETTE['text']};
    border: 1px solid {PALETTE['line']};
    border-radius: 6px;
    padding: 6px;
    font-size: 12px;
    selection-background-color: {PALETTE['selection']};
    selection-color: {PALETTE['selection_text']};
}}

QLabel {{
    background-color: transparent;
}}

QGroupBox {{
    background-color: {PALETTE['panel']};
    border: 1px solid {PALETTE['line']};
    border-radius: 8px;
    margin-top: 12px;
    padding: 12px 8px 8px 8px;
    font-weight: 600;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 9px;
    padding: 0 5px;
    color: {PALETTE['text_muted']};
    background-color: {PALETTE['panel']};
}}

QListWidget, QTreeWidget, QTableWidget {{
    background-color: {PALETTE['panel']};
    border: 1px solid {PALETTE['line']};
    border-radius: 8px;
    outline: none;
}}

QListWidget::item, QTreeWidget::item {{
    padding: 6px 8px;
}}

QListWidget::item:selected, QTreeWidget::item:selected, QTableWidget::item:selected {{
    background-color: {PALETTE['selection']};
    color: {PALETTE['selection_text']};
}}

QListWidget::item:hover, QTreeWidget::item:hover {{
    background-color: {PALETTE['panel_alt']};
}}

QScrollArea {{
    border: none;
    background-color: transparent;
}}

QScrollBar:vertical {{
    background-color: transparent;
    width: 10px;
    border: none;
}}

QScrollBar::handle:vertical {{
    background-color: {PALETTE['line_strong']};
    border-radius: 5px;
    min-height: 24px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {PALETTE['text_subtle']};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background-color: transparent;
    height: 10px;
    border: none;
}}

QScrollBar::handle:horizontal {{
    background-color: {PALETTE['line_strong']};
    border-radius: 5px;
    min-width: 24px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: {PALETTE['text_subtle']};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

QStatusBar {{
    background-color: {PALETTE['bg_deep']};
    color: {PALETTE['text_muted']};
    border-top: 1px solid {PALETTE['line']};
}}

QMenuBar {{
    background-color: {PALETTE['panel']};
    color: {PALETTE['text']};
}}

QMenuBar::item:selected {{
    background-color: {PALETTE['panel_alt']};
}}

QMenu {{
    background-color: {PALETTE['panel']};
    color: {PALETTE['text']};
    border: 1px solid {PALETTE['line']};
}}

QMenu::item {{
    padding: 5px 20px;
}}

QMenu::item:selected {{
    background-color: {PALETTE['primary_soft']};
}}

QProgressBar {{
    background-color: {PALETTE['bg']};
    border: 1px solid {PALETTE['line']};
    border-radius: 6px;
    text-align: center;
    color: {PALETTE['text']};
}}

QProgressBar::chunk {{
    background-color: {PALETTE['primary']};
    border-radius: 5px;
}}

QToolTip {{
    background-color: {PALETTE['bg_deep']};
    color: {PALETTE['text']};
    border: 1px solid {PALETTE['line']};
    border-radius: 6px;
    padding: 4px;
}}

QSplitter::handle {{
    background-color: transparent;
}}

QSplitter::handle:hover {{
    background-color: {PALETTE['primary']};
}}

QHeaderView::section {{
    background-color: {PALETTE['panel_alt']};
    color: {PALETTE['text']};
    border: 1px solid {PALETTE['line']};
    padding: 6px;
}}

QCheckBox, QRadioButton {{
    color: {PALETTE['text']};
    spacing: 6px;
}}

QCheckBox::indicator, QRadioButton::indicator {{
    width: 15px;
    height: 15px;
    border-radius: 4px;
    border: 1px solid {PALETTE['line_strong']};
    background-color: {PALETTE['bg']};
}}

QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: {PALETTE['primary']};
    border-color: {PALETTE['primary']};
}}

QSlider::groove:horizontal {{
    background-color: {PALETTE['line']};
    height: 6px;
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background-color: {PALETTE['primary']};
    width: 14px;
    height: 14px;
    margin: -4px 0;
    border-radius: 7px;
}}
"""


def apply_theme(app: QApplication) -> None:
    """应用全局中文字体和 Scheme-A 实色工作台主题。"""

    selected_font = apply_application_font(app)
    palette = app.palette()
    palette.setColor(QPalette.Window, QColor(PALETTE["bg"]))
    palette.setColor(QPalette.WindowText, QColor(PALETTE["text"]))
    palette.setColor(QPalette.Base, QColor(PALETTE["bg"]))
    palette.setColor(QPalette.AlternateBase, QColor(PALETTE["panel_alt"]))
    palette.setColor(QPalette.Text, QColor(PALETTE["text"]))
    palette.setColor(QPalette.Button, QColor(PALETTE["panel"]))
    palette.setColor(QPalette.ButtonText, QColor(PALETTE["text"]))
    palette.setColor(QPalette.ToolTipBase, QColor(PALETTE["bg_deep"]))
    palette.setColor(QPalette.ToolTipText, QColor(PALETTE["text"]))
    palette.setColor(QPalette.PlaceholderText, QColor(PALETTE["text_subtle"]))
    palette.setColor(QPalette.Highlight, QColor(PALETTE["selection"]))
    palette.setColor(QPalette.HighlightedText, QColor(PALETTE["selection_text"]))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(PALETTE["text_subtle"]))
    app.setPalette(palette)
    app.setStyleSheet(STYLESHEET)

    print(f"当前界面字体：{selected_font}")
