"""Dark workspace theme and shared style helpers for PyQt5."""
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

PALETTE = {
    "bg": "#11141b",
    "bg_deep": "#0b0e14",
    "canvas": "#151922",
    "panel": "#1b202b",
    "panel_alt": "#232938",
    "panel_raised": "#2b3242",
    "line": "#303849",
    "line_strong": "#465268",
    "text": "#edf2ff",
    "text_muted": "#aab4c8",
    "text_subtle": "#748198",
    "ink": "#ffffff",
    "primary": "#7c5cff",
    "primary_hover": "#9b7cff",
    "primary_pressed": "#6746e8",
    "primary_soft": "#292247",
    "selection": "#1f7aff",
    "selection_text": "#ffffff",
    "success": "#44d19d",
    "success_soft": "#173729",
    "danger": "#ff7a7a",
    "danger_soft": "#3b2025",
    "warning": "#f6bd60",
    "warning_soft": "#3d2d16",
    "violet": "#9b6dff",
    "teal": "#4bd3c2",
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
        "body": f"color: {PALETTE['text']}; font-size: 14px;",
        "small": f"color: {PALETTE['text']}; font-size: 12px;",
        "muted": f"color: {PALETTE['text_muted']}; font-size: 13px;",
        "hint": f"color: {PALETTE['text_subtle']}; font-size: 12px;",
        "success": f"color: {PALETTE['success']}; font-size: 14px; font-weight: 650;",
        "warning": f"color: {PALETTE['warning']}; font-size: 14px; font-weight: 650;",
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
        " border-radius: 8px;"
        " padding: 4px 10px;"
        " font-size: 11px;"
        f" font-weight: {weight};"
        "}"
        f"QToolButton:hover {{ background-color: {PALETTE['primary_soft']};"
        f" border-color: {PALETTE['primary']}; }}"
    )


STYLESHEET = f"""
QWidget {{
    background-color: {PALETTE['bg']};
    color: {PALETTE['text']};
    font-size: 14px;
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
    padding: 11px 22px;
    border: none;
    margin-right: 2px;
    min-height: 26px;
}}

QTabBar::tab:selected {{
    background-color: {PALETTE['primary_soft']};
    color: {PALETTE['text']};
    border-bottom: 3px solid {PALETTE['primary']};
}}

QTabBar::tab:hover {{
    color: {PALETTE['text']};
    background-color: {PALETTE['panel']};
}}

QToolBar {{
    background-color: {PALETTE['panel']};
    border: 1px solid {PALETTE['line']};
    border-radius: 10px;
    spacing: 8px;
    padding: 10px 12px;
}}

QToolBar::separator {{
    background-color: {PALETTE['line']};
    width: 1px;
    margin: 5px 8px;
}}

QPushButton {{
    background-color: {PALETTE['panel']};
    color: {PALETTE['text']};
    border: 1px solid {PALETTE['line']};
    border-radius: 9px;
    padding: 8px 14px;
    font-weight: 600;
    min-height: 18px;
}}

QPushButton:hover {{
    background-color: {PALETTE['panel_alt']};
    border-color: {PALETTE['primary']};
}}

QPushButton:focus {{
    border-color: {PALETTE['primary']};
}}

QPushButton:pressed {{
    background-color: {PALETTE['panel_raised']};
    padding-top: 7px;
    padding-bottom: 5px;
}}

QPushButton:checked {{
    background-color: {PALETTE['primary_soft']};
    color: {PALETTE['primary']};
    border-color: {PALETTE['primary']};
}}

QPushButton:disabled {{
    color: {PALETTE['text_subtle']};
    background-color: {PALETTE['panel_raised']};
    border-color: {PALETTE['line']};
}}

QPushButton[role="primary"] {{
    background-color: {PALETTE['primary']};
    color: {PALETTE['ink']};
    border-color: {PALETTE['primary']};
}}

QPushButton[role="primary"]:hover {{
    background-color: {PALETTE['primary_hover']};
    border-color: {PALETTE['primary_hover']};
}}

QPushButton[role="primary"]:pressed {{
    background-color: {PALETTE['primary_pressed']};
    border-color: {PALETTE['primary_pressed']};
}}

QPushButton[role="danger"] {{
    background-color: {PALETTE['danger_soft']};
    color: {PALETTE['danger']};
    border-color: {PALETTE['danger']};
}}

QPushButton[role="danger"]:hover {{
    background-color: {PALETTE['danger']};
    color: {PALETTE['ink']};
}}

QPushButton[role="success"] {{
    background-color: {PALETTE['success_soft']};
    color: {PALETTE['success']};
    border-color: {PALETTE['success']};
}}

QPushButton[role="secondary"] {{
    background-color: {PALETTE['panel']};
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
    border-radius: 7px;
    padding: 5px 8px;
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
    background-color: {PALETTE['panel_alt']};
    color: {PALETTE['text']};
    border: 1px solid {PALETTE['line']};
    border-radius: 7px;
    padding: 5px 8px;
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
    background-color: {PALETTE['panel_alt']};
    color: {PALETTE['text']};
    border: 1px solid {PALETTE['line']};
    border-radius: 8px;
    padding: 8px;
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
    margin-top: 14px;
    padding: 15px 10px 10px 10px;
    font-weight: 650;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 6px;
    color: {PALETTE['text_muted']};
    background-color: {PALETTE['panel']};
}}

QListWidget, QTreeWidget, QTableWidget {{
    background-color: {PALETTE['panel']};
    border: 1px solid {PALETTE['line']};
    border-radius: 10px;
    outline: none;
}}

QListWidget::item, QTreeWidget::item {{
    padding: 8px 10px;
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
    background-color: {PALETTE['panel']};
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
    padding: 5px 22px;
}}

QMenu::item:selected {{
    background-color: {PALETTE['primary_soft']};
}}

QProgressBar {{
    background-color: {PALETTE['panel_alt']};
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
    background-color: {PALETTE['text']};
    color: {PALETTE['panel']};
    border: 1px solid {PALETTE['line_strong']};
    padding: 5px;
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
    border-radius: 5px;
    border: 1px solid {PALETTE['line_strong']};
    background-color: {PALETTE['panel']};
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
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}}
"""


def apply_theme(app: QApplication) -> None:
    """应用全局中文字体和暗色主题。"""

    selected_font = apply_application_font(app)
    palette = app.palette()
    palette.setColor(QPalette.Highlight, QColor(PALETTE["selection"]))
    palette.setColor(QPalette.HighlightedText, QColor(PALETTE["selection_text"]))
    app.setPalette(palette)
    app.setStyleSheet(STYLESHEET)

    print(f"当前界面字体：{selected_font}")
