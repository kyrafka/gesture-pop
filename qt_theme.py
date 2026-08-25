from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication


COLORS = {
    "canvas": "#101216",
    "sidebar": "#14171b",
    "surface": "#181c21",
    "surface_2": "#20252c",
    "border": "#2b3139",
    "text": "#edf0f3",
    "muted": "#929ba6",
    "icon": "#aeb6c0",
    "disabled": "#626a74",
    "teal": "#39b980",
    "teal_dark": "#183529",
    "success": "#4bc38a",
    "amber": "#dca94d",
    "coral": "#e66f69",
    "blue": "#5b8def",
    "blue_dark": "#1a2740",
}


def configure_application_font(app: QApplication) -> str:
    preferred = "Segoe UI"
    if preferred not in QFontDatabase.families() and os.name == "nt":
        font_path = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "segoeui.ttf"
        if font_path.is_file():
            font_id = QFontDatabase.addApplicationFont(str(font_path))
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                preferred = families[0]
    app.setFont(QFont(preferred, 10))
    return preferred


def application_stylesheet() -> str:
    return f"""
    * {{
        font-family: "Segoe UI";
        font-size: 10pt;
        color: {COLORS['text']};
    }}
    QMainWindow, QWidget#appRoot {{ background: {COLORS['canvas']}; }}
    QWidget#sidebar {{
        background: {COLORS['sidebar']};
        border-right: 1px solid {COLORS['border']};
    }}
    QLabel#brand {{ font-size: 16pt; font-weight: 650; }}
    QLabel#brandMark {{ color: {COLORS['blue']}; font-size: 18pt; font-weight: 750; }}
    QLabel#pageTitle {{ font-size: 16pt; font-weight: 650; }}
    QLabel#sectionTitle {{ font-size: 11pt; font-weight: 650; }}
    QLabel#muted, QLabel.muted {{ color: {COLORS['muted']}; }}
    QLabel#eyebrow {{ color: {COLORS['muted']}; font-size: 8pt; font-weight: 650; }}
    QLabel#metric {{ font-size: 22pt; font-weight: 700; }}
    QLabel#success {{ color: {COLORS['success']}; font-weight: 600; }}
    QLabel#warning {{ color: {COLORS['amber']}; font-weight: 600; }}
    QLabel#error {{ color: {COLORS['coral']}; font-weight: 600; }}
    QFrame#panel, QFrame#metricPanel, QFrame#sampleCard {{
        background: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
    }}
    QFrame#metricPanel {{ background: {COLORS['surface_2']}; }}
    QPushButton {{
        min-height: 32px;
        padding: 3px 11px;
        background: {COLORS['surface_2']};
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
        font-weight: 550;
    }}
    QPushButton:hover {{ border-color: #46505c; background: #252b33; }}
    QPushButton:pressed {{ background: #12151a; border-color: {COLORS['blue']}; }}
    QPushButton:disabled {{ color: #66717d; background: #171c22; }}
    QPushButton#primary {{
        color: #ffffff;
        background: {COLORS['blue']};
        border-color: {COLORS['blue']};
    }}
    QPushButton#primary:hover {{ background: #6f9df3; }}
    QPushButton#primary:pressed {{ background: #4779d7; }}
    QPushButton#primary:disabled {{
        color: #737b85;
        background: #1b2026;
        border-color: #2c333b;
    }}
    QPushButton#warningButton {{
        color: #181106;
        background: {COLORS['amber']};
        border-color: {COLORS['amber']};
    }}
    QPushButton#warningButton:disabled {{
        color: #7c6b4c;
        background: #2c2922;
        border-color: #484132;
    }}
    QPushButton#danger {{ color: {COLORS['coral']}; }}
    QPushButton#navButton {{
        min-height: 38px;
        padding: 3px 10px;
        text-align: left;
        border-color: transparent;
        background: transparent;
        color: {COLORS['muted']};
    }}
    QPushButton#navButton:hover {{ background: #1b1f25; color: {COLORS['text']}; }}
    QPushButton#navButton:checked {{
        color: {COLORS['text']};
        background: {COLORS['blue_dark']};
        border-color: #314a76;
    }}
    QPushButton#sidebarToggle {{
        min-width: 30px;
        max-width: 30px;
        min-height: 30px;
        max-height: 30px;
        padding: 0;
        background: transparent;
        border-color: transparent;
    }}
    QPushButton#sidebarToggle:hover {{ background: {COLORS['surface_2']}; }}
    QListWidget {{
        background: transparent;
        border: 0;
        outline: 0;
    }}
    QListWidget::item {{
        min-height: 54px;
        padding: 5px;
        border-radius: 3px;
        color: {COLORS['muted']};
    }}
    QListWidget::item:hover {{ background: {COLORS['surface']}; }}
    QListWidget::item:selected {{
        color: {COLORS['text']};
        background: {COLORS['surface_2']};
        border-left: 2px solid {COLORS['blue']};
    }}
    QProgressBar {{
        height: 8px;
        border: 0;
        border-radius: 4px;
        background: #2a333d;
        text-align: center;
        color: transparent;
    }}
    QProgressBar::chunk {{ background: {COLORS['blue']}; border-radius: 4px; }}
    QScrollArea {{ border: 0; background: transparent; }}
    QScrollArea > QWidget > QWidget {{ background: transparent; }}
    QToolTip {{
        color: {COLORS['text']};
        background: {COLORS['surface_2']};
        border: 1px solid {COLORS['border']};
        padding: 5px;
    }}
    QStatusBar {{
        color: {COLORS['muted']};
        background: {COLORS['sidebar']};
        border-top: 1px solid {COLORS['border']};
    }}
    QFrame#toast_ok, QFrame#toast_warning, QFrame#toast_busy, QFrame#toast_error {{
        background: #242a31;
        border: 1px solid #3b444f;
        border-radius: 4px;
    }}
    QFrame#toast_ok {{ border-left: 3px solid {COLORS['success']}; }}
    QFrame#toast_warning {{ border-left: 3px solid {COLORS['amber']}; }}
    QFrame#toast_busy {{ border-left: 3px solid {COLORS['blue']}; }}
    QFrame#toast_error {{ border-left: 3px solid {COLORS['coral']}; }}
    QDialog {{ background: {COLORS['canvas']}; }}
    QLineEdit, QSpinBox, QComboBox {{
        min-height: 32px;
        padding: 2px 8px;
        background: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
        selection-background-color: {COLORS['blue_dark']};
    }}
    QMenu {{ background: {COLORS['surface']}; border: 1px solid {COLORS['border']}; }}
    QMenu::item {{ padding: 7px 24px 7px 12px; }}
    QMenu::item:selected {{ background: {COLORS['blue_dark']}; }}
    QSplitter::handle {{ background: {COLORS['border']}; width: 1px; }}
    """
