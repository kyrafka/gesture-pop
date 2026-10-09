from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication


COLORS = {
    "canvas": "#121516",
    "sidebar": "#171b1c",
    "surface": "#1b2021",
    "surface_2": "#242b2c",
    "border": "#30393a",
    "text": "#edf2f1",
    "muted": "#9da9a6",
    "icon": "#b8c2bf",
    "disabled": "#697471",
    "teal": "#38b982",
    "teal_dark": "#193a30",
    "success": "#4bc38a",
    "amber": "#dca94d",
    "coral": "#e66f69",
    "blue": "#6b9ce8",
    "blue_dark": "#243246",
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
    QLabel#qualityScore {{ font-size: 22pt; font-weight: 700; }}
    QLabel#qualityScore[qualityState="good"] {{ color: {COLORS['success']}; }}
    QLabel#qualityScore[qualityState="medium"] {{ color: {COLORS['amber']}; }}
    QLabel#qualityScore[qualityState="low"] {{ color: {COLORS['coral']}; }}
    QLabel#success {{ color: {COLORS['success']}; font-weight: 600; }}
    QLabel#warning {{ color: {COLORS['amber']}; font-weight: 600; }}
    QLabel#error {{ color: {COLORS['coral']}; font-weight: 600; }}
    QFrame#panel, QFrame#metricPanel, QFrame#sampleCard {{
        background: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
    }}
    QFrame#metricPanel {{ background: {COLORS['surface_2']}; }}
    QPushButton {{
        min-height: 34px;
        padding: 4px 12px;
        background: {COLORS['surface_2']};
        border: 1px solid {COLORS['border']};
        border-radius: 5px;
        font-weight: 550;
    }}
    QPushButton:hover {{ border-color: #4a5857; background: #2a3232; }}
    QPushButton:pressed {{ background: #151a19; border-color: {COLORS['teal']}; }}
    QPushButton:focus {{ border: 1px solid {COLORS['teal']}; }}
    QPushButton:disabled {{ color: #71807b; background: #1c2222; }}
    QPushButton#primary {{
        color: #ffffff;
        background: {COLORS['teal']};
        border-color: {COLORS['teal']};
    }}
    QPushButton#primary:hover {{ background: #4ac990; }}
    QPushButton#primary:pressed {{ background: #2f9e6d; }}
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
        background: {COLORS['teal_dark']};
        border-color: #2a5a47;
        border-left: 2px solid {COLORS['teal']};
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
    QProgressBar#qualityBar[qualityState="good"]::chunk {{ background: {COLORS['success']}; }}
    QProgressBar#qualityBar[qualityState="medium"]::chunk {{ background: {COLORS['amber']}; }}
    QProgressBar#qualityBar[qualityState="low"]::chunk {{ background: {COLORS['coral']}; }}
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
    QSlider::groove:horizontal {{ height: 4px; background: #37413f; border-radius: 2px; }}
    QSlider::sub-page:horizontal {{ background: {COLORS['teal']}; border-radius: 2px; }}
    QSlider::handle:horizontal {{
        width: 14px;
        margin: -5px 0;
        background: {COLORS['text']};
        border: 2px solid {COLORS['teal']};
        border-radius: 7px;
    }}
    QSlider::handle:horizontal:hover {{ background: {COLORS['teal']}; }}
    """
