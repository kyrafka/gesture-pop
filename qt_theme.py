from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication


COLORS = {
    "canvas": "#0f1216",
    "sidebar": "#151a20",
    "surface": "#1b222a",
    "surface_2": "#222b35",
    "border": "#303b47",
    "text": "#f3f6f8",
    "muted": "#9eabb8",
    "teal": "#3dd6b4",
    "teal_dark": "#173f39",
    "amber": "#f6bd55",
    "coral": "#ff7b72",
    "blue": "#75a7ff",
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
    QLabel#brand {{ font-size: 18pt; font-weight: 700; }}
    QLabel#brandMark {{ color: {COLORS['teal']}; font-size: 20pt; font-weight: 800; }}
    QLabel#pageTitle {{ font-size: 17pt; font-weight: 700; }}
    QLabel#sectionTitle {{ font-size: 12pt; font-weight: 700; }}
    QLabel#muted, QLabel.muted {{ color: {COLORS['muted']}; }}
    QLabel#eyebrow {{ color: {COLORS['teal']}; font-size: 9pt; font-weight: 700; }}
    QLabel#metric {{ font-size: 22pt; font-weight: 700; }}
    QLabel#success {{ color: {COLORS['teal']}; font-weight: 600; }}
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
        font-weight: 600;
    }}
    QPushButton:hover {{ border-color: {COLORS['muted']}; background: #293440; }}
    QPushButton:pressed {{ background: #11161c; }}
    QPushButton:disabled {{ color: #66717d; background: #171c22; }}
    QPushButton#primary {{
        color: #071511;
        background: {COLORS['teal']};
        border-color: {COLORS['teal']};
    }}
    QPushButton#primary:hover {{ background: #63e0c4; }}
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
        min-height: 40px;
        padding: 3px 10px;
        text-align: left;
        border-color: transparent;
        background: transparent;
        color: {COLORS['muted']};
    }}
    QPushButton#navButton:hover {{ background: {COLORS['surface']}; color: {COLORS['text']}; }}
    QPushButton#navButton:checked {{
        color: {COLORS['text']};
        background: {COLORS['teal_dark']};
        border-color: #296f62;
    }}
    QListWidget {{
        background: transparent;
        border: 0;
        outline: 0;
    }}
    QListWidget::item {{
        min-height: 54px;
        padding: 5px;
        border-radius: 5px;
        color: {COLORS['muted']};
    }}
    QListWidget::item:hover {{ background: {COLORS['surface']}; }}
    QListWidget::item:selected {{
        color: {COLORS['text']};
        background: {COLORS['surface_2']};
        border-left: 3px solid {COLORS['teal']};
    }}
    QProgressBar {{
        height: 8px;
        border: 0;
        border-radius: 4px;
        background: #2a333d;
        text-align: center;
        color: transparent;
    }}
    QProgressBar::chunk {{ background: {COLORS['teal']}; border-radius: 4px; }}
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
    QDialog {{ background: {COLORS['canvas']}; }}
    QLineEdit, QSpinBox, QComboBox {{
        min-height: 32px;
        padding: 2px 8px;
        background: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        border-radius: 5px;
        selection-background-color: {COLORS['teal_dark']};
    }}
    QMenu {{ background: {COLORS['surface']}; border: 1px solid {COLORS['border']}; }}
    QMenu::item {{ padding: 7px 24px 7px 12px; }}
    QMenu::item:selected {{ background: {COLORS['teal_dark']}; }}
    QSplitter::handle {{ background: {COLORS['border']}; width: 1px; }}
    """
