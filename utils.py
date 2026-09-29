# -*- coding: utf-8 -*-
"""
utils.py
--------
Shared colour/font constants and reusable PySide6 helper widgets.
Import this in film_tab.py, fibre_tab.py, and main.py.
"""

from PyQt5.QtWidgets import QLabel, QLineEdit, QTextEdit, QFrame
from PyQt5.QtGui import QFont

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette  (black & white / grayscale)
#
# Names describe the ROLE the colour plays in the UI, not the literal hue —
# so swapping the whole palette later only ever means editing the values
# below, never touching call sites elsewhere in the app.
# ─────────────────────────────────────────────────────────────────────────────
BG_MAIN     = "#ffffff"   # page / panel background
BG_SURFACE  = "#f0f0f0"   # secondary surface (tab pane, progress track)
PRIMARY     = "#000000"   # buttons, active tab, key highlights
TEXT_MUTED  = "#3a3a3a"   # de-emphasised text / secondary action colour
WARNING     = "#6e6e6e"
ERROR       = "#1a1a1a"

FONT_HEAD = ("Courier New", 16, "bold")
FONT_BODY = ("Courier New", 12)
FONT_MONO = ("Courier New", 10)


def qfont(family="Courier New", size=12, bold=False) -> QFont:
    f = QFont(family, size)
    f.setBold(bold)
    return f


# ─────────────────────────────────────────────────────────────────────────────
# App-wide stylesheet
# ─────────────────────────────────────────────────────────────────────────────
APP_STYLE = f"""
QMainWindow, QDialog {{
    background: {BG_MAIN};
}}
QWidget {{
    background: {BG_MAIN};
    color: {PRIMARY};
    font-family: "Courier New";
    font-size: 12pt;
}}
QTabWidget::pane {{
    border: 1px solid {PRIMARY};
    background: {BG_SURFACE};
}}
QTabBar::tab {{
    background: {PRIMARY};
    color: {BG_MAIN};
    padding: 8px 22px;
    font-size: 11pt;
    font-weight: bold;
    border: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}}
QTabBar::tab:selected {{
    background: {BG_MAIN};
    color: {PRIMARY};
    border-bottom: 2px solid {PRIMARY};
}}

QFrame#settingsPanel {{
    background: {BG_MAIN};
    border: 1px solid {PRIMARY};
    border-radius: 4px;
}}

QPushButton {{
    background: {PRIMARY};
    color: {BG_MAIN};
    border: none;
    border-radius: 8px;
    padding: 6px 16px;
    font-family: "Courier New";
    font-size: 10pt;
    font-weight: bold;
}}
QPushButton:hover    {{ background: #333333; }}
QPushButton:pressed  {{ background: #555555; color: {BG_MAIN}; }}
QPushButton:disabled {{ background: #c8c8c8; color: #6e6e6e; }}

QPushButton#accent {{
    background: {PRIMARY};
    color: {BG_MAIN};
    border-radius: 8px;
    font-weight: bold;
}}
QPushButton#accent:hover {{ background: #333333; }}

QLineEdit {{
    background: {BG_MAIN};
    color: {PRIMARY};
    border: 1px solid {PRIMARY};
    border-radius: 6px;
    padding: 4px 10px;
    font-family: "Courier New";
    font-size: 10pt;
}}
QLineEdit:focus {{
    border: 2px solid {PRIMARY};}}

QListWidget {{
    background: {BG_MAIN};
    color: {PRIMARY};
    border: 1px solid {PRIMARY};
    border-radius: 6px;
    font-family: "Courier New";
    font-size: 9pt;
}}
QListWidget::item:selected {{
    background: {PRIMARY};
    color: {BG_MAIN};
}}

QTextEdit {{
    background: {BG_MAIN};
    color: {PRIMARY};
    border: 1px solid {PRIMARY};
    border-radius: 6px;
    font-family: "Courier New";
    font-size: 9pt;
}}

QScrollBar:vertical {{
    background: {BG_MAIN};
    width: 8px;
}}
QScrollBar::handle:vertical {{
    background: {PRIMARY};
    border-radius: 4px;
    min-height: 20px;
}}

QProgressBar {{
    background: {BG_SURFACE};
    color: {PRIMARY};
    border: 1px solid {PRIMARY};
    border-radius: 4px;
    text-align: center;
    height: 8px;
}}
QProgressBar::chunk {{ background: {PRIMARY}; border-radius: 4px; }}

QCheckBox {{
    color: {PRIMARY};
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    background: {BG_MAIN};
    border: 1px solid {PRIMARY};
    border-radius: 3px;
}}
QCheckBox::indicator:checked {{
    background: {PRIMARY};
    border: 1px solid {PRIMARY};
}}

QRadioButton {{
    color: {PRIMARY};
    spacing: 6px;
}}
QRadioButton::indicator {{
    width: 14px; height: 14px;
}}

QLabel#section {{
    color: {PRIMARY};
    font-size: 14pt;
    padding-top: 8px;
}}
QLabel#heading {{
    color: {PRIMARY};
    font-size: 14pt;
    font-weight: bold;
}}
QFrame#separator {{
    background: {PRIMARY};
    max-height: 1px;
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helper widget factories
# ─────────────────────────────────────────────────────────────────────────────

def heading_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("heading")
    lbl.setFont(qfont(size=16, bold=True))
    return lbl


def section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("section")
    lbl.setFont(qfont(size=12))
    return lbl


def make_entry(default: str = "", width: int = 100) -> QLineEdit:
    e = QLineEdit(default)
    e.setFixedWidth(width)
    return e


def make_log(height: int = 160) -> QTextEdit:
    log = QTextEdit()
    log.setReadOnly(True)
    log.setFixedHeight(height)
    log.setFont(qfont(size=9))
    return log


def log_write(log_widget: QTextEdit, msg: str):
    log_widget.append(msg)
    sb = log_widget.verticalScrollBar()
    sb.setValue(sb.maximum())


def hseparator() -> QFrame:
    f = QFrame()
    f.setObjectName("separator")
    f.setFrameShape(QFrame.HLine)
    return f