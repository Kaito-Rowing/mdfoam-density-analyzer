from __future__ import annotations


COLORS = {
    "background": "#121212",
    "sidebar": "#0d0f0e",
    "surface": "#181818",
    "surface_alt": "#202221",
    "input": "#252525",
    "border": "#3a3d3b",
    "text": "#ffffff",
    "muted": "#b3b3b3",
    "md_series": "#f2f5f3",
    "accent": "#31d17c",
    "accent_hover": "#43e38e",
    "accent_pressed": "#25b96a",
    "danger": "#f3727f",
    "warning": "#ffa42b",
    "info": "#539df5",
    "grid": "#3d423f",
}


APP_STYLESHEET = """
QWidget {
    background: #121212;
    color: #ffffff;
    font-size: 13px;
}
QMainWindow, QDialog {
    background: #121212;
}
QLabel, QCheckBox, QRadioButton {
    background: transparent;
}
QFrame#sidebar {
    background: #0d0f0e;
    border-right: 1px solid #292b2a;
}
QLabel#brandTitle {
    color: #ffffff;
    font-size: 19px;
    font-weight: 700;
}
QLabel#brandSubtitle, QLabel[muted="true"] {
    color: #8f9692;
    font-size: 11px;
    font-weight: 600;
}
QLabel#pageTitle {
    color: #ffffff;
    font-size: 22px;
    font-weight: 700;
}
QLabel#pageSubtitle {
    color: #9ba19e;
    font-size: 12px;
}
QPushButton {
    min-height: 30px;
    padding: 2px 15px;
    border: 1px solid #4d514f;
    border-radius: 16px;
    background: #252525;
    color: #ffffff;
    font-weight: 600;
}
QPushButton:hover {
    background: #303230;
    border-color: #747a76;
}
QPushButton:pressed {
    background: #1d1f1e;
}
QPushButton:disabled {
    color: #666b68;
    background: #1a1b1a;
    border-color: #292b2a;
}
QPushButton[variant="primary"] {
    color: #07140d;
    background: #31d17c;
    border-color: #31d17c;
}
QPushButton[variant="primary"]:hover {
    background: #43e38e;
    border-color: #43e38e;
}
QPushButton[variant="primary"]:pressed {
    background: #25b96a;
    border-color: #25b96a;
}
QPushButton[variant="danger"] {
    color: #f3727f;
    background: transparent;
    border-color: #754047;
}
QPushButton[variant="danger"]:hover {
    color: #ffffff;
    background: #7a303a;
}
QPushButton[nav="true"] {
    min-height: 42px;
    padding: 0 16px;
    border: 0;
    border-radius: 8px;
    background: transparent;
    color: #a7ada9;
    text-align: left;
    font-size: 14px;
}
QPushButton[nav="true"]:hover {
    color: #ffffff;
    background: #1b1e1c;
}
QPushButton[nav="true"]:checked {
    color: #ffffff;
    background: #25382e;
    border-left: 3px solid #31d17c;
    font-weight: 700;
}
QGroupBox {
    margin-top: 13px;
    padding: 16px 14px 14px 14px;
    border: 1px solid #303331;
    border-radius: 10px;
    background: #181818;
    font-weight: 700;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 6px;
    color: #dce1de;
    background: #181818;
}
QFrame[card="true"] {
    background: #181818;
    border: 1px solid #303331;
    border-radius: 10px;
}
QFrame[kpi="true"] {
    background: #181818;
    border: 1px solid #303331;
    border-radius: 10px;
}
QLabel[kpiLabel="true"] {
    color: #8f9692;
    font-size: 11px;
    font-weight: 600;
}
QLabel[kpiValue="true"] {
    color: #ffffff;
    font-size: 18px;
    font-weight: 700;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QListWidget, QTableWidget {
    color: #ffffff;
    background: #252525;
    border: 1px solid #444846;
    border-radius: 6px;
    selection-background-color: #2d7250;
    selection-color: #ffffff;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    min-height: 30px;
    padding: 0 8px;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QTextEdit:focus, QListWidget:focus, QTableWidget:focus {
    border: 1px solid #31d17c;
}
QComboBox::drop-down {
    border: 0;
    width: 24px;
}
QComboBox QAbstractItemView {
    background: #252525;
    color: #ffffff;
    border: 1px solid #4d514f;
    selection-background-color: #2d7250;
}
QListWidget {
    padding: 5px;
}
QListWidget::item {
    min-height: 28px;
    padding: 3px 8px;
    border-radius: 5px;
}
QListWidget::item:hover {
    background: #303230;
}
QListWidget::item:selected {
    background: #2d7250;
}
QTableWidget {
    gridline-color: #343735;
    alternate-background-color: #202221;
}
QTableWidget::item {
    padding: 5px 8px;
}
QTableWidget::item:selected {
    background: #2d7250;
    color: #ffffff;
}
QHeaderView::section {
    color: #c8ceca;
    background: #202221;
    border: 0;
    border-right: 1px solid #343735;
    border-bottom: 1px solid #343735;
    padding: 7px 8px;
    font-weight: 700;
}
QTabWidget::pane {
    border: 1px solid #303331;
    border-radius: 8px;
    background: #181818;
    top: -1px;
}
QTabBar::tab {
    color: #a7ada9;
    background: #181818;
    border: 0;
    border-bottom: 2px solid transparent;
    padding: 9px 14px;
    font-weight: 600;
}
QTabBar::tab:hover {
    color: #ffffff;
    background: #202221;
}
QTabBar::tab:selected {
    color: #ffffff;
    border-bottom-color: #31d17c;
}
QCheckBox {
    spacing: 7px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #626864;
    border-radius: 4px;
    background: #252525;
}
QCheckBox::indicator:checked {
    background: #31d17c;
    border-color: #31d17c;
}
QProgressBar {
    min-height: 8px;
    max-height: 8px;
    border: 0;
    border-radius: 4px;
    background: #303331;
    color: transparent;
}
QProgressBar::chunk {
    border-radius: 4px;
    background: #31d17c;
}
QSlider::groove:horizontal {
    height: 4px;
    border-radius: 2px;
    background: #3a3d3b;
}
QSlider::handle:horizontal {
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
    background: #31d17c;
}
QSplitter::handle {
    background: #292b2a;
}
QScrollArea {
    border: 0;
    background: transparent;
}
QScrollBar:vertical {
    width: 11px;
    background: #181818;
}
QScrollBar:horizontal {
    height: 11px;
    background: #181818;
}
QScrollBar::handle {
    min-height: 24px;
    min-width: 24px;
    border-radius: 5px;
    background: #4a4e4b;
}
QScrollBar::handle:hover {
    background: #666c68;
}
QScrollBar::add-line, QScrollBar::sub-line {
    width: 0;
    height: 0;
}
QToolTip {
    color: #ffffff;
    background: #252525;
    border: 1px solid #555a57;
    padding: 5px;
}
"""
