from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QObject, QPropertyAnimation, QTimer, Qt
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QWidget,
)
import qtawesome as qta

from qt_theme import COLORS


def app_icon(name: str, color: str | None = None):
    return qta.icon(
        name,
        color=color or COLORS["icon"],
        color_disabled=COLORS["disabled"],
    )


def apply_button_icon(button: QPushButton, name: str, color: str | None = None) -> QPushButton:
    button.setIcon(app_icon(name, color))
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    return button


class FadingStackedWidget(QStackedWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._animation: QPropertyAnimation | None = None
        self._effect: QGraphicsOpacityEffect | None = None

    def setCurrentIndex(self, index: int) -> None:
        if index == self.currentIndex():
            return
        super().setCurrentIndex(index)
        page = self.currentWidget()
        if page is None or not self.isVisible():
            return

        effect = QGraphicsOpacityEffect(page)
        page.setGraphicsEffect(effect)
        animation = QPropertyAnimation(effect, b"opacity", self)
        animation.setDuration(150)
        animation.setStartValue(0.2)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.finished.connect(lambda target=page: target.setGraphicsEffect(None))
        self._effect = effect
        self._animation = animation
        animation.start()


class Toast(QFrame):
    def __init__(self, message: str, level: str, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName(f"toast_{level}")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMinimumWidth(300)
        self.setMaximumWidth(440)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(9)
        icon_names = {
            "ok": "fa6s.circle-check",
            "warning": "fa6s.triangle-exclamation",
            "busy": "fa6s.clock",
            "error": "fa6s.circle-xmark",
        }
        icon_colors = {
            "ok": COLORS["success"],
            "warning": COLORS["amber"],
            "busy": COLORS["blue"],
            "error": COLORS["coral"],
        }
        icon = QLabel()
        icon.setPixmap(app_icon(icon_names.get(level, "fa6s.circle-info"), icon_colors.get(level)).pixmap(17, 17))
        text = QLabel(message)
        text.setWordWrap(True)
        layout.addWidget(icon)
        layout.addWidget(text, 1)

        self._effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._effect)
        self._animation: QPropertyAnimation | None = None
        self._dismiss_timer = QTimer(self)
        self._dismiss_timer.setSingleShot(True)
        self._dismiss_timer.timeout.connect(self.dismiss)

    def reveal(self) -> None:
        self.adjustSize()
        self.show()
        self.raise_()
        animation = QPropertyAnimation(self._effect, b"opacity", self)
        animation.setDuration(140)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._animation = animation
        animation.start()
        self._dismiss_timer.start(3200)

    def dismiss(self) -> None:
        animation = QPropertyAnimation(self._effect, b"opacity", self)
        animation.setDuration(180)
        animation.setStartValue(self._effect.opacity())
        animation.setEndValue(0.0)
        animation.setEasingCurve(QEasingCurve.Type.InCubic)
        animation.finished.connect(self.deleteLater)
        self._animation = animation
        animation.start()


class FadeController(QObject):
    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._animations: dict[int, QPropertyAnimation] = {}

    def pulse(self, widget: QWidget) -> None:
        effect = widget.graphicsEffect()
        if not isinstance(effect, QGraphicsOpacityEffect):
            effect = QGraphicsOpacityEffect(widget)
            widget.setGraphicsEffect(effect)
        animation = QPropertyAnimation(effect, b"opacity", self)
        animation.setDuration(180)
        animation.setStartValue(0.45)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        key = id(widget)
        animation.finished.connect(lambda: self._animations.pop(key, None))
        self._animations[key] = animation
        animation.start()
