from __future__ import annotations

import os
import unittest
from pathlib import Path


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from gesture_studio_qt import GestureStudioQt, valid_label
except ImportError:
    QApplication = None
    GestureStudioQt = None
    valid_label = None


@unittest.skipIf(QApplication is None, "PySide6 no esta instalado")
class GestureStudioQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_builds_all_workflow_pages_without_camera(self) -> None:
        window = GestureStudioQt(start_camera=False)
        try:
            self.assertEqual(window.pages.count(), 4)
            self.assertEqual(len(window.nav_buttons), 4)
            self.assertEqual(set(window.labels), set(window.gesture_map))
        finally:
            window.close()

    def test_label_validation_matches_image_mapping_rules(self) -> None:
        self.assertTrue(valid_label("mano_arriba-2"))
        self.assertFalse(valid_label("mano arriba"))
        self.assertFalse(valid_label("../mano"))

    def test_navigation_icons_transition_and_toast_feedback(self) -> None:
        window = GestureStudioQt(start_camera=False)
        window.show()
        try:
            self.assertFalse(window.nav_buttons[0].icon().isNull())
            window._switch_page(1)
            QTest.qWait(320)
            self.assertEqual(window.pages.currentIndex(), 1)
            self.assertIsNone(window.pages.currentWidget().graphicsEffect())

            window._notify("Muestra guardada", "ok")
            QApplication.processEvents()
            self.assertIsNotNone(window.toast)
            self.assertTrue(window.toast.isVisible())
            self.assertEqual(window.toast.objectName(), "toast_ok")
        finally:
            window.close()

    def test_sidebar_collapses_to_icon_rail_and_expands(self) -> None:
        window = GestureStudioQt(start_camera=False)
        window.show()
        try:
            window.set_sidebar_collapsed(True)
            QTest.qWait(260)
            self.assertEqual(window.sidebar.width(), window.sidebar_collapsed_width)
            self.assertFalse(window.gesture_section.isVisible())
            self.assertTrue(all(not button.text() for button in window.nav_buttons))

            window.set_sidebar_collapsed(False)
            QTest.qWait(260)
            self.assertEqual(window.sidebar.width(), window.sidebar_expanded_width)
            self.assertTrue(window.gesture_section.isVisible())
            self.assertEqual(window.nav_buttons[0].text(), "Captura")
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
