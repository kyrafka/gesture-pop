from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from gesture_features import FeatureResult, HandPose, HandTrackingInfo
    from gesture_studio_qt import GestureStudioQt, valid_label
except ImportError:
    QApplication = None
    FeatureResult = None
    HandPose = None
    HandTrackingInfo = None
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
            self.assertEqual(window.pages.count(), 5)
            self.assertEqual(len(window.nav_buttons), 5)
            self.assertEqual(set(window.labels), set(window.gesture_map))
            self.assertFalse(hasattr(window, "heavy_toggle"))
            self.assertEqual(window.heavy_badge.text(), "Detector de manos")
            self.assertFalse(hasattr(window, "heavy_info_button"))
            self.assertFalse(hasattr(window, "vector_summary_label"))
            self.assertFalse(hasattr(window, "face_label"))
            self.assertFalse(hasattr(window, "vector_status"))
            self.assertEqual(
                window.gesture_list.horizontalScrollBarPolicy(),
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
            )
            self.assertEqual(
                window.inspector_scroll.horizontalScrollBarPolicy(),
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
            )
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
            QTest.qWait(500)
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

    def test_recognition_settings_preview_and_save(self) -> None:
        window = GestureStudioQt(start_camera=False)
        try:
            self.assertEqual(window.confidence_value.text(), "68%")
            self.assertEqual(window.votes_value.text(), "7 / 10")
            self.assertFalse(hasattr(window, "margin_slider"))

            window.confidence_slider.setValue(77)
            window.votes_slider.setValue(8)
            self.assertTrue(window.save_recognition_button.isEnabled())

            with patch("gesture_studio_qt.save_config") as save:
                window.save_recognition_settings()

            saved_config = save.call_args.args[0]
            self.assertEqual(saved_config.confidence_threshold, 0.77)
            self.assertEqual(saved_config.confidence_margin, window.config.confidence_margin)
            self.assertEqual(saved_config.stability_frames, 8)
            self.assertFalse(window.save_recognition_button.isEnabled())
        finally:
            window.close()

    def test_capture_quality_rewards_sharp_clear_frames_and_surfaces_blur(self) -> None:
        window = GestureStudioQt(start_camera=False)
        try:
            pose = HandPose(
                index=1,
                center_x=0.5,
                center_y=0.5,
                width=0.3,
                height=0.45,
                angle_deg=0.0,
                tilt_deg=0.0,
                zone="medio-centro",
                bbox=(0.3, 0.25, 0.7, 0.75),
            )
            result = FeatureResult(
                vector=np.zeros(194, dtype=np.float32),
                debug="test",
                hands=[[SimpleNamespace() for _ in range(21)]],
                hand_poses=[pose],
                tracking=HandTrackingInfo(),
            )
            pattern = (np.indices((240, 320)).sum(axis=0) % 2 * 180 + 30).astype(np.uint8)
            sharp_frame = np.repeat(pattern[:, :, None], 3, axis=2)
            blurred_frame = np.full((240, 320, 3), 120, dtype=np.uint8)

            sharp_score, _ = window._capture_quality(result, True, 0.0, 1, sharp_frame)
            blurred_score, blurred_notes = window._capture_quality(result, True, 0.0, 1, blurred_frame)

            self.assertGreater(sharp_score, blurred_score)
            self.assertTrue(any("borrosa" in note.lower() for note in blurred_notes))
        finally:
            window.close()

    def test_capture_cannot_be_saved_when_expected_hand_is_missing(self) -> None:
        window = GestureStudioQt(start_camera=False)
        try:
            window.current_frame = np.full((240, 320, 3), 100, dtype=np.uint8)
            window._update_telemetry(None, False, float("inf"))

            self.assertFalse(window.capture_button.isEnabled())
            self.assertIn("Faltan manos", window.quality_status_label.text())
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
