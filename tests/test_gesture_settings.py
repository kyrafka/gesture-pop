from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gesture_settings


class GestureSettingsTests(unittest.TestCase):
    def test_loads_defaults_and_persists_expected_hands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings_file = Path(tmp) / "gesture_settings.json"
            with patch.object(gesture_settings, "GESTURE_SETTINGS_FILE", settings_file):
                settings = gesture_settings.load_gesture_settings(["saludo", "juntas"])
                self.assertEqual(gesture_settings.expected_hands_for(settings, "saludo"), 1)

                settings["juntas"]["expected_hands"] = 2
                gesture_settings.save_gesture_settings(settings)

                loaded = gesture_settings.load_gesture_settings(["saludo", "juntas"])
                self.assertEqual(gesture_settings.expected_hands_for(loaded, "juntas"), 2)
                self.assertEqual(gesture_settings.expected_hands_for(loaded, "saludo"), 1)


if __name__ == "__main__":
    unittest.main()
