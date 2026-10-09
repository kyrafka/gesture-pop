from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from app_config import AppConfig, load_config, save_config


class AppConfigTests(unittest.TestCase):
    def test_rtmpose_is_mandatory_and_old_toggle_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "app_config.json"
            path.write_text(
                '{"heavy_hand_assist": false, "heavy_hand_idle_interval_seconds": 1.5}',
                encoding="utf-8",
            )

            config = load_config(path)

            self.assertFalse(hasattr(config, "heavy_hand_assist"))
            self.assertEqual(config.heavy_hand_interval_seconds, 0.42)
            self.assertEqual(config.heavy_hand_stale_seconds, 0.75)

    def test_recognition_settings_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "app_config.json"
            config = replace(
                AppConfig(),
                confidence_threshold=0.77,
                confidence_margin=0.19,
                stability_frames=8,
            )

            save_config(config, path)
            loaded = load_config(path)

            self.assertEqual(loaded.confidence_threshold, 0.77)
            self.assertEqual(loaded.confidence_margin, 0.19)
            self.assertEqual(loaded.stability_frames, 8)


if __name__ == "__main__":
    unittest.main()
