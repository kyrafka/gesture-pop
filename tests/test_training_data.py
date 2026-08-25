from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import train_gestures


class TrainingUndoTests(unittest.TestCase):
    def test_reference_marker_does_not_delete_camera_photo(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = root / "samples.csv"
            manifest = root / "manifest.csv"
            captures = root / "captures"
            captures.mkdir()
            camera_photo = captures / "camera.jpg"
            camera_photo.write_bytes(b"photo")

            with samples.open("w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerows([["label", "f0"], ["gesture", "1"], ["gesture", "2"]])
            with manifest.open("w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerows(
                    [
                        ["sample_id", "label", "frame_path", "captured_at"],
                        ["camera-id", "gesture", str(camera_photo), "now"],
                        ["reference-id", "gesture", "", "now"],
                    ]
                )

            with (
                patch.object(train_gestures, "SAMPLES_FILE", samples),
                patch.object(train_gestures, "MANIFEST_FILE", manifest),
                patch.object(train_gestures, "CAPTURE_DIR", captures),
                patch.object(train_gestures, "ROOT", root),
            ):
                removed, removed_path, sample_id = train_gestures.remove_last_sample_with_id("gesture")

            self.assertTrue(removed)
            self.assertIsNone(removed_path)
            self.assertEqual(sample_id, "reference-id")
            self.assertTrue(camera_photo.exists())


if __name__ == "__main__":
    unittest.main()
