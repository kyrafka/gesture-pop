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

    def test_records_align_manifests_to_latest_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = root / "samples.csv"
            manifest = root / "manifest.csv"
            captures = root / "captures"
            captures.mkdir()
            first_photo = captures / "first.jpg"
            second_photo = captures / "second.jpg"
            first_photo.write_bytes(b"first")
            second_photo.write_bytes(b"second")
            with samples.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["label", "f0"])
                writer.writerows([["gesture", str(index)] for index in range(8)])
            with manifest.open("w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerows(
                    [
                        ["sample_id", "label", "frame_path", "captured_at"],
                        ["first-id", "gesture", "captures/first.jpg", "one"],
                        ["second-id", "gesture", "captures/second.jpg", "two"],
                    ]
                )

            with (
                patch.object(train_gestures, "SAMPLES_FILE", samples),
                patch.object(train_gestures, "MANIFEST_FILE", manifest),
                patch.object(train_gestures, "CAPTURE_DIR", captures),
                patch.object(train_gestures, "ROOT", root),
            ):
                records = train_gestures.load_sample_records("gesture")

            self.assertEqual(len(records), 8)
            self.assertTrue(all(record.source == "vector_only" for record in records[:6]))
            self.assertEqual([record.sample_id for record in records[6:]], ["first-id", "second-id"])
            self.assertEqual(records[6].frame_path, first_photo)

    def test_remove_specific_camera_sample_keeps_other_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = root / "samples.csv"
            manifest = root / "manifest.csv"
            captures = root / "captures"
            captures.mkdir()
            first_photo = captures / "first.jpg"
            second_photo = captures / "second.jpg"
            first_photo.write_bytes(b"first")
            second_photo.write_bytes(b"second")
            with samples.open("w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerows(
                    [["label", "f0"], ["gesture", "legacy"], ["gesture", "first"], ["gesture", "second"]]
                )
            with manifest.open("w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerows(
                    [
                        ["sample_id", "label", "frame_path", "captured_at"],
                        ["first-id", "gesture", "captures/first.jpg", "one"],
                        ["second-id", "gesture", "captures/second.jpg", "two"],
                    ]
                )

            with (
                patch.object(train_gestures, "SAMPLES_FILE", samples),
                patch.object(train_gestures, "MANIFEST_FILE", manifest),
                patch.object(train_gestures, "CAPTURE_DIR", captures),
                patch.object(train_gestures, "ROOT", root),
            ):
                record = train_gestures.load_sample_records("gesture")[1]
                removed, removed_path, sample_id = train_gestures.remove_sample_record(record)

            self.assertTrue(removed)
            self.assertEqual(removed_path, first_photo)
            self.assertEqual(sample_id, "first-id")
            self.assertFalse(first_photo.exists())
            self.assertTrue(second_photo.exists())
            with samples.open("r", newline="", encoding="utf-8") as fh:
                self.assertEqual(list(csv.reader(fh)), [["label", "f0"], ["gesture", "legacy"], ["gesture", "second"]])
            with manifest.open("r", newline="", encoding="utf-8") as fh:
                manifest_rows = list(csv.reader(fh))
            self.assertEqual([row[0] for row in manifest_rows[1:]], ["second-id"])


if __name__ == "__main__":
    unittest.main()
