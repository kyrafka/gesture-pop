from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from gesture_features import FeatureResult, HandPose
import reference_images
from reference_images import ReferenceAnalysis, ReferenceQuality, evaluate_reference_quality


def make_result(pose: HandPose | None) -> FeatureResult:
    hands = [[object()] * 21] if pose else []
    return FeatureResult(
        vector=np.zeros(194, dtype=np.float32),
        debug="hand1=yes hand2=no face=no" if pose else "hand1=no hand2=no face=no",
        hands=hands,
        hand_poses=[pose] if pose else [],
    )


class ReferenceQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        grid = np.indices((480, 640)).sum(axis=0) % 2
        self.sharp_image = np.repeat((grid * 180 + 35)[:, :, None], 3, axis=2).astype(np.uint8)

    def test_rejects_image_without_a_hand(self) -> None:
        quality = evaluate_reference_quality(self.sharp_image, make_result(None))

        self.assertFalse(quality.can_accept)
        self.assertEqual(quality.score, 0)

    def test_accepts_clear_centered_hand(self) -> None:
        pose = HandPose(1, 0.5, 0.5, 0.3, 0.4, 0.0, -10.0, "medio-centro", (0.35, 0.3, 0.65, 0.7))

        quality = evaluate_reference_quality(self.sharp_image, make_result(pose))

        self.assertTrue(quality.can_accept)
        self.assertEqual(quality.score, 100)

    def test_warns_about_small_cropped_hand(self) -> None:
        pose = HandPose(1, 0.04, 0.5, 0.08, 0.2, 0.0, 0.0, "medio-izq", (0.0, 0.4, 0.08, 0.6))

        quality = evaluate_reference_quality(self.sharp_image, make_result(pose))

        self.assertTrue(quality.can_accept)
        self.assertLess(quality.score, 100)
        self.assertGreaterEqual(len(quality.messages), 2)


class ReferenceStorageTests(unittest.TestCase):
    def test_stores_original_annotation_vector_and_training_state(self) -> None:
        pose = HandPose(1, 0.5, 0.5, 0.3, 0.4, 0.0, 0.0, "medio-centro", (0.35, 0.3, 0.65, 0.7))
        result = make_result(pose)
        image = np.full((360, 480, 3), 120, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jpg"
            cv2.imwrite(str(source), image)
            data_dir = root / "data"
            references = data_dir / "references"
            manifest = data_dir / "reference_manifest.csv"
            vectors = data_dir / "reference_vectors.csv"
            analysis = ReferenceAnalysis(
                source_path=source,
                original=image,
                annotated=image.copy(),
                result=result,
                quality=ReferenceQuality(True, 100, ("ok",)),
            )

            with (
                patch.object(reference_images, "ROOT", root),
                patch.object(reference_images, "DATA_DIR", data_dir),
                patch.object(reference_images, "REFERENCE_DIR", references),
                patch.object(reference_images, "REFERENCE_MANIFEST_FILE", manifest),
                patch.object(reference_images, "REFERENCE_VECTORS_FILE", vectors),
            ):
                record = reference_images.store_reference("gesture", analysis, used_for_training=True)
                loaded = reference_images.load_reference_records("gesture")
                changed = reference_images.mark_reference_not_training(record.reference_id)
                updated = reference_images.load_reference_records("gesture")

            self.assertTrue(record.original_path.is_file())
            self.assertTrue(record.annotated_path.is_file())
            self.assertTrue(vectors.is_file())
            self.assertEqual(len(loaded), 1)
            self.assertTrue(loaded[0].used_for_training)
            self.assertTrue(changed)
            self.assertFalse(updated[0].used_for_training)


if __name__ == "__main__":
    unittest.main()
