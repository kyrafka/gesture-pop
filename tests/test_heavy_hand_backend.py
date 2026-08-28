from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from heavy_hand_backend import (
    HeavyHandAssistant,
    HeavyHandObservation,
    HeavyHandPoint,
    HeavyHandResult,
    find_heavy_model_paths,
    normalize_rtmlib_output,
)


class HeavyModelDiscoveryTests(unittest.TestCase):
    def test_finds_detector_and_pose_in_nested_downloads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            detector = root / "rtmdet" / "release" / "end2end.onnx"
            pose = root / "rtmpose" / "release" / "end2end.onnx"
            detector.parent.mkdir(parents=True)
            pose.parent.mkdir(parents=True)
            detector.write_bytes(b"det")
            pose.write_bytes(b"pose")

            found_detector, found_pose = find_heavy_model_paths(root)

            self.assertEqual(found_detector, detector)
            self.assertEqual(found_pose, pose)


class HeavyOutputTests(unittest.TestCase):
    def test_normalizes_pixel_keypoints_and_rejects_low_confidence_hand(self) -> None:
        keypoints = np.zeros((2, 21, 2), dtype=np.float32)
        keypoints[0, :, 0] = np.linspace(100, 300, 21)
        keypoints[0, :, 1] = np.linspace(50, 250, 21)
        keypoints[1] = keypoints[0]
        scores = np.vstack(
            [np.full(21, 0.8, dtype=np.float32), np.full(21, 0.02, dtype=np.float32)]
        )

        observations = normalize_rtmlib_output(keypoints, scores, width=400, height=300)

        self.assertEqual(len(observations), 1)
        self.assertAlmostEqual(observations[0].points[0].x, 0.25)
        self.assertAlmostEqual(observations[0].points[-1].y, 250 / 300)
        self.assertAlmostEqual(observations[0].confidence, 0.8, places=5)

    def test_latest_result_expires_without_starting_a_process(self) -> None:
        assistant = HeavyHandAssistant(enabled=False)
        hand = HeavyHandObservation(
            points=tuple(HeavyHandPoint(0.5, 0.5, 0.9) for _ in range(21)),
            confidence=0.9,
        )
        assistant.latest_result = HeavyHandResult(
            request_id=1,
            captured_at=10.0,
            completed_at=10.1,
            inference_ms=70.0,
            hands=(hand,),
            provider="ONNX CPU",
        )

        self.assertIsNotNone(assistant.fresh_result(now=10.3, max_age_seconds=0.45))
        self.assertIsNone(assistant.fresh_result(now=10.6, max_age_seconds=0.45))


if __name__ == "__main__":
    unittest.main()
