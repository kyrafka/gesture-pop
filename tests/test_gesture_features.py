from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from gesture_features import (
    FeatureResult,
    HandDetection,
    HandIdentityTracker,
    LandmarkFeatureExtractor,
    analyze_hand_pose,
    draw_landmarks,
    merge_hand_detections,
)


def make_hand(
    wrist: tuple[float, float, float],
    middle_mcp: tuple[float, float, float],
) -> list[SimpleNamespace]:
    points = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    points[0] = SimpleNamespace(x=wrist[0], y=wrist[1], z=wrist[2])
    points[9] = SimpleNamespace(x=middle_mcp[0], y=middle_mcp[1], z=middle_mcp[2])
    return points


def make_tracking_hand(center_x: float, shape: float = 1.0) -> list[SimpleNamespace]:
    points = []
    for index in range(21):
        column = (index % 5) - 2
        row = index // 5
        points.append(
            SimpleNamespace(
                x=center_x + column * 0.018 * shape,
                y=0.72 - row * 0.055 - abs(column) * 0.006,
                z=-row * 0.008 * shape,
            )
        )
    return points


class HandPoseTests(unittest.TestCase):
    def test_upright_hand_has_zero_screen_angle(self) -> None:
        hand = make_hand((0.5, 0.8, 0.0), (0.5, 0.5, -0.1))

        pose = analyze_hand_pose(hand)

        self.assertAlmostEqual(pose.angle_deg, 0.0, places=4)
        self.assertLess(pose.tilt_deg, 0.0)
        self.assertEqual(pose.zone, "medio-centro")

    def test_hand_pointing_right_has_positive_ninety_degrees(self) -> None:
        hand = make_hand((0.2, 0.5, 0.0), (0.5, 0.5, 0.0))

        pose = analyze_hand_pose(hand)

        self.assertAlmostEqual(pose.angle_deg, 90.0, places=4)

    def test_landmark_overlay_draws_box_and_axis(self) -> None:
        hand = make_hand((0.3, 0.8, 0.0), (0.5, 0.4, -0.1))
        pose = analyze_hand_pose(hand)
        result = FeatureResult(
            vector=np.zeros(194, dtype=np.float32),
            debug="hand1=yes hand2=no face=no",
            hands=[hand],
            hand_poses=[pose],
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        draw_landmarks(frame, result)

        self.assertGreater(int(frame.sum()), 0)


class HandIdentityTrackerTests(unittest.TestCase):
    def test_preserves_hand_slots_when_positions_cross(self) -> None:
        tracker = HandIdentityTracker()
        tracker.update(
            [
                HandDetection(make_tracking_hand(0.30, 0.9), "Left", 0.95),
                HandDetection(make_tracking_hand(0.70, 1.1), "Right", 0.96),
            ],
            now=0.0,
        )
        tracker.update(
            [
                HandDetection(make_tracking_hand(0.42, 0.9), "Left", 0.95),
                HandDetection(make_tracking_hand(0.58, 1.1), "Right", 0.96),
            ],
            now=0.03,
        )
        hands, tracking = tracker.update(
            [
                HandDetection(make_tracking_hand(0.43, 1.1), "Right", 0.96),
                HandDetection(make_tracking_hand(0.57, 0.9), "Left", 0.95),
            ],
            now=0.06,
        )

        centers = [sum(point.x for point in hand) / len(hand) for hand in hands]
        self.assertGreater(centers[0], centers[1])
        self.assertEqual(tracking.handedness, ("Left", "Right"))
        self.assertTrue(tracking.identity_locked)

    def test_holds_missing_hand_only_for_short_occlusion(self) -> None:
        tracker = HandIdentityTracker()
        tracker.update(
            [
                HandDetection(make_tracking_hand(0.32), "Left", 0.9),
                HandDetection(make_tracking_hand(0.68), "Right", 0.9),
            ],
            now=1.0,
        )

        held_hands, held = tracker.update(
            [HandDetection(make_tracking_hand(0.38), "Left", 0.9)],
            now=1.20,
        )
        expired_hands, expired = tracker.update(
            [HandDetection(make_tracking_hand(0.40), "Left", 0.9)],
            now=1.90,
        )

        self.assertEqual(len(held_hands), 2)
        self.assertEqual(held.live_hands, 1)
        self.assertEqual(held.cached_hands, 1)
        self.assertEqual(held.mode, "occlusion_hold")
        self.assertEqual(len(expired_hands), 1)
        self.assertEqual(expired.cached_hands, 0)

    def test_rtpose_source_is_reported_and_keeps_previous_depth(self) -> None:
        tracker = HandIdentityTracker()
        original = make_tracking_hand(0.42)
        tracker.update([HandDetection(original, "Left", 0.9)], now=2.0)
        rtpose = [SimpleNamespace(x=point.x + 0.01, y=point.y, z=0.0) for point in original]
        scores = tuple(0.1 if index == 10 else 0.8 for index in range(21))

        hands, tracking = tracker.update(
            [
                HandDetection(
                    rtpose,
                    source="rtmpose",
                    confidence=0.8,
                    landmark_scores=scores,
                )
            ],
            now=2.05,
        )

        self.assertEqual(tracking.sources, ("rtmpose",))
        self.assertEqual(tracking.assisted_hands, 1)
        self.assertAlmostEqual(hands[0][10].x, original[10].x, places=5)
        self.assertAlmostEqual(hands[0][10].z, original[10].z, places=5)


class HandDetectionMergeTests(unittest.TestCase):
    def test_adds_only_the_heavy_hand_missing_from_mediapipe(self) -> None:
        primary = [HandDetection(make_tracking_hand(0.30), "Left", 0.9)]
        supplemental = [
            HandDetection(make_tracking_hand(0.31), source="rtmpose", confidence=0.95),
            HandDetection(make_tracking_hand(0.72), source="rtmpose", confidence=0.88),
        ]

        merged = merge_hand_detections(primary, supplemental)

        self.assertEqual(len(merged), 2)
        self.assertIs(merged[0], primary[0])
        self.assertIs(merged[1], supplemental[1])

    def test_does_not_duplicate_a_single_matching_heavy_hand(self) -> None:
        primary = [HandDetection(make_tracking_hand(0.45), "Left", 0.9)]
        supplemental = [
            HandDetection(make_tracking_hand(0.47), source="rtmpose", confidence=0.92)
        ]

        merged = merge_hand_detections(primary, supplemental)

        self.assertEqual(merged, primary)

    def test_expected_hand_count_prevents_a_phantom_second_hand(self) -> None:
        primary = [HandDetection(make_tracking_hand(0.30), "Left", 0.9)]
        supplemental = [
            HandDetection(make_tracking_hand(0.31), source="rtmpose", confidence=0.95),
            HandDetection(make_tracking_hand(0.72), source="rtmpose", confidence=0.88),
        ]
        extractor = LandmarkFeatureExtractor.__new__(LandmarkFeatureExtractor)
        extractor.hand_tracker = HandIdentityTracker()
        extractor.backend = SimpleNamespace(detect=lambda _frame: (primary, []))

        one_hand = extractor.extract(
            np.zeros((100, 100, 3), dtype=np.uint8),
            supplemental,
            expected_hands=1,
        )

        self.assertIsNotNone(one_hand)
        self.assertEqual(len(one_hand.hands), 1)


if __name__ == "__main__":
    unittest.main()
