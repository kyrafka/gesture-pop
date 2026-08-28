from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from gesture_features import (
    FeatureResult,
    HandDetection,
    HandIdentityTracker,
    analyze_hand_pose,
    draw_landmarks,
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


if __name__ == "__main__":
    unittest.main()
