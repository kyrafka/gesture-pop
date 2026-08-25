from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from gesture_features import FeatureResult, analyze_hand_pose, draw_landmarks


def make_hand(
    wrist: tuple[float, float, float],
    middle_mcp: tuple[float, float, float],
) -> list[SimpleNamespace]:
    points = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    points[0] = SimpleNamespace(x=wrist[0], y=wrist[1], z=wrist[2])
    points[9] = SimpleNamespace(x=middle_mcp[0], y=middle_mcp[1], z=middle_mcp[2])
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


if __name__ == "__main__":
    unittest.main()
