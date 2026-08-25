from __future__ import annotations

import unittest

from gesture_features import HandPose
from guided_capture import BASE_TARGETS, build_capture_targets


def pose_at(x: float, y: float) -> HandPose:
    return HandPose(
        index=1,
        center_x=x,
        center_y=y,
        width=0.2,
        height=0.3,
        angle_deg=0.0,
        tilt_deg=0.0,
        zone="test",
        bbox=(x - 0.1, y - 0.15, x + 0.1, y + 0.15),
    )


class GuidedCaptureTests(unittest.TestCase):
    def test_targets_cover_requested_sample_count(self) -> None:
        targets = build_capture_targets(14)

        self.assertEqual(len(targets), 14)
        self.assertEqual(targets[0], BASE_TARGETS[0])
        self.assertEqual(targets[6], BASE_TARGETS[0])

    def test_target_accepts_only_pose_inside_bounds(self) -> None:
        center = BASE_TARGETS[0]

        self.assertTrue(center.matches(pose_at(0.5, 0.5)))
        self.assertFalse(center.matches(pose_at(0.1, 0.5)))
        self.assertFalse(center.matches(None))

    def test_zero_total_has_no_targets(self) -> None:
        self.assertEqual(build_capture_targets(0), [])


if __name__ == "__main__":
    unittest.main()
