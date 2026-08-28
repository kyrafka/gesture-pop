from __future__ import annotations

import unittest

import numpy as np

from gesture_runtime import FeatureStabilityTracker, TemporalGestureDecider


class FeatureStabilityTrackerTests(unittest.TestCase):
    def test_requires_a_full_stable_window(self) -> None:
        tracker = FeatureStabilityTracker(frames=3, threshold=0.05)

        self.assertFalse(tracker.update(np.array([1.0, 2.0]))[0])
        self.assertFalse(tracker.update(np.array([1.01, 2.01]))[0])
        stable, movement = tracker.update(np.array([0.99, 1.99]))

        self.assertTrue(stable)
        self.assertLess(movement, 0.05)

    def test_resets_when_landmarks_disappear(self) -> None:
        tracker = FeatureStabilityTracker(frames=2, threshold=0.05)
        tracker.update(np.array([1.0]))
        tracker.update(None)

        self.assertEqual(tracker.sample_count, 0)

    def test_restarts_window_after_an_extreme_vector_jump(self) -> None:
        tracker = FeatureStabilityTracker(frames=3, threshold=0.05, jump_threshold=0.4)
        tracker.update(np.array([0.0, 0.0]))
        tracker.update(np.array([0.01, 0.01]))

        stable, movement = tracker.update(np.array([2.0, 2.0]))

        self.assertFalse(stable)
        self.assertGreater(movement, 0.4)
        self.assertEqual(tracker.sample_count, 1)


class TemporalGestureDeciderTests(unittest.TestCase):
    def make_decider(self) -> TemporalGestureDecider:
        return TemporalGestureDecider(
            confidence_threshold=0.65,
            confidence_margin=0.20,
            window=5,
            required_votes=3,
            release_frames=2,
            cooldown_seconds=0.0,
        )

    def test_triggers_only_after_enough_votes(self) -> None:
        decider = self.make_decider()
        classes = ["a", "b"]

        first = decider.observe([0.9, 0.1], classes, now=1.0)
        second = decider.observe([0.85, 0.15], classes, now=2.0)
        third = decider.observe([0.8, 0.2], classes, now=3.0)

        self.assertIsNone(first.triggered_label)
        self.assertIsNone(second.triggered_label)
        self.assertEqual(third.triggered_label, "a")

    def test_held_gesture_does_not_repeat_until_released(self) -> None:
        decider = self.make_decider()
        classes = ["a", "b"]
        for now in range(1, 4):
            state = decider.observe([0.9, 0.1], classes, now=float(now))
        self.assertEqual(state.triggered_label, "a")

        held = decider.observe([0.9, 0.1], classes, now=4.0)
        self.assertIsNone(held.triggered_label)
        decider.observe(None, None, now=5.0)
        released = decider.observe(None, None, now=6.0)
        self.assertTrue(released.armed)

        triggered = None
        for now in range(7, 10):
            triggered = decider.observe([0.9, 0.1], classes, now=float(now)).triggered_label
        self.assertEqual(triggered, "a")

    def test_rejects_small_margin(self) -> None:
        decider = self.make_decider()
        state = None
        for now in range(1, 6):
            state = decider.observe([0.55, 0.45], ["a", "b"], now=float(now))

        self.assertIsNone(state.stable_label)
        self.assertIsNone(state.triggered_label)


if __name__ == "__main__":
    unittest.main()
