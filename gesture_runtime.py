from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PredictionState:
    frame_label: str | None
    frame_confidence: float
    frame_margin: float
    stable_label: str | None
    stable_confidence: float
    stable_votes: int
    required_votes: int
    triggered_label: str | None
    armed: bool


class FeatureStabilityTracker:
    def __init__(self, frames: int, threshold: float) -> None:
        self.frames = frames
        self.threshold = threshold
        self._vectors: deque[np.ndarray] = deque(maxlen=frames)

    def reset(self) -> None:
        self._vectors.clear()

    @property
    def sample_count(self) -> int:
        return len(self._vectors)

    def update(self, vector: np.ndarray | None) -> tuple[bool, float]:
        if vector is None:
            self.reset()
            return False, float("inf")

        self._vectors.append(np.asarray(vector, dtype=np.float32))
        if len(self._vectors) < self.frames:
            return False, float("inf")

        matrix = np.stack(self._vectors)
        center = matrix.mean(axis=0)
        rms = np.sqrt(np.mean((matrix - center) ** 2, axis=1))
        movement = float(rms.max())
        return movement <= self.threshold, movement


class TemporalGestureDecider:
    def __init__(
        self,
        confidence_threshold: float,
        confidence_margin: float,
        window: int,
        required_votes: int,
        release_frames: int,
        cooldown_seconds: float,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.confidence_margin = confidence_margin
        self.required_votes = required_votes
        self.release_frames = release_frames
        self.cooldown_seconds = cooldown_seconds
        self._history: deque[tuple[str | None, float]] = deque(maxlen=window)
        self._armed = True
        self._latched_label: str | None = None
        self._release_count = 0
        self._last_trigger = float("-inf")

    def observe(
        self,
        probabilities: Sequence[float] | None,
        classes: Sequence[object] | None,
        now: float,
    ) -> PredictionState:
        frame_label, frame_confidence, frame_margin = self._frame_prediction(probabilities, classes)
        accepted = (
            frame_label
            if frame_confidence >= self.confidence_threshold and frame_margin >= self.confidence_margin
            else None
        )
        self._history.append((accepted, frame_confidence if accepted else 0.0))

        self._update_arming(accepted)
        stable_label, stable_confidence, stable_votes = self._stable_prediction()

        triggered = None
        if (
            stable_label is not None
            and self._armed
            and now - self._last_trigger >= self.cooldown_seconds
        ):
            triggered = stable_label
            self._armed = False
            self._latched_label = stable_label
            self._release_count = 0
            self._last_trigger = now

        return PredictionState(
            frame_label=frame_label,
            frame_confidence=frame_confidence,
            frame_margin=frame_margin,
            stable_label=stable_label,
            stable_confidence=stable_confidence,
            stable_votes=stable_votes,
            required_votes=self.required_votes,
            triggered_label=triggered,
            armed=self._armed,
        )

    def _frame_prediction(
        self,
        probabilities: Sequence[float] | None,
        classes: Sequence[object] | None,
    ) -> tuple[str | None, float, float]:
        if probabilities is None or classes is None or len(probabilities) == 0:
            return None, 0.0, 0.0

        values = np.asarray(probabilities, dtype=np.float32)
        order = np.argsort(values)[::-1]
        best_index = int(order[0])
        second = float(values[order[1]]) if len(order) > 1 else 0.0
        best = float(values[best_index])
        return str(classes[best_index]), best, best - second

    def _stable_prediction(self) -> tuple[str | None, float, int]:
        labels = [label for label, _ in self._history if label is not None]
        if not labels:
            return None, 0.0, 0

        label, votes = Counter(labels).most_common(1)[0]
        confidences = [confidence for item, confidence in self._history if item == label]
        average = float(sum(confidences) / len(confidences))
        if votes < self.required_votes:
            return None, average, votes
        return label, average, votes

    def _update_arming(self, accepted: str | None) -> None:
        if self._armed:
            return

        if accepted is None or accepted != self._latched_label:
            self._release_count += 1
        else:
            self._release_count = 0

        if self._release_count >= self.release_frames:
            self._armed = True
            self._latched_label = None
            self._release_count = 0
            self._history.clear()
