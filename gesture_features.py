from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import mediapipe as mp
import numpy as np


HAND_POINTS = 21
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
FACE_INDEXES = [1, 10, 13, 14, 33, 61, 152, 199, 263, 291]
ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"
HAND_MODEL_FILE = MODEL_DIR / "hand_landmarker.task"
FACE_MODEL_FILE = MODEL_DIR / "face_landmarker.task"


@dataclass(frozen=True)
class HandTrackingProfile:
    name: str
    occlusion_grace_seconds: float
    dropout_grace_seconds: float
    smoothing_alpha: float
    shape_weight: float
    handedness_penalty: float
    reacquire_cost: float


BALANCED_TRACKING_PROFILE = HandTrackingProfile(
    name="equilibrado",
    occlusion_grace_seconds=0.65,
    dropout_grace_seconds=0.14,
    smoothing_alpha=0.68,
    shape_weight=0.38,
    handedness_penalty=0.20,
    reacquire_cost=0.72,
)


@dataclass(frozen=True)
class LandmarkPoint:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class HandDetection:
    landmarks: list[object]
    handedness: str | None = None
    handedness_score: float = 0.0


@dataclass(frozen=True)
class HandTrackingInfo:
    mode: str = "searching"
    live_hands: int = 0
    cached_hands: int = 0
    identity_locked: bool = False
    crossing: bool = False
    overlap_ratio: float = 0.0
    handedness: tuple[str, ...] = ()


@dataclass
class FeatureResult:
    vector: np.ndarray
    debug: str
    hands: list[list[object]] = field(default_factory=list)
    faces: list[list[object]] = field(default_factory=list)
    hand_poses: list["HandPose"] = field(default_factory=list)
    tracking: HandTrackingInfo = field(default_factory=HandTrackingInfo)
    processing_ms: float = 0.0


@dataclass(frozen=True)
class HandPose:
    index: int
    center_x: float
    center_y: float
    width: float
    height: float
    angle_deg: float
    tilt_deg: float
    zone: str
    bbox: tuple[float, float, float, float]


@dataclass
class _TrackedHand:
    track_id: int
    landmarks: list[LandmarkPoint]
    center: tuple[float, float]
    velocity: tuple[float, float]
    handedness: str | None
    handedness_score: float
    last_seen_at: float


class HandIdentityTracker:
    """Keeps stable hand slots without adding another neural network."""

    def __init__(self, profile: HandTrackingProfile = BALANCED_TRACKING_PROFILE) -> None:
        self.profile = profile
        self._tracks: list[_TrackedHand] = []

    def reset(self) -> None:
        self._tracks.clear()

    def update(
        self,
        detections: list[HandDetection],
        now: float | None = None,
    ) -> tuple[list[list[LandmarkPoint]], HandTrackingInfo]:
        timestamp = time.monotonic() if now is None else now
        observations = [
            HandDetection(
                landmarks=_copy_landmarks(detection.landmarks),
                handedness=_normalize_handedness(detection.handedness),
                handedness_score=float(detection.handedness_score),
            )
            for detection in detections[:2]
            if len(detection.landmarks) >= HAND_POINTS
        ]

        if not observations:
            return self._handle_full_dropout(timestamp)

        if len(observations) == 2:
            return self._update_pair(observations, timestamp)

        return self._update_single(observations[0], timestamp)

    def _handle_full_dropout(
        self,
        timestamp: float,
    ) -> tuple[list[list[LandmarkPoint]], HandTrackingInfo]:
        if self._tracks and timestamp - max(track.last_seen_at for track in self._tracks) <= self.profile.dropout_grace_seconds:
            hands = self._ordered_landmarks()
            return hands, self._tracking_info(
                mode="dropout_hold",
                live_hands=0,
                cached_hands=len(hands),
                identity_locked=len(hands) == 2,
            )

        self.reset()
        return [], HandTrackingInfo()

    def _update_pair(
        self,
        observations: list[HandDetection],
        timestamp: float,
    ) -> tuple[list[list[LandmarkPoint]], HandTrackingInfo]:
        overlap = _hand_overlap_ratio([observation.landmarks for observation in observations])
        crossing = overlap >= 0.10

        if len(self._tracks) != 2:
            self._initialize_tracks(observations, timestamp)
            mode = "crossing" if crossing else "locked"
            return self._ordered_landmarks(), self._tracking_info(
                mode=mode,
                live_hands=2,
                identity_locked=True,
                crossing=crossing,
                overlap_ratio=overlap,
            )

        direct_cost = self._match_cost(self._tracks[0], observations[0]) + self._match_cost(
            self._tracks[1], observations[1]
        )
        swapped_cost = self._match_cost(self._tracks[0], observations[1]) + self._match_cost(
            self._tracks[1], observations[0]
        )
        assigned = observations if direct_cost <= swapped_cost else [observations[1], observations[0]]
        best_cost = min(direct_cost, swapped_cost) / 2.0

        if best_cost > self.profile.reacquire_cost:
            self._initialize_tracks(observations, timestamp)
            mode = "reacquired"
        else:
            for track, observation in zip(self._tracks, assigned):
                self._update_track(track, observation, timestamp)
            mode = "crossing" if crossing else "locked"

        return self._ordered_landmarks(), self._tracking_info(
            mode=mode,
            live_hands=2,
            identity_locked=True,
            crossing=crossing,
            overlap_ratio=overlap,
        )

    def _update_single(
        self,
        observation: HandDetection,
        timestamp: float,
    ) -> tuple[list[list[LandmarkPoint]], HandTrackingInfo]:
        if not self._tracks:
            self._initialize_tracks([observation], timestamp)
            return self._ordered_landmarks(), self._tracking_info(mode="single", live_hands=1)

        if len(self._tracks) == 1:
            self._update_track(self._tracks[0], observation, timestamp)
            return self._ordered_landmarks(), self._tracking_info(mode="single", live_hands=1)

        matched_index = int(np.argmin([self._match_cost(track, observation) for track in self._tracks]))
        missing_index = 1 - matched_index
        self._update_track(self._tracks[matched_index], observation, timestamp)
        missing_age = timestamp - self._tracks[missing_index].last_seen_at

        if missing_age <= self.profile.occlusion_grace_seconds:
            overlap = _hand_overlap_ratio(self._ordered_landmarks())
            return self._ordered_landmarks(), self._tracking_info(
                mode="occlusion_hold",
                live_hands=1,
                cached_hands=1,
                identity_locked=True,
                crossing=True,
                overlap_ratio=overlap,
            )

        remaining = self._tracks[matched_index]
        remaining.track_id = 0
        self._tracks = [remaining]
        return self._ordered_landmarks(), self._tracking_info(mode="single", live_hands=1)

    def _initialize_tracks(self, observations: list[HandDetection], timestamp: float) -> None:
        ordered = sorted(observations, key=lambda observation: _hand_center(observation.landmarks)[0])
        self._tracks = []
        for track_id, observation in enumerate(ordered):
            landmarks = _copy_landmarks(observation.landmarks)
            self._tracks.append(
                _TrackedHand(
                    track_id=track_id,
                    landmarks=landmarks,
                    center=_hand_center(landmarks),
                    velocity=(0.0, 0.0),
                    handedness=_normalize_handedness(observation.handedness),
                    handedness_score=float(observation.handedness_score),
                    last_seen_at=timestamp,
                )
            )

    def _match_cost(self, track: _TrackedHand, observation: HandDetection) -> float:
        center = _hand_center(observation.landmarks)
        predicted = (
            track.center[0] + track.velocity[0],
            track.center[1] + track.velocity[1],
        )
        position_cost = math.hypot(center[0] - predicted[0], center[1] - predicted[1])
        previous_shape = np.asarray(
            _normalized_landmarks((point.x, point.y, point.z) for point in track.landmarks),
            dtype=np.float32,
        )
        current_shape = np.asarray(
            _normalized_landmarks((point.x, point.y, point.z) for point in observation.landmarks),
            dtype=np.float32,
        )
        shape_cost = float(np.sqrt(np.mean((previous_shape - current_shape) ** 2)))
        handedness = _normalize_handedness(observation.handedness)
        handedness_cost = 0.0
        if track.handedness and handedness and track.handedness != handedness:
            confidence = min(track.handedness_score, float(observation.handedness_score))
            handedness_cost = self.profile.handedness_penalty * max(0.45, confidence)
        return position_cost + self.profile.shape_weight * shape_cost + handedness_cost

    def _update_track(self, track: _TrackedHand, observation: HandDetection, timestamp: float) -> None:
        raw_landmarks = _copy_landmarks(observation.landmarks)
        raw_center = _hand_center(raw_landmarks)
        delta = (raw_center[0] - track.center[0], raw_center[1] - track.center[1])
        speed = math.hypot(*delta)
        alpha = min(0.88, self.profile.smoothing_alpha + max(0.0, speed - 0.12))
        track.landmarks = _blend_landmarks(track.landmarks, raw_landmarks, alpha)
        new_center = _hand_center(track.landmarks)
        track.velocity = (
            track.velocity[0] * 0.45 + (new_center[0] - track.center[0]) * 0.55,
            track.velocity[1] * 0.45 + (new_center[1] - track.center[1]) * 0.55,
        )
        track.center = new_center
        handedness = _normalize_handedness(observation.handedness)
        score = float(observation.handedness_score)
        if handedness and (
            not track.handedness
            or handedness == track.handedness
            or score >= track.handedness_score + 0.18
        ):
            track.handedness = handedness
            track.handedness_score = score
        track.last_seen_at = timestamp

    def _ordered_landmarks(self) -> list[list[LandmarkPoint]]:
        return [track.landmarks for track in sorted(self._tracks, key=lambda item: item.track_id)]

    def _tracking_info(
        self,
        mode: str,
        live_hands: int,
        cached_hands: int = 0,
        identity_locked: bool = False,
        crossing: bool = False,
        overlap_ratio: float = 0.0,
    ) -> HandTrackingInfo:
        tracks = sorted(self._tracks, key=lambda item: item.track_id)
        return HandTrackingInfo(
            mode=mode,
            live_hands=live_hands,
            cached_hands=cached_hands,
            identity_locked=identity_locked,
            crossing=crossing,
            overlap_ratio=overlap_ratio,
            handedness=tuple(track.handedness or "?" for track in tracks),
        )


class LandmarkFeatureExtractor:
    def __init__(self) -> None:
        self.hand_tracker = HandIdentityTracker()
        if hasattr(mp, "solutions"):
            self.backend = _SolutionsBackend()
            return

        _validate_model_file(HAND_MODEL_FILE, required=True)
        _validate_model_file(FACE_MODEL_FILE, required=False)

        self.backend = _TasksBackend(HAND_MODEL_FILE, FACE_MODEL_FILE if FACE_MODEL_FILE.exists() else None)

    def close(self) -> None:
        self.backend.close()

    def extract(self, frame_bgr: np.ndarray) -> FeatureResult | None:
        started_at = time.perf_counter()
        detections, faces = self.backend.detect(frame_bgr)
        hands, tracking = self.hand_tracker.update(detections)

        parts: list[float] = []
        debug_parts: list[str] = []

        for hand_index in range(2):
            if hand_index < len(hands):
                hand_vec = _normalized_landmarks(
                    [(lm.x, lm.y, lm.z) for lm in hands[hand_index]]
                )
                parts.extend(hand_vec)
                parts.extend(_hand_distances(hands[hand_index]))
                debug_parts.append(f"hand{hand_index + 1}=yes")
            else:
                parts.extend([0.0] * (HAND_POINTS * 3 + 15))
                debug_parts.append(f"hand{hand_index + 1}=no")

        if tracking.cached_hands:
            debug_parts.append("occlusion_hold=yes")
        if tracking.identity_locked:
            debug_parts.append("identity_lock=yes")
        if tracking.crossing:
            debug_parts.append("crossing=yes")
        debug_parts.append(f"tracking={tracking.mode}")

        if faces:
            face = faces[0]
            face_points = [face[i] for i in FACE_INDEXES]
            parts.extend(_normalized_landmarks([(lm.x, lm.y, lm.z) for lm in face_points]))
            parts.extend(_face_hand_distances(face, hands))
            debug_parts.append("face=yes")
        else:
            parts.extend([0.0] * (len(FACE_INDEXES) * 3 + 8))
            debug_parts.append("face=no")

        if not hands and not faces:
            return None

        return FeatureResult(
            np.array(parts, dtype=np.float32),
            " ".join(debug_parts),
            hands=hands,
            faces=faces,
            hand_poses=[analyze_hand_pose(hand, index + 1) for index, hand in enumerate(hands)],
            tracking=tracking,
            processing_ms=(time.perf_counter() - started_at) * 1000.0,
        )


def draw_landmarks(frame_bgr: np.ndarray, result: FeatureResult | None) -> None:
    if result is None:
        return

    h, w = frame_bgr.shape[:2]

    for hand_index, hand in enumerate(result.hands):
        pose = (
            result.hand_poses[hand_index]
            if hand_index < len(result.hand_poses)
            else analyze_hand_pose(hand, hand_index + 1)
        )
        color = (70, 220, 110) if hand_index == 0 else (80, 185, 255)
        for start, end in HAND_CONNECTIONS:
            cv2.line(
                frame_bgr,
                _to_pixel(hand[start], w, h),
                _to_pixel(hand[end], w, h),
                color,
                2,
            )
        for point in hand:
            cv2.circle(frame_bgr, _to_pixel(point, w, h), 3, color, -1)

        x1, y1, x2, y2 = _bbox_to_pixels(pose.bbox, w, h)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        center = (int(pose.center_x * w), int(pose.center_y * h))
        cv2.drawMarker(frame_bgr, center, color, cv2.MARKER_CROSS, 14, 2)
        cv2.arrowedLine(
            frame_bgr,
            _to_pixel(hand[0], w, h),
            _to_pixel(hand[9], w, h),
            color,
            3,
            tipLength=0.22,
        )

        side = result.tracking.handedness[hand_index] if hand_index < len(result.tracking.handedness) else "?"
        label = f"M{pose.index}/{side[:1]}  A:{pose.angle_deg:+.0f}deg  T:{pose.tilt_deg:+.0f}deg"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        label_y = max(0, y1 - 25)
        label_x2 = min(w - 1, x1 + text_size[0] + 12)
        cv2.rectangle(frame_bgr, (x1, label_y), (label_x2, label_y + 23), (18, 24, 29), -1)
        cv2.rectangle(frame_bgr, (x1, label_y), (label_x2, label_y + 23), color, 1)
        cv2.putText(
            frame_bgr,
            label,
            (x1 + 6, label_y + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    if len(result.hand_poses) == 2 and result.tracking.identity_locked:
        first, second = result.hand_poses
        first_center = (int(first.center_x * w), int(first.center_y * h))
        second_center = (int(second.center_x * w), int(second.center_y * h))
        relation_color = (80, 190, 255) if result.tracking.crossing else (130, 155, 170)
        cv2.line(frame_bgr, first_center, second_center, relation_color, 1, cv2.LINE_AA)
        if result.tracking.crossing:
            midpoint = (
                int((first_center[0] + second_center[0]) / 2),
                max(24, int((first_center[1] + second_center[1]) / 2) - 12),
            )
            cv2.putText(
                frame_bgr,
                "CRUCE: ID FIJA",
                midpoint,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                relation_color,
                1,
                cv2.LINE_AA,
            )

    for face in result.faces:
        for index in FACE_INDEXES:
            cv2.circle(frame_bgr, _to_pixel(face[index], w, h), 2, (255, 180, 0), -1)


def summarize_vector(result: FeatureResult | None) -> list[str]:
    if result is None:
        return ["sin vector"]

    vector = result.vector
    hand1_start = 0
    hand1_len = HAND_POINTS * 3
    hand1_dist_start = hand1_start + hand1_len
    hand2_start = hand1_dist_start + 15
    hand2_dist_start = hand2_start + hand1_len
    face_start = hand2_dist_start + 15
    face_len = len(FACE_INDEXES) * 3
    face_dist_start = face_start + face_len

    hand1_energy = float(np.linalg.norm(vector[hand1_start:hand1_start + hand1_len]))
    hand2_energy = float(np.linalg.norm(vector[hand2_start:hand2_start + hand1_len]))
    face_energy = float(np.linalg.norm(vector[face_start:face_start + face_len]))

    thumb_index_1 = float(vector[hand1_dist_start]) if len(vector) > hand1_dist_start else 0.0
    wrist_index_1 = float(vector[hand1_dist_start + 8]) if len(vector) > hand1_dist_start + 8 else 0.0
    face_hand_1 = float(vector[face_dist_start]) if len(vector) > face_dist_start else 0.0

    return [
        f"vec| h1={hand1_energy:.2f} h2={hand2_energy:.2f} face={face_energy:.2f}",
        f"dist| pulgar-indice={thumb_index_1:.2f} muneca-indice={wrist_index_1:.2f}",
        f"cara| nariz-muneca={face_hand_1:.2f}",
    ]


def analyze_hand_pose(landmarks, index: int = 1) -> HandPose:
    points = np.array([(point.x, point.y, point.z) for point in landmarks], dtype=np.float32)
    min_x, min_y = points[:, :2].min(axis=0)
    max_x, max_y = points[:, :2].max(axis=0)
    padding = 0.025
    x1 = max(0.0, float(min_x - padding))
    y1 = max(0.0, float(min_y - padding))
    x2 = min(1.0, float(max_x + padding))
    y2 = min(1.0, float(max_y + padding))
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    wrist = points[0]
    middle_mcp = points[9]
    direction = middle_mcp - wrist
    angle_deg = math.degrees(math.atan2(float(direction[0]), float(-direction[1])))
    planar_length = max(float(np.linalg.norm(direction[:2])), 1e-6)
    tilt_deg = math.degrees(math.atan2(float(direction[2]), planar_length))

    return HandPose(
        index=index,
        center_x=center_x,
        center_y=center_y,
        width=x2 - x1,
        height=y2 - y1,
        angle_deg=angle_deg,
        tilt_deg=tilt_deg,
        zone=_screen_zone(center_x, center_y),
        bbox=(x1, y1, x2, y2),
    )


def _hand_center(landmarks) -> tuple[float, float]:
    points = np.array([(point.x, point.y) for point in landmarks], dtype=np.float32)
    center = points.mean(axis=0)
    return float(center[0]), float(center[1])


def _copy_landmarks(landmarks) -> list[LandmarkPoint]:
    return [LandmarkPoint(float(point.x), float(point.y), float(point.z)) for point in landmarks]


def _blend_landmarks(
    previous: list[LandmarkPoint],
    current: list[LandmarkPoint],
    alpha: float,
) -> list[LandmarkPoint]:
    if len(previous) != len(current):
        return current
    previous_weight = 1.0 - alpha
    return [
        LandmarkPoint(
            x=old.x * previous_weight + new.x * alpha,
            y=old.y * previous_weight + new.y * alpha,
            z=old.z * previous_weight + new.z * alpha,
        )
        for old, new in zip(previous, current)
    ]


def _normalize_handedness(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower()
    if normalized.startswith("l"):
        return "Left"
    if normalized.startswith("r"):
        return "Right"
    return value.strip()


def _hand_overlap_ratio(hands) -> float:
    if len(hands) < 2:
        return 0.0
    boxes: list[tuple[float, float, float, float]] = []
    for hand in hands[:2]:
        points = np.asarray([(point.x, point.y) for point in hand], dtype=np.float32)
        min_x, min_y = points.min(axis=0)
        max_x, max_y = points.max(axis=0)
        boxes.append((float(min_x), float(min_y), float(max_x), float(max_y)))
    first, second = boxes
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = max((first[2] - first[0]) * (first[3] - first[1]), 1e-6)
    second_area = max((second[2] - second[0]) * (second[3] - second[1]), 1e-6)
    return min(1.0, intersection / min(first_area, second_area))


def _solutions_handedness_label(items, index: int) -> str | None:
    try:
        return _normalize_handedness(items[index].classification[0].label)
    except (AttributeError, IndexError, TypeError):
        return None


def _solutions_handedness_score(items, index: int) -> float:
    try:
        return float(items[index].classification[0].score)
    except (AttributeError, IndexError, TypeError, ValueError):
        return 0.0


def _tasks_handedness_label(items, index: int) -> str | None:
    try:
        category = items[index][0]
        return _normalize_handedness(category.category_name or category.display_name)
    except (AttributeError, IndexError, TypeError):
        return None


def _tasks_handedness_score(items, index: int) -> float:
    try:
        return float(items[index][0].score)
    except (AttributeError, IndexError, TypeError, ValueError):
        return 0.0


class _SolutionsBackend:
    def __init__(self) -> None:
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.68,
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    def close(self) -> None:
        self.hands.close()
        self.face_mesh.close()

    def detect(self, frame_bgr: np.ndarray) -> tuple[list[HandDetection], list[list[object]]]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(rgb)
        face_results = self.face_mesh.process(rgb)

        hands = [list(hand.landmark) for hand in (hand_results.multi_hand_landmarks or [])]
        handedness = list(hand_results.multi_handedness or [])
        detections = [
            HandDetection(
                landmarks=hand,
                handedness=_solutions_handedness_label(handedness, index),
                handedness_score=_solutions_handedness_score(handedness, index),
            )
            for index, hand in enumerate(hands)
        ]
        faces = [list(face.landmark) for face in (face_results.multi_face_landmarks or [])]
        return detections, faces


class _TasksBackend:
    def __init__(self, hand_model_path: Path, face_model_path: Path | None) -> None:
        base_options = mp.tasks.BaseOptions
        vision = mp.tasks.vision

        hand_options = vision.HandLandmarkerOptions(
            base_options=base_options(model_asset_path=str(hand_model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.52,
            min_tracking_confidence=0.68,
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

        self.face_landmarker = None
        if face_model_path is not None:
            face_options = vision.FaceLandmarkerOptions(
                base_options=base_options(model_asset_path=str(face_model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.6,
                min_face_presence_confidence=0.6,
                min_tracking_confidence=0.6,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self.face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

    def close(self) -> None:
        self.hand_landmarker.close()
        if self.face_landmarker is not None:
            self.face_landmarker.close()

    def detect(self, frame_bgr: np.ndarray) -> tuple[list[HandDetection], list[list[object]]]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.monotonic() * 1000)

        hand_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        hands = list(hand_result.hand_landmarks)
        handedness = list(hand_result.handedness)
        detections = [
            HandDetection(
                landmarks=list(hand),
                handedness=_tasks_handedness_label(handedness, index),
                handedness_score=_tasks_handedness_score(handedness, index),
            )
            for index, hand in enumerate(hands)
        ]

        faces: list[list[object]] = []
        if self.face_landmarker is not None:
            face_result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
            faces = list(face_result.face_landmarks)

        return detections, faces


def _normalized_landmarks(points: Iterable[tuple[float, float, float]]) -> list[float]:
    arr = np.array(list(points), dtype=np.float32)
    center = arr.mean(axis=0)
    shifted = arr - center
    scale = float(np.max(np.linalg.norm(shifted[:, :2], axis=1)))
    if scale < 1e-6:
        scale = 1.0
    return (shifted / scale).flatten().tolist()


def _distance(a, b) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _hand_distances(landmarks) -> list[float]:
    pairs = [
        (4, 8), (4, 12), (4, 16), (4, 20),
        (8, 12), (12, 16), (16, 20),
        (0, 4), (0, 8), (0, 12), (0, 16), (0, 20),
        (5, 8), (9, 12), (13, 16),
    ]
    palm = max(_distance(landmarks[0], landmarks[9]), 1e-6)
    return [_distance(landmarks[a], landmarks[b]) / palm for a, b in pairs]


def _face_hand_distances(face_landmarks, hands) -> list[float]:
    if not hands:
        return [0.0] * 8

    nose = face_landmarks[1]
    mouth = face_landmarks[13]
    values: list[float] = []
    for hand_index in range(2):
        if hand_index < len(hands):
            wrist = hands[hand_index][0]
            index_tip = hands[hand_index][8]
            values.extend([
                _distance(nose, wrist),
                _distance(nose, index_tip),
                _distance(mouth, wrist),
                _distance(mouth, index_tip),
            ])
        else:
            values.extend([0.0] * 4)
    return values


def _to_pixel(point, width: int, height: int) -> tuple[int, int]:
    return int(point.x * width), int(point.y * height)


def _bbox_to_pixels(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(width - 1, int(x1 * width))),
        max(0, min(height - 1, int(y1 * height))),
        max(0, min(width - 1, int(x2 * width))),
        max(0, min(height - 1, int(y2 * height))),
    )


def _screen_zone(center_x: float, center_y: float) -> str:
    horizontal = "izq" if center_x < 1 / 3 else "der" if center_x > 2 / 3 else "centro"
    vertical = "arriba" if center_y < 1 / 3 else "abajo" if center_y > 2 / 3 else "medio"
    return f"{vertical}-{horizontal}"


def _validate_model_file(path: Path, required: bool) -> None:
    if not path.exists():
        if required:
            raise RuntimeError(
                f"No encontre {path.as_posix()}. "
                "Tu instalacion actual de MediaPipe usa mediapipe.tasks y necesita ese modelo."
            )
        return

    if not path.is_file():
        raise RuntimeError(
            f"{path.as_posix()} existe, pero no es un archivo valido. "
            "Borra eso y coloca el .task real."
        )

    size = path.stat().st_size
    if size <= 0:
        raise RuntimeError(
            f"{path.as_posix()} existe, pero esta vacio. "
            "Descarga de nuevo el .task real y reemplazalo."
        )
