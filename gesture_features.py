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
HAND_OCCLUSION_GRACE_SECONDS = 0.35


@dataclass
class FeatureResult:
    vector: np.ndarray
    debug: str
    hands: list[list[object]] = field(default_factory=list)
    faces: list[list[object]] = field(default_factory=list)
    hand_poses: list["HandPose"] = field(default_factory=list)


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


class LandmarkFeatureExtractor:
    def __init__(self) -> None:
        self._last_hands: list[list[object]] = []
        self._last_hands_seen_at = 0.0
        if hasattr(mp, "solutions"):
            self.backend = _SolutionsBackend()
            return

        _validate_model_file(HAND_MODEL_FILE, required=True)
        _validate_model_file(FACE_MODEL_FILE, required=False)

        self.backend = _TasksBackend(HAND_MODEL_FILE, FACE_MODEL_FILE if FACE_MODEL_FILE.exists() else None)

    def close(self) -> None:
        self.backend.close()

    def extract(self, frame_bgr: np.ndarray) -> FeatureResult | None:
        hands, faces = self.backend.detect(frame_bgr)
        hands = sorted(hands, key=lambda hand: hand[0].x)
        hands, using_cached_hand = self._stabilize_occluded_hands(hands)

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

        if using_cached_hand:
            debug_parts.append("occlusion_hold=yes")

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
        )

    def _stabilize_occluded_hands(self, hands: list[list[object]]) -> tuple[list[list[object]], bool]:
        now = time.monotonic()
        if len(hands) >= 2:
            self._last_hands = hands[:2]
            self._last_hands_seen_at = now
            return hands[:2], False

        if not hands:
            self._last_hands = []
            return hands, False

        if len(self._last_hands) < 2 or now - self._last_hands_seen_at > HAND_OCCLUSION_GRACE_SECONDS:
            self._last_hands = hands
            self._last_hands_seen_at = now
            return hands, False

        current_center = _hand_center(hands[0])
        previous_centers = [_hand_center(hand) for hand in self._last_hands]
        matched_index = int(np.argmin([
            math.hypot(current_center[0] - center[0], current_center[1] - center[1])
            for center in previous_centers
        ]))
        missing_index = 1 - matched_index
        stabilized = [None, None]
        stabilized[matched_index] = hands[0]
        stabilized[missing_index] = self._last_hands[missing_index]
        merged = [hand for hand in stabilized if hand is not None]
        merged = sorted(merged, key=lambda hand: hand[0].x)
        self._last_hands = merged
        return merged, True


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

        label = f"M{pose.index}  A:{pose.angle_deg:+.0f}deg  T:{pose.tilt_deg:+.0f}deg"
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


class _SolutionsBackend:
    def __init__(self) -> None:
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
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

    def detect(self, frame_bgr: np.ndarray) -> tuple[list[list[object]], list[list[object]]]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(rgb)
        face_results = self.face_mesh.process(rgb)

        hands = [list(hand.landmark) for hand in (hand_results.multi_hand_landmarks or [])]
        faces = [list(face.landmark) for face in (face_results.multi_face_landmarks or [])]
        return hands, faces


class _TasksBackend:
    def __init__(self, hand_model_path: Path, face_model_path: Path | None) -> None:
        base_options = mp.tasks.BaseOptions
        vision = mp.tasks.vision

        hand_options = vision.HandLandmarkerOptions(
            base_options=base_options(model_asset_path=str(hand_model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
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

    def detect(self, frame_bgr: np.ndarray) -> tuple[list[list[object]], list[list[object]]]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.monotonic() * 1000)

        hand_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        hands = list(hand_result.hand_landmarks)

        faces: list[list[object]] = []
        if self.face_landmarker is not None:
            face_result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
            faces = list(face_result.face_landmarks)

        return hands, faces


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
