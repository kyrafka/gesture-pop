from __future__ import annotations

import importlib.util
import multiprocessing
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from gesture_features import HandDetection


ROOT = Path(__file__).parent
HEAVY_MODEL_DIR = ROOT / "models" / "heavy"
HEAVY_MAX_HANDS = 2


@dataclass(frozen=True)
class HeavyHandPoint:
    x: float
    y: float
    score: float


@dataclass(frozen=True)
class HeavyHandObservation:
    points: tuple[HeavyHandPoint, ...]
    confidence: float


@dataclass(frozen=True)
class HeavyHandResult:
    request_id: int
    captured_at: float
    completed_at: float
    inference_ms: float
    hands: tuple[HeavyHandObservation, ...]
    provider: str
    error: str = ""


@dataclass(frozen=True)
class HeavyAssistantStatus:
    state: str
    message: str
    provider: str = ""
    inference_ms: float | None = None
    hands: int = 0


def find_heavy_model_paths(model_dir: Path = HEAVY_MODEL_DIR) -> tuple[Path | None, Path | None]:
    if not model_dir.exists():
        return None, None

    models = sorted(model_dir.rglob("*.onnx"))
    detector = next((path for path in models if "rtmdet" in str(path).lower()), None)
    pose = next((path for path in models if "rtmpose" in str(path).lower()), None)
    return detector, pose


def heavy_assistant_availability(model_dir: Path = HEAVY_MODEL_DIR) -> tuple[bool, str]:
    if importlib.util.find_spec("rtmlib") is None or importlib.util.find_spec("onnxruntime") is None:
        return False, "Falta instalar requirements-heavy.txt"

    detector, pose = find_heavy_model_paths(model_dir)
    missing = []
    if detector is None:
        missing.append("RTMDet")
    if pose is None:
        missing.append("RTMPose")
    if missing:
        return False, f"Faltan modelos: {', '.join(missing)}"
    return True, "RTMPose disponible"


def normalize_rtmlib_output(
    keypoints: object,
    scores: object,
    width: int,
    height: int,
    min_point_score: float = 0.15,
) -> tuple[HeavyHandObservation, ...]:
    points_array = np.asarray(keypoints, dtype=np.float32)
    scores_array = np.asarray(scores, dtype=np.float32)
    if points_array.size == 0 or width < 1 or height < 1:
        return ()
    if points_array.ndim == 2:
        points_array = points_array[np.newaxis, ...]
    if scores_array.ndim == 1:
        scores_array = scores_array[np.newaxis, ...]
    if points_array.ndim != 3 or points_array.shape[1] < 21 or points_array.shape[2] < 2:
        return ()

    observations: list[HeavyHandObservation] = []
    for hand_index in range(min(len(points_array), HEAVY_MAX_HANDS)):
        hand_points = points_array[hand_index, :21, :2]
        if hand_index < len(scores_array):
            hand_scores = np.asarray(scores_array[hand_index, :21], dtype=np.float32)
        else:
            hand_scores = np.ones(21, dtype=np.float32)
        if len(hand_scores) < 21 or not np.isfinite(hand_points).all():
            continue

        finite_scores = np.where(np.isfinite(hand_scores), hand_scores, 0.0)
        visible_points = int(np.count_nonzero(finite_scores >= min_point_score))
        confidence = float(np.clip(np.mean(finite_scores), 0.0, 1.0))
        if visible_points < 8 or confidence < min_point_score:
            continue

        normalized = tuple(
            HeavyHandPoint(
                x=float(np.clip(point[0] / width, 0.0, 1.0)),
                y=float(np.clip(point[1] / height, 0.0, 1.0)),
                score=float(np.clip(score, 0.0, 1.0)),
            )
            for point, score in zip(hand_points, finite_scores)
        )
        observations.append(HeavyHandObservation(points=normalized, confidence=confidence))
    return tuple(observations)


def result_to_hand_detections(result: HeavyHandResult | None) -> list[HandDetection]:
    from gesture_features import HandDetection, LandmarkPoint

    if result is None:
        return []
    return [
        HandDetection(
            landmarks=[LandmarkPoint(point.x, point.y, 0.0) for point in observation.points],
            source="rtmpose",
            confidence=observation.confidence,
            landmark_scores=tuple(point.score for point in observation.points),
        )
        for observation in result.hands
    ]


class HeavyHandAssistant:
    """Runs RTMPose in a disposable process so camera frames never wait for it."""

    def __init__(
        self,
        enabled: bool = True,
        model_dir: Path = HEAVY_MODEL_DIR,
        max_frame_width: int = 960,
    ) -> None:
        self.enabled = enabled
        self.model_dir = model_dir
        self.max_frame_width = max(320, int(max_frame_width))
        self.detector_path, self.pose_path = find_heavy_model_paths(model_dir)
        self.status = HeavyAssistantStatus("off", "RTMPose desactivado")
        self.latest_result: HeavyHandResult | None = None
        self._context = None
        self._request_queue = None
        self._result_queue = None
        self._process = None
        self._pending_request_id: int | None = None
        self._next_request_id = 1

    def start(self) -> HeavyAssistantStatus:
        if not self.enabled:
            return self.status

        available, message = heavy_assistant_availability(self.model_dir)
        if not available or self.detector_path is None or self.pose_path is None:
            self.status = HeavyAssistantStatus("unavailable", message)
            return self.status

        try:
            self._context = multiprocessing.get_context("spawn")
            self._request_queue = self._context.Queue(maxsize=1)
            self._result_queue = self._context.Queue(maxsize=8)
            self._process = self._context.Process(
                target=_heavy_worker_main,
                args=(
                    str(self.detector_path),
                    str(self.pose_path),
                    self._request_queue,
                    self._result_queue,
                ),
                name="gesture-pop-rtmpose",
                daemon=True,
            )
            self._process.start()
        except Exception as exc:
            self.status = HeavyAssistantStatus("error", f"RTMPose no inicio: {exc}")
            return self.status

        self.status = HeavyAssistantStatus("starting", "Cargando RTMPose en segundo plano...")
        return self.status

    @property
    def is_running(self) -> bool:
        return bool(self._process is not None and self._process.is_alive())

    @property
    def has_pending_request(self) -> bool:
        return self._pending_request_id is not None

    def submit(self, frame_bgr: np.ndarray, reason: str, captured_at: float | None = None) -> bool:
        if not self.is_running or self._request_queue is None or self.has_pending_request:
            return False

        frame = frame_bgr
        height, width = frame.shape[:2]
        if width > self.max_frame_width:
            scale = self.max_frame_width / float(width)
            frame = cv2.resize(
                frame,
                (self.max_frame_width, max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if not ok:
            return False

        request_id = self._next_request_id
        request = (
            "infer",
            request_id,
            time.monotonic() if captured_at is None else captured_at,
            encoded.tobytes(),
            reason,
        )
        try:
            self._request_queue.put_nowait(request)
        except queue.Full:
            return False

        self._next_request_id += 1
        self._pending_request_id = request_id
        self.status = HeavyAssistantStatus("active", f"RTMPose analizando ({reason})...")
        return True

    def poll(self) -> tuple[HeavyHandResult | None, HeavyAssistantStatus | None]:
        if self._result_queue is None:
            return None, None

        newest_result: HeavyHandResult | None = None
        newest_status: HeavyAssistantStatus | None = None
        while True:
            try:
                message = self._result_queue.get_nowait()
            except queue.Empty:
                break

            kind = message[0]
            if kind == "status":
                newest_status = HeavyAssistantStatus(
                    state=message[1],
                    message=message[2],
                    provider=message[3],
                )
                self.status = newest_status
                continue

            if kind != "result":
                continue
            newest_result = HeavyHandResult(
                request_id=message[1],
                captured_at=message[2],
                completed_at=message[3],
                inference_ms=message[4],
                hands=message[5],
                provider=message[6],
                error=message[7],
            )
            if newest_result.request_id == self._pending_request_id:
                self._pending_request_id = None
            self.latest_result = newest_result
            if newest_result.error:
                newest_status = HeavyAssistantStatus(
                    "error",
                    f"RTMPose fallo: {newest_result.error}",
                    newest_result.provider,
                    newest_result.inference_ms,
                )
            else:
                hand_word = "mano" if len(newest_result.hands) == 1 else "manos"
                newest_status = HeavyAssistantStatus(
                    "ready",
                    f"RTMPose listo · {newest_result.inference_ms:.0f} ms · {len(newest_result.hands)} {hand_word}",
                    newest_result.provider,
                    newest_result.inference_ms,
                    len(newest_result.hands),
                )
            self.status = newest_status

        if self._process is not None and not self._process.is_alive() and self.status.state not in {
            "error",
            "off",
        }:
            self._pending_request_id = None
            newest_status = HeavyAssistantStatus("error", "RTMPose se detuvo; MediaPipe sigue activo")
            self.status = newest_status
        return newest_result, newest_status

    def fresh_result(self, now: float | None = None, max_age_seconds: float = 0.45) -> HeavyHandResult | None:
        if self.latest_result is None or self.latest_result.error:
            return None
        timestamp = time.monotonic() if now is None else now
        if timestamp - self.latest_result.captured_at > max(0.0, max_age_seconds):
            return None
        return self.latest_result

    def close(self) -> None:
        process = self._process
        if process is not None and process.is_alive():
            try:
                if self._request_queue is not None:
                    while True:
                        self._request_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._request_queue.put_nowait(("stop",))
            except (AttributeError, queue.Full):
                pass
            process.join(timeout=1.5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=0.8)

        for channel in (self._request_queue, self._result_queue):
            if channel is not None:
                channel.close()
                channel.cancel_join_thread()
        self._process = None
        self._request_queue = None
        self._result_queue = None
        self._pending_request_id = None
        self.status = HeavyAssistantStatus("off", "RTMPose detenido")


def _heavy_worker_main(detector_path: str, pose_path: str, request_queue, result_queue) -> None:
    provider = "ONNX CPU"
    try:
        from rtmlib import Hand

        model = Hand(
            det=detector_path,
            pose=pose_path,
            mode="lightweight",
            to_openpose=False,
            backend="onnxruntime",
            device="cpu",
        )
        result_queue.put(("status", "ready", "RTMPose listo", provider))
    except Exception as exc:
        result_queue.put(("status", "error", f"RTMPose no cargo: {exc}", provider))
        return

    while True:
        request = request_queue.get()
        if not request or request[0] == "stop":
            return
        _, request_id, captured_at, encoded, _reason = request
        started_at = time.perf_counter()
        completed_at = time.monotonic()
        try:
            image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError("frame JPEG invalido")
            keypoints, scores = model(image)
            hands = normalize_rtmlib_output(keypoints, scores, image.shape[1], image.shape[0])
            completed_at = time.monotonic()
            inference_ms = (time.perf_counter() - started_at) * 1000.0
            result_queue.put(
                (
                    "result",
                    request_id,
                    captured_at,
                    completed_at,
                    inference_ms,
                    hands,
                    provider,
                    "",
                )
            )
        except Exception as exc:
            completed_at = time.monotonic()
            result_queue.put(
                (
                    "result",
                    request_id,
                    captured_at,
                    completed_at,
                    (time.perf_counter() - started_at) * 1000.0,
                    (),
                    provider,
                    str(exc),
                )
            )
