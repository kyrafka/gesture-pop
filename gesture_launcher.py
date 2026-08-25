from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np

from app_config import AppConfig, load_config, load_gesture_map
from gesture_features import LandmarkFeatureExtractor, draw_landmarks
from gesture_runtime import PredictionState, TemporalGestureDecider


ROOT = Path(__file__).parent
MODEL_FILE = ROOT / "models" / "gesture_model.joblib"


def main() -> None:
    if not MODEL_FILE.exists():
        raise SystemExit("No existe models/gesture_model.joblib. Entrena primero y pulsa s.")

    try:
        config = load_config()
        gesture_map = load_gesture_map()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    payload = joblib.load(MODEL_FILE)
    model = payload["model"]
    images = load_images(gesture_map)
    if not images:
        raise SystemExit("No pude cargar ninguna imagen configurada en gesture_map.json.")

    decider = TemporalGestureDecider(
        confidence_threshold=config.confidence_threshold,
        confidence_margin=config.confidence_margin,
        window=config.prediction_window,
        required_votes=config.stability_frames,
        release_frames=config.release_frames,
        cooldown_seconds=config.cooldown_seconds,
    )

    try:
        extractor = LandmarkFeatureExtractor()
    except RuntimeError as exc:
        raise SystemExit(
            f"{exc}\n\n"
            "Coloca models/hand_landmarker.task y, si quieres usar cara tambien, models/face_landmarker.task."
        ) from exc

    cap = cv2.VideoCapture(config.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        extractor.close()
        raise SystemExit(
            f"No pude abrir la camara {config.camera_index}. Cambia camera_index en app_config.json."
        )

    active_label: str | None = None
    active_until = 0.0
    show_vectors = True

    print("Lanzador listo. Manten un gesto estable; v alterna vectores y q sale.")
    validation = payload.get("validation_accuracy")
    if validation is not None:
        print(f"Precision estimada del modelo al entrenar: {validation:.0%}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            result = extractor.extract(frame)
            now = time.monotonic()
            probabilities = None
            classes = None

            if result is not None and result.hands:
                expected_features = payload.get("feature_count")
                if expected_features is not None and len(result.vector) != expected_features:
                    raise RuntimeError(
                        "El modelo usa otro formato de vectores. Vuelve a entrenarlo con train_gestures.py."
                    )
                probabilities = model.predict_proba([result.vector])[0]
                classes = model.classes_

            state = decider.observe(probabilities, classes, now)
            if state.triggered_label is not None:
                active_label = state.triggered_label
                active_until = now + config.overlay_seconds
                image_path = gesture_map.get(active_label)
                opened = bool(image_path and open_image_file(image_path))
                print(
                    f"Disparo: {active_label} "
                    f"({state.stable_confidence:.0%}, {state.stable_votes} votos) | "
                    f"archivo={'abierto' if opened else 'no disponible'}"
                )

            if active_label and now <= active_until and active_label in images:
                frame = overlay_center(frame, images[active_label])

            if show_vectors:
                draw_landmarks(frame, result)
            draw_launcher_ui(frame, state, probabilities, classes, active_label, active_until, now, config)
            cv2.imshow("Lanzador de imagenes por gesto", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("v"):
                show_vectors = not show_vectors
    finally:
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()


def load_images(gesture_map: dict[str, Path]) -> dict[str, np.ndarray]:
    images: dict[str, np.ndarray] = {}
    for label, path in gesture_map.items():
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is not None:
            images[label] = image
        else:
            print(f"Aviso: no pude abrir la imagen de {label}: {path.name}")
    return images


def open_image_file(path: Path) -> bool:
    image_path = path.resolve()
    if not image_path.is_file():
        print(f"Aviso: no existe la imagen que se intento abrir: {image_path}")
        return False
    try:
        if sys.platform == "win32":
            startfile = getattr(os, "startfile")
            startfile(str(image_path))
        elif sys.platform == "darwin":
            subprocess.Popen(
                ["open", str(image_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.Popen(
                ["xdg-open", str(image_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except (AttributeError, OSError) as exc:
        print(f"Aviso: no pude abrir {image_path.name} con el visor del sistema: {exc}")
        return False
    return True


def draw_launcher_ui(
    frame: np.ndarray,
    state: PredictionState,
    probabilities,
    classes,
    active_label: str | None,
    active_until: float,
    now: float,
    config: AppConfig,
) -> None:
    height, width = frame.shape[:2]
    panel_width = min(430, max(300, int(width * 0.36)))
    panel_height = min(235, height)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.76, frame, 0.24, 0, frame)

    if state.stable_label:
        title = f"Confirmado: {state.stable_label}"
        color = (50, 220, 80)
    elif state.frame_label:
        title = f"Analizando: {state.frame_label}"
        color = (40, 190, 255)
    else:
        title = "Muestra una mano"
        color = (180, 180, 180)
    put_fitted(frame, title, (18, 31), panel_width - 36, 0.64, color, 2)

    votes = min(state.stable_votes, state.required_votes)
    progress = votes / state.required_votes
    cv2.rectangle(frame, (18, 47), (panel_width - 18, 63), (60, 60, 60), -1)
    cv2.rectangle(
        frame,
        (18, 47),
        (18 + int((panel_width - 36) * progress), 63),
        color,
        -1,
    )
    cv2.putText(
        frame,
        f"estabilidad {votes}/{state.required_votes}",
        (18, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    for row, (label, probability) in enumerate(top_predictions(probabilities, classes, limit=3)):
        y = 111 + row * 38
        put_fitted(frame, label, (18, y), panel_width * 0.52, 0.48, (235, 235, 235), 1)
        bar_x = int(panel_width * 0.54)
        bar_width = panel_width - bar_x - 54
        cv2.rectangle(frame, (bar_x, y - 13), (bar_x + bar_width, y), (65, 65, 65), -1)
        cv2.rectangle(frame, (bar_x, y - 13), (bar_x + int(bar_width * probability), y), (70, 200, 240), -1)
        cv2.putText(
            frame,
            f"{probability:.0%}",
            (bar_x + bar_width + 6, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

    status = "ARMADO" if state.armed else "SUELTA EL GESTO PARA REARMAR"
    status_color = (90, 230, 110) if state.armed else (80, 190, 255)
    put_fitted(frame, status, (18, panel_height - 17), panel_width - 36, 0.46, status_color, 1)

    if active_label and now <= active_until:
        put_fitted(
            frame,
            f"Imagen activa: {active_label}",
            (max(18, width - 360), height - 24),
            min(342, width - 36),
            0.58,
            (0, 255, 255),
            2,
        )
    else:
        controls = (
            f"umbral {config.confidence_threshold:.0%} | "
            f"margen {config.confidence_margin:.0%} | v vectores | q sale"
        )
        put_fitted(frame, controls, (18, height - 24), width - 36, 0.48, (255, 255, 255), 1)


def top_predictions(probabilities, classes, limit: int) -> list[tuple[str, float]]:
    if probabilities is None or classes is None:
        return []
    values = np.asarray(probabilities, dtype=np.float32)
    order = np.argsort(values)[::-1][:limit]
    return [(str(classes[index]), float(values[index])) for index in order]


def put_fitted(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    max_width: float,
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    current_scale = scale
    while current_scale > 0.34:
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, current_scale, thickness)[0][0]
        if text_width <= max_width:
            break
        current_scale -= 0.04
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, current_scale, color, thickness, cv2.LINE_AA)


def overlay_center(frame: np.ndarray, image: np.ndarray) -> np.ndarray:
    frame_height, frame_width = frame.shape[:2]
    image_height, image_width = image.shape[:2]
    scale = min(
        frame_width * 0.46 / image_width,
        frame_height * 0.58 / image_height,
        1.0,
    )
    target_width = max(1, int(image_width * scale))
    target_height = max(1, int(image_height * scale))
    resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

    x = (frame_width - target_width) // 2
    y = max(0, int(frame_height * 0.58) - target_height // 2)
    y = min(y, frame_height - target_height)
    roi = frame[y:y + target_height, x:x + target_width]

    if resized.ndim == 2:
        roi[:] = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    elif resized.shape[2] == 4:
        alpha = resized[:, :, 3:4].astype(np.float32) / 255.0
        blended = alpha * resized[:, :, :3] + (1.0 - alpha) * roi
        roi[:] = blended.astype(np.uint8)
    else:
        roi[:] = resized[:, :, :3]
    return frame


if __name__ == "__main__":
    main()
