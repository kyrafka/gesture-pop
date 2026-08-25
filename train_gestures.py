from __future__ import annotations

import csv
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app_config import AppConfig, load_config, load_gesture_map
from gesture_features import FeatureResult, LandmarkFeatureExtractor, draw_landmarks, summarize_vector
from gesture_runtime import FeatureStabilityTracker


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
CAPTURE_DIR = DATA_DIR / "captures"
SAMPLES_FILE = DATA_DIR / "gesture_samples.csv"
MANIFEST_FILE = DATA_DIR / "capture_manifest.csv"
MODEL_FILE = MODEL_DIR / "gesture_model.joblib"


@dataclass(frozen=True)
class TrainingSampleRecord:
    csv_row_index: int
    label: str
    ordinal: int
    sample_id: str | None
    frame_path: Path | None
    captured_at: str
    source: str


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    CAPTURE_DIR.mkdir(exist_ok=True)

    try:
        config = load_config()
        gesture_map = load_gesture_map()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    labels = list(gesture_map)
    if not labels:
        raise SystemExit("No hay imagenes validas en la carpeta imagenes.")

    try:
        extractor = LandmarkFeatureExtractor()
    except RuntimeError as exc:
        raise SystemExit(
            f"{exc}\n\n"
            "Mapeo actual de teclas:\n"
            f"{format_label_legend(labels)}\n\n"
            "Coloca tambien estos modelos en models/ si usas mediapipe.tasks:\n"
            "- hand_landmarker.task\n"
            "- face_landmarker.task (opcional pero recomendado)"
        ) from exc

    cap = cv2.VideoCapture(config.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        extractor.close()
        raise SystemExit(
            f"No pude abrir la camara {config.camera_index}. Cambia camera_index en app_config.json."
        )

    selected = 0
    show_vectors = True
    sample_counts = load_sample_counts(labels)
    stability = FeatureStabilityTracker(
        config.capture_stability_frames,
        config.capture_stability_threshold,
    )
    last_capture_time = float("-inf")
    notice = "Selecciona un gesto, mantenlo quieto y pulsa c."
    notice_until = time.monotonic() + 4.0
    preview = None
    preview_until = 0.0

    print("Entrenador listo. 1-9 etiqueta, c captura, u deshace, s entrena, q sale.")
    print("Mapeo actual:")
    print(format_label_legend(labels))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            result = extractor.extract(frame)
            label = labels[selected]
            hand_ready = is_hand_ready(result)
            stable, movement = stability.update(result.vector if hand_ready and result else None)
            ready = hand_ready and stable
            now = time.monotonic()

            if show_vectors:
                draw_landmarks(frame, result)

            guidance = build_guidance(result, label, sample_counts, stable, movement, config)
            visible_notice = notice if now <= notice_until else ""
            draw_training_ui(
                frame,
                result,
                label,
                labels,
                sample_counts,
                ready,
                stability.sample_count,
                config,
                guidance,
                visible_notice,
                show_vectors,
            )
            cv2.imshow("Entrenar gestos", frame)

            if preview is not None and now <= preview_until:
                cv2.imshow("Ultima captura (u para deshacer)", preview)
            elif preview is not None:
                _close_window("Ultima captura (u para deshacer)")
                preview = None

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if ord("1") <= key <= ord("9"):
                idx = key - ord("1")
                if idx < len(labels):
                    selected = idx
                    stability.reset()
                    notice = f"Etiqueta activa: {labels[selected]}"
                    notice_until = now + 2.0
            elif key == ord("c"):
                if result is None or not hand_ready:
                    notice = "Captura rechazada: necesito ver al menos una mano."
                    notice_until = now + 2.5
                elif not stable:
                    notice = "Captura rechazada: manten el gesto quieto un momento."
                    notice_until = now + 2.5
                elif now - last_capture_time < config.capture_min_interval_seconds:
                    notice = "Espera un instante antes de la siguiente captura."
                    notice_until = now + 1.5
                else:
                    sample_id = create_sample_id()
                    append_sample(label, result.vector)
                    saved_path = save_capture_frame(sample_id, label, frame, config)
                    append_manifest(sample_id, label, saved_path)
                    sample_counts[label] = sample_counts.get(label, 0) + 1
                    last_capture_time = now
                    preview = make_preview(frame, label, sample_counts[label])
                    preview_until = now + 3.0
                    notice = f"Captura guardada: {label} ({sample_counts[label]})"
                    notice_until = now + 2.5
                    print(notice)
            elif key == ord("u"):
                removed, removed_path = remove_last_sample(label)
                if removed:
                    sample_counts[label] = max(0, sample_counts.get(label, 0) - 1)
                    preview = None
                    _close_window("Ultima captura (u para deshacer)")
                    detail = f" y {removed_path.name}" if removed_path else ""
                    notice = f"Ultima muestra de {label} eliminada{detail}."
                    print(notice)
                else:
                    notice = f"No hay muestras de {label} para eliminar."
                notice_until = now + 2.5
            elif key == ord("s"):
                message = train_model(labels, config)
                notice = message
                notice_until = now + 5.0
            elif key == ord("v"):
                show_vectors = not show_vectors
    finally:
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()


def append_sample(label: str, vector: np.ndarray) -> None:
    is_new = not SAMPLES_FILE.exists()
    with SAMPLES_FILE.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if is_new:
            writer.writerow(["label", *[f"f{i}" for i in range(len(vector))]])
        writer.writerow([label, *vector.tolist()])


def append_manifest(sample_id: str, label: str, frame_path: Path | None) -> None:
    is_new = not MANIFEST_FILE.exists()
    relative_path = frame_path.relative_to(ROOT).as_posix() if frame_path else ""
    with MANIFEST_FILE.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if is_new:
            writer.writerow(["sample_id", "label", "frame_path", "captured_at"])
        writer.writerow([sample_id, label, relative_path, datetime.now().isoformat(timespec="seconds")])


def create_sample_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def save_capture_frame(
    sample_id: str,
    label: str,
    frame: np.ndarray,
    config: AppConfig,
) -> Path | None:
    if not config.save_capture_frames:
        return None

    folder_name = re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_") or "gesto"
    label_dir = CAPTURE_DIR / folder_name
    label_dir.mkdir(parents=True, exist_ok=True)
    path = label_dir / f"{sample_id}.jpg"
    if not cv2.imwrite(str(path), frame):
        print(f"Aviso: no pude guardar la foto de revision en {path}.")
        return None
    return path


def remove_last_sample(label: str) -> tuple[bool, Path | None]:
    removed, removed_path, _sample_id = remove_last_sample_with_id(label)
    return removed, removed_path


def remove_last_sample_with_id(label: str) -> tuple[bool, Path | None, str | None]:
    if not SAMPLES_FILE.exists():
        return False, None, None

    with SAMPLES_FILE.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    remove_index = next(
        (index for index in range(len(rows) - 1, 0, -1) if rows[index] and rows[index][0] == label),
        None,
    )
    if remove_index is None:
        return False, None, None

    rows.pop(remove_index)
    with SAMPLES_FILE.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)

    removed_path, sample_id = _remove_last_manifest_entry(label)
    return True, removed_path, sample_id


def _remove_last_manifest_entry(label: str) -> tuple[Path | None, str | None]:
    if not MANIFEST_FILE.exists():
        return None, None

    with MANIFEST_FILE.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    remove_index = next(
        (index for index in range(len(rows) - 1, 0, -1) if len(rows[index]) >= 2 and rows[index][1] == label),
        None,
    )
    if remove_index is None:
        return None, None

    row = rows.pop(remove_index)
    with MANIFEST_FILE.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)

    sample_id = row[0] if row else None
    if len(row) < 3 or not row[2]:
        return None, sample_id
    path = (ROOT / row[2]).resolve()
    capture_root = CAPTURE_DIR.resolve()
    if path.is_relative_to(capture_root) and path.is_file():
        path.unlink()
        return path, sample_id
    return None, sample_id


def format_label_legend(labels: list[str]) -> str:
    return "\n".join(f"{index}. {label}" for index, label in enumerate(labels, start=1))


def load_sample_counts(labels: list[str]) -> dict[str, int]:
    counts = {label: 0 for label in labels}
    if not SAMPLES_FILE.exists():
        return counts

    with SAMPLES_FILE.open("r", newline="", encoding="utf-8") as fh:
        rows = csv.reader(fh)
        next(rows, None)
        for row in rows:
            if row and row[0] in counts:
                counts[row[0]] += 1
    return counts


def load_sample_records(label: str | None = None) -> list[TrainingSampleRecord]:
    if not SAMPLES_FILE.exists():
        return []

    with SAMPLES_FILE.open("r", newline="", encoding="utf-8") as fh:
        sample_rows = list(csv.reader(fh))
    if len(sample_rows) < 2:
        return []

    manifest_by_label: dict[str, list[dict[str, str]]] = {}
    if MANIFEST_FILE.exists():
        with MANIFEST_FILE.open("r", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                row_label = row.get("label", "")
                if row_label:
                    manifest_by_label.setdefault(row_label, []).append(row)

    sample_indices_by_label: dict[str, list[int]] = {}
    for csv_index, row in enumerate(sample_rows[1:], start=1):
        if not row:
            continue
        row_label = row[0]
        if label is None or row_label == label:
            sample_indices_by_label.setdefault(row_label, []).append(csv_index)

    manifest_for_index: dict[int, dict[str, str]] = {}
    for row_label, csv_indices in sample_indices_by_label.items():
        manifests = manifest_by_label.get(row_label, [])
        matched_count = min(len(csv_indices), len(manifests))
        if not matched_count:
            continue
        for csv_index, manifest in zip(csv_indices[-matched_count:], manifests[-matched_count:]):
            manifest_for_index[csv_index] = manifest

    records: list[TrainingSampleRecord] = []
    for row_label, csv_indices in sample_indices_by_label.items():
        for ordinal, csv_index in enumerate(csv_indices, start=1):
            manifest = manifest_for_index.get(csv_index)
            frame_path = _resolve_manifest_frame(manifest.get("frame_path", "")) if manifest else None
            sample_id = (manifest.get("sample_id") or None) if manifest else None
            source = "camera" if frame_path else "manifest_only" if sample_id else "vector_only"
            records.append(
                TrainingSampleRecord(
                    csv_row_index=csv_index,
                    label=row_label,
                    ordinal=ordinal,
                    sample_id=sample_id,
                    frame_path=frame_path,
                    captured_at=manifest.get("captured_at", "") if manifest else "",
                    source=source,
                )
            )
    records.sort(key=lambda record: record.csv_row_index)
    return records


def remove_sample_record(record: TrainingSampleRecord) -> tuple[bool, Path | None, str | None]:
    if not SAMPLES_FILE.exists():
        return False, None, None
    with SAMPLES_FILE.open("r", newline="", encoding="utf-8") as fh:
        sample_rows = list(csv.reader(fh))
    if (
        record.csv_row_index < 1
        or record.csv_row_index >= len(sample_rows)
        or not sample_rows[record.csv_row_index]
        or sample_rows[record.csv_row_index][0] != record.label
    ):
        return False, None, None

    sample_rows.pop(record.csv_row_index)
    with SAMPLES_FILE.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(sample_rows)

    removed_path: Path | None = None
    if record.sample_id and MANIFEST_FILE.exists():
        with MANIFEST_FILE.open("r", newline="", encoding="utf-8") as fh:
            manifest_rows = list(csv.reader(fh))
        manifest_index = next(
            (
                index
                for index in range(1, len(manifest_rows))
                if len(manifest_rows[index]) >= 2
                and manifest_rows[index][0] == record.sample_id
                and manifest_rows[index][1] == record.label
            ),
            None,
        )
        if manifest_index is not None:
            manifest_rows.pop(manifest_index)
            with MANIFEST_FILE.open("w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerows(manifest_rows)

    if record.frame_path is not None:
        path = record.frame_path.resolve()
        if path.is_relative_to(CAPTURE_DIR.resolve()) and path.is_file():
            path.unlink()
            removed_path = path
    return True, removed_path, record.sample_id


def _resolve_manifest_frame(value: str) -> Path | None:
    if not value:
        return None
    raw = Path(value)
    path = raw if raw.is_absolute() else ROOT / raw
    path = path.resolve()
    if path.is_relative_to(CAPTURE_DIR.resolve()) and path.is_file():
        return path
    return None


def is_hand_ready(result: FeatureResult | None) -> bool:
    return result is not None and bool(result.hands)


def build_guidance(
    result: FeatureResult | None,
    selected_label: str,
    sample_counts: dict[str, int],
    stable: bool,
    movement: float,
    config: AppConfig,
) -> list[str]:
    remaining = max(config.target_samples_per_gesture - sample_counts.get(selected_label, 0), 0)
    if result is None or not result.hands:
        return [
            "Entra en cuadro con al menos una mano visible.",
            f"Faltan {remaining} muestras para llegar al objetivo.",
        ]

    if not stable:
        movement_text = "calculando" if not np.isfinite(movement) else f"movimiento {movement:.3f}"
        return [
            f"Manten el gesto quieto ({movement_text}).",
            f"Faltan {remaining} muestras; cambia un poco angulo, distancia y luz.",
        ]

    face_text = "Cara detectada." if result.faces else "Sin cara; esta bien para gestos solo de mano."
    return [
        "Gesto estable: pulsa c para capturar.",
        f"{face_text} Faltan {remaining} muestras para el objetivo.",
    ]


def draw_training_ui(
    frame: np.ndarray,
    result: FeatureResult | None,
    selected_label: str,
    labels: list[str],
    sample_counts: dict[str, int],
    ready: bool,
    stability_count: int,
    config: AppConfig,
    guidance: list[str],
    notice: str,
    show_vectors: bool,
) -> None:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, min(190, height)), (20, 20, 20), -1)
    cv2.rectangle(overlay, (0, max(0, height - 96)), (width, height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    status_color = (50, 220, 80) if ready else (40, 180, 255)
    status = "LISTO PARA CAPTURAR" if ready else "MANTEN EL GESTO QUIETO"
    cv2.circle(frame, (22, 25), 9, status_color, -1)
    put_fitted(frame, status, (40, 32), width * 0.56, 0.64, status_color, 2)
    put_fitted(frame, f"Etiqueta: {selected_label}", (20, 65), width * 0.58, 0.72, (0, 255, 255), 2)

    debug = result.debug if result else "sin landmarks"
    put_fitted(frame, debug, (20, 94), width * 0.58, 0.52, (180, 255, 180), 1)
    stability_text = f"Estabilidad: {min(stability_count, config.capture_stability_frames)}/{config.capture_stability_frames}"
    put_fitted(frame, stability_text, (20, 120), width * 0.58, 0.5, (220, 220, 220), 1)

    if show_vectors:
        for index, line in enumerate(summarize_vector(result)[:2]):
            put_fitted(frame, line, (20, 146 + index * 24), width * 0.58, 0.48, (160, 230, 255), 1)

    right_x = max(int(width * 0.62), width - 340)
    put_fitted(frame, f"Muestras / objetivo {config.target_samples_per_gesture}", (right_x, 28), width - right_x - 10, 0.5, (255, 255, 255), 1)
    for index, label in enumerate(labels[:6]):
        marker = ">" if label == selected_label else " "
        count = sample_counts.get(label, 0)
        color = (255, 220, 120) if label == selected_label else (220, 220, 220)
        put_fitted(frame, f"{marker} {index + 1}. {label}: {count}", (right_x, 54 + index * 22), width - right_x - 10, 0.46, color, 1)

    base_y = max(210, height - 72)
    for index, line in enumerate(guidance[:2]):
        put_fitted(frame, line, (20, base_y + index * 23), width - 30, 0.5, (220, 255, 220), 1)
    if notice:
        put_fitted(frame, notice, (20, height - 25), width - 30, 0.52, (120, 220, 255), 2)
    else:
        controls = "1-9 cambia | c captura | u deshace | s entrena | v vectores | q sale"
        put_fitted(frame, controls, (20, height - 25), width - 30, 0.48, (255, 255, 255), 1)


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


def make_preview(frame: np.ndarray, label: str, count: int) -> np.ndarray:
    preview = frame.copy()
    height, width = preview.shape[:2]
    target_width = min(640, width)
    scale = target_width / width
    preview = cv2.resize(preview, (target_width, int(height * scale)), interpolation=cv2.INTER_AREA)
    cv2.rectangle(preview, (0, 0), (preview.shape[1], 45), (20, 20, 20), -1)
    cv2.putText(
        preview,
        f"{label} - muestra {count}",
        (15, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return preview


def _close_window(name: str) -> None:
    try:
        cv2.destroyWindow(name)
    except cv2.error:
        pass


def train_model(expected_labels: list[str], config: AppConfig) -> str:
    if not SAMPLES_FILE.exists():
        message = "Aun no hay muestras."
        print(message)
        return message

    with SAMPLES_FILE.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    data = [row for row in rows[1:] if row and row[0] in expected_labels]
    if len(data) < 6:
        message = "Captura mas muestras antes de entrenar."
        print(message)
        return message

    widths = {len(row) for row in data}
    if len(widths) != 1:
        message = "El CSV mezcla vectores de tamanos distintos; revisa data/gesture_samples.csv."
        print(message)
        return message

    y = [row[0] for row in data]
    x = np.array([[float(value) for value in row[1:]] for row in data], dtype=np.float32)
    counts = {label: y.count(label) for label in expected_labels}
    min_count = min(counts.values()) if counts else 0
    if min_count < 3:
        missing = {label: 3 - count for label, count in counts.items() if count < 3}
        ready = {label: count for label, count in counts.items() if count >= 3}
        message = (
            f"Necesito 3 muestras en cada gesto. Faltan por capturar: {missing}. "
            f"Gestos que ya cumplen: {ready or 'ninguno'}."
        )
        print(message)
        return message

    neighbors = max(1, min(5, min_count - 1))
    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=neighbors, weights="distance"),
    )

    validation_accuracy = None
    if min_count >= 4:
        folds = min(5, min_count)
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        try:
            scores = cross_val_score(model, x, y, cv=splitter, scoring="accuracy")
            validation_accuracy = float(scores.mean())
        except ValueError as exc:
            print(f"No pude calcular validacion cruzada: {exc}")

    model.fit(x, y)
    payload = {
        "model": model,
        "labels": expected_labels,
        "feature_count": int(x.shape[1]),
        "sample_counts": counts,
        "validation_accuracy": validation_accuracy,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    joblib.dump(payload, MODEL_FILE)

    score_text = f" Precision estimada: {validation_accuracy:.0%}." if validation_accuracy is not None else ""
    target_warning = ""
    if min_count < config.target_samples_per_gesture:
        target_warning = f" Recomendado: {config.target_samples_per_gesture} por gesto."
    message = f"Modelo guardado.{score_text}{target_warning}"
    print(f"{message} Conteo: {counts}")
    return message


if __name__ == "__main__":
    main()
