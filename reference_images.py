from __future__ import annotations

import csv
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from gesture_features import FeatureResult, LandmarkFeatureExtractor, draw_landmarks


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
REFERENCE_DIR = DATA_DIR / "references"
REFERENCE_MANIFEST_FILE = DATA_DIR / "reference_manifest.csv"
REFERENCE_VECTORS_FILE = DATA_DIR / "reference_vectors.csv"


@dataclass(frozen=True)
class ReferenceQuality:
    can_accept: bool
    score: int
    messages: tuple[str, ...]


@dataclass
class ReferenceAnalysis:
    source_path: Path
    original: np.ndarray
    annotated: np.ndarray
    result: FeatureResult | None
    quality: ReferenceQuality


@dataclass(frozen=True)
class ReferenceRecord:
    reference_id: str
    label: str
    original_path: Path
    annotated_path: Path
    created_at: str
    used_for_training: bool
    quality_score: int


def analyze_reference(
    path: Path,
    extractor: LandmarkFeatureExtractor,
) -> ReferenceAnalysis:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("El archivo no contiene una imagen que OpenCV pueda analizar.")

    result = extractor.extract(image)
    annotated = image.copy()
    draw_landmarks(annotated, result)
    quality = evaluate_reference_quality(image, result)
    return ReferenceAnalysis(path, image, annotated, result, quality)


def evaluate_reference_quality(
    image: np.ndarray,
    result: FeatureResult | None,
) -> ReferenceQuality:
    if result is None or not result.hands:
        return ReferenceQuality(
            can_accept=False,
            score=0,
            messages=("No detecte ninguna mano; usa otra imagen o un recorte mas claro.",),
        )

    score = 100
    messages: list[str] = []
    height, width = image.shape[:2]
    if min(width, height) < 320:
        score -= 10
        messages.append("Resolucion baja; intenta usar al menos 320 px en el lado corto.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if sharpness < 35.0:
        score -= 15
        messages.append("La imagen parece borrosa; los dedos pueden confundirse.")

    brightness = float(gray.mean())
    if brightness < 35.0:
        score -= 12
        messages.append("La imagen esta muy oscura.")
    elif brightness > 225.0:
        score -= 12
        messages.append("La imagen esta sobreexpuesta.")

    for pose in result.hand_poses:
        if pose.width < 0.12 or pose.height < 0.12:
            score -= 18
            messages.append(f"Mano {pose.index}: ocupa poco espacio; acercala o recorta la foto.")
        x1, y1, x2, y2 = pose.bbox
        if x1 <= 0.001 or y1 <= 0.001 or x2 >= 0.999 or y2 >= 0.999:
            score -= 18
            messages.append(f"Mano {pose.index}: toca el borde y podria estar recortada.")

    if not messages:
        messages.append("Deteccion limpia: la referencia es adecuada.")
    return ReferenceQuality(True, max(0, score), tuple(dict.fromkeys(messages)))


def store_reference(
    label: str,
    analysis: ReferenceAnalysis,
    used_for_training: bool,
) -> ReferenceRecord:
    if analysis.result is None or not analysis.result.hands:
        raise ValueError("La referencia necesita al menos una mano detectada.")

    DATA_DIR.mkdir(exist_ok=True)
    folder_name = re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_") or "gesto"
    label_dir = REFERENCE_DIR / folder_name
    label_dir.mkdir(parents=True, exist_ok=True)

    reference_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = analysis.source_path.suffix.lower() or ".jpg"
    original_path = label_dir / f"{reference_id}{suffix}"
    annotated_path = label_dir / f"{reference_id}_detected.jpg"
    shutil.copy2(analysis.source_path, original_path)
    if not cv2.imwrite(str(annotated_path), analysis.annotated):
        original_path.unlink(missing_ok=True)
        raise OSError("No pude guardar la vista marcada de la referencia.")

    created_at = datetime.now().isoformat(timespec="seconds")
    record = ReferenceRecord(
        reference_id=reference_id,
        label=label,
        original_path=original_path,
        annotated_path=annotated_path,
        created_at=created_at,
        used_for_training=used_for_training,
        quality_score=analysis.quality.score,
    )
    _append_manifest(record, analysis.result)
    _append_reference_vector(record, analysis.result.vector)
    return record


def load_reference_records(label: str | None = None) -> list[ReferenceRecord]:
    if not REFERENCE_MANIFEST_FILE.exists():
        return []

    records: list[ReferenceRecord] = []
    with REFERENCE_MANIFEST_FILE.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if label is not None and row.get("label") != label:
                continue
            original_path = (ROOT / row.get("original_path", "")).resolve()
            annotated_path = (ROOT / row.get("annotated_path", "")).resolve()
            if not original_path.is_file() or not annotated_path.is_file():
                continue
            records.append(
                ReferenceRecord(
                    reference_id=row.get("reference_id", ""),
                    label=row.get("label", ""),
                    original_path=original_path,
                    annotated_path=annotated_path,
                    created_at=row.get("created_at", ""),
                    used_for_training=row.get("used_for_training", "false").lower() == "true",
                    quality_score=int(row.get("quality_score", "0") or 0),
                )
            )
    records.sort(key=lambda record: (record.created_at, record.reference_id))
    return records


def mark_reference_not_training(reference_id: str) -> bool:
    if not REFERENCE_MANIFEST_FILE.exists():
        return False
    with REFERENCE_MANIFEST_FILE.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
        fieldnames = list(rows[0].keys()) if rows else []
    changed = False
    for row in rows:
        if row.get("reference_id") == reference_id and row.get("used_for_training") == "true":
            row["used_for_training"] = "false"
            changed = True
            break
    if changed:
        with REFERENCE_MANIFEST_FILE.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return changed


def _append_manifest(record: ReferenceRecord, result: FeatureResult) -> None:
    is_new = not REFERENCE_MANIFEST_FILE.exists()
    with REFERENCE_MANIFEST_FILE.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if is_new:
            writer.writerow(
                [
                    "reference_id",
                    "label",
                    "original_path",
                    "annotated_path",
                    "created_at",
                    "used_for_training",
                    "quality_score",
                    "hand_count",
                    "face_detected",
                    "debug",
                ]
            )
        writer.writerow(
            [
                record.reference_id,
                record.label,
                record.original_path.relative_to(ROOT).as_posix(),
                record.annotated_path.relative_to(ROOT).as_posix(),
                record.created_at,
                str(record.used_for_training).lower(),
                record.quality_score,
                len(result.hands),
                str(bool(result.faces)).lower(),
                result.debug,
            ]
        )


def _append_reference_vector(record: ReferenceRecord, vector: np.ndarray) -> None:
    is_new = not REFERENCE_VECTORS_FILE.exists()
    with REFERENCE_VECTORS_FILE.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if is_new:
            writer.writerow(["reference_id", "label", *[f"f{i}" for i in range(len(vector))]])
        writer.writerow([record.reference_id, record.label, *vector.tolist()])
