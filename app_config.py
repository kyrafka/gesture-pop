from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path


ROOT = Path(__file__).parent
CONFIG_FILE = ROOT / "app_config.json"
GESTURE_MAP_FILE = ROOT / "gesture_map.json"
IMAGE_DIR = ROOT / "imagenes"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class AppConfig:
    camera_index: int = 0
    target_samples_per_gesture: int = 20
    save_capture_frames: bool = True
    capture_min_interval_seconds: float = 0.35
    capture_stability_frames: int = 6
    capture_stability_threshold: float = 0.12
    heavy_hand_assist: bool = True
    heavy_hand_interval_seconds: float = 0.32
    heavy_hand_idle_interval_seconds: float = 1.5
    heavy_hand_stale_seconds: float = 0.55
    confidence_threshold: float = 0.68
    confidence_margin: float = 0.16
    prediction_window: int = 10
    stability_frames: int = 7
    release_frames: int = 5
    cooldown_seconds: float = 1.2
    overlay_seconds: float = 1.8


def load_config(path: Path = CONFIG_FILE) -> AppConfig:
    if not path.exists():
        return AppConfig()

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"No pude leer {path.name}: {exc}") from exc

    allowed = {field.name for field in fields(AppConfig)}
    values = {key: value for key, value in raw.items() if key in allowed}
    try:
        config = AppConfig(**values)
    except TypeError as exc:
        raise RuntimeError(f"Configuracion invalida en {path.name}: {exc}") from exc

    _validate_config(config)
    return config


def load_gesture_map() -> dict[str, Path]:
    discovered = {
        path.stem: path
        for path in sorted(IMAGE_DIR.iterdir())
        if path.suffix.lower() in IMAGE_SUFFIXES
    }
    if not GESTURE_MAP_FILE.exists():
        return discovered

    try:
        raw = json.loads(GESTURE_MAP_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"No pude leer {GESTURE_MAP_FILE.name}: {exc}") from exc

    if not isinstance(raw, dict):
        raise RuntimeError(f"{GESTURE_MAP_FILE.name} debe contener un objeto etiqueta -> archivo.")

    mapping: dict[str, Path] = {}
    referenced: set[Path] = set()
    for label, filename in raw.items():
        if not isinstance(label, str) or not isinstance(filename, str):
            raise RuntimeError(f"Cada entrada de {GESTURE_MAP_FILE.name} debe usar textos.")
        path = IMAGE_DIR / filename
        if path.exists() and path.suffix.lower() in IMAGE_SUFFIXES:
            mapping[label] = path
            referenced.add(path.resolve())

    for label, path in discovered.items():
        if path.resolve() not in referenced and label not in mapping:
            mapping[label] = path
    return mapping


def save_gesture_map(mapping: dict[str, Path]) -> None:
    payload = {label: path.name for label, path in mapping.items()}
    GESTURE_MAP_FILE.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _validate_config(config: AppConfig) -> None:
    if not isinstance(config.heavy_hand_assist, bool):
        raise RuntimeError("heavy_hand_assist debe ser true o false.")

    positive_ints = {
        "target_samples_per_gesture": config.target_samples_per_gesture,
        "capture_stability_frames": config.capture_stability_frames,
        "prediction_window": config.prediction_window,
        "stability_frames": config.stability_frames,
        "release_frames": config.release_frames,
    }
    for name, value in positive_ints.items():
        if not isinstance(value, int) or value < 1:
            raise RuntimeError(f"{name} debe ser un entero mayor que cero.")

    if config.stability_frames > config.prediction_window:
        raise RuntimeError("stability_frames no puede superar prediction_window.")

    probabilities = {
        "confidence_threshold": config.confidence_threshold,
        "confidence_margin": config.confidence_margin,
    }
    for name, value in probabilities.items():
        if not 0.0 <= float(value) <= 1.0:
            raise RuntimeError(f"{name} debe estar entre 0 y 1.")

    non_negative = {
        "capture_min_interval_seconds": config.capture_min_interval_seconds,
        "capture_stability_threshold": config.capture_stability_threshold,
        "heavy_hand_interval_seconds": config.heavy_hand_interval_seconds,
        "heavy_hand_idle_interval_seconds": config.heavy_hand_idle_interval_seconds,
        "heavy_hand_stale_seconds": config.heavy_hand_stale_seconds,
        "cooldown_seconds": config.cooldown_seconds,
        "overlay_seconds": config.overlay_seconds,
    }
    for name, value in non_negative.items():
        if float(value) < 0.0:
            raise RuntimeError(f"{name} no puede ser negativo.")
