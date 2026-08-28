from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

from heavy_hand_backend import HEAVY_MODEL_DIR, find_heavy_model_paths


MODELS = {
    "rtmdet": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "rtmdet_nano_8xb32-300e_hand-267f9c8f.zip"
    ),
    "rtmpose": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip"
    ),
}


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Gesture-Pop/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def install_models() -> tuple[Path, Path]:
    HEAVY_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    detector, pose = find_heavy_model_paths()
    if detector is not None and pose is not None:
        _remove_download_archives()
        return detector, pose

    for name, url in MODELS.items():
        archive = HEAVY_MODEL_DIR / f"{name}_hand.zip"
        target = HEAVY_MODEL_DIR / name
        if not archive.exists():
            print(f"Descargando {name}...")
            download(url, archive)
        print(f"Extrayendo {name}...")
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(target)
        archive.unlink(missing_ok=True)

    detector, pose = find_heavy_model_paths()
    if detector is None or pose is None:
        raise RuntimeError("La descarga termino, pero no encontre ambos modelos ONNX.")
    return detector, pose


def _remove_download_archives() -> None:
    for name in MODELS:
        (HEAVY_MODEL_DIR / f"{name}_hand.zip").unlink(missing_ok=True)


def main() -> None:
    detector, pose = install_models()
    print("Asistencia pesada lista:")
    print(f"- Detector: {detector}")
    print(f"- Pose: {pose}")


if __name__ == "__main__":
    main()
