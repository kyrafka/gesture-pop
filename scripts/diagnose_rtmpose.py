from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from heavy_hand_backend import HeavyHandAssistant  # noqa: E402


def find_test_image() -> Path:
    captures = ROOT / "data" / "captures"
    for suffix in ("*.jpg", "*.jpeg", "*.png"):
        match = next(captures.rglob(suffix), None)
        if match is not None:
            return match
    raise RuntimeError("No hay una captura en data/captures para probar RTMPose.")


def main() -> int:
    image_path = find_test_image()
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"No pude leer {image_path}")

    assistant = HeavyHandAssistant(enabled=True)
    try:
        status = assistant.start()
        print(status.message)
        if status.state in {"error", "unavailable"}:
            return 1
        if not assistant.submit(frame, "diagnostico"):
            raise RuntimeError("El proceso no acepto el frame de prueba.")

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            result, status = assistant.poll()
            if status is not None:
                print(status.message)
            if result is not None:
                if result.error:
                    return 1
                print(
                    f"OK: {len(result.hands)} mano(s), "
                    f"{result.inference_ms:.0f} ms, {result.provider}"
                )
                return 0
            time.sleep(0.04)
        raise RuntimeError("RTMPose no respondio en 10 segundos.")
    finally:
        assistant.close()


if __name__ == "__main__":
    raise SystemExit(main())
