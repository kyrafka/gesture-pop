from __future__ import annotations

from dataclasses import dataclass

from gesture_features import HandPose


@dataclass(frozen=True)
class CaptureTarget:
    key: str
    title: str
    instruction: str
    bounds: tuple[float, float, float, float]

    def matches(self, pose: HandPose | None) -> bool:
        if pose is None:
            return False
        x1, y1, x2, y2 = self.bounds
        return x1 <= pose.center_x <= x2 and y1 <= pose.center_y <= y2


BASE_TARGETS = (
    CaptureTarget("center", "CENTRO", "Coloca la mano al centro", (0.34, 0.25, 0.66, 0.75)),
    CaptureTarget("left", "IZQUIERDA", "Mueve el mismo gesto a la izquierda", (0.08, 0.22, 0.40, 0.78)),
    CaptureTarget("right", "DERECHA", "Mueve el mismo gesto a la derecha", (0.60, 0.22, 0.92, 0.78)),
    CaptureTarget("upper", "ARRIBA", "Sube el mismo gesto", (0.25, 0.06, 0.75, 0.42)),
    CaptureTarget("lower", "ABAJO", "Baja el mismo gesto", (0.25, 0.58, 0.75, 0.92)),
    CaptureTarget("center-wide", "CENTRO", "Vuelve al centro con una variacion natural", (0.28, 0.20, 0.72, 0.80)),
)


def build_capture_targets(total: int) -> list[CaptureTarget]:
    if total < 1:
        return []
    return [BASE_TARGETS[index % len(BASE_TARGETS)] for index in range(total)]
