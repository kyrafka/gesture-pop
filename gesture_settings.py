from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parent
GESTURE_SETTINGS_FILE = ROOT / "gesture_settings.json"
VALID_EXPECTED_HANDS = {1, 2}


def load_gesture_settings(labels: list[str] | None = None) -> dict[str, dict[str, int]]:
    settings: dict[str, dict[str, int]] = {}
    if GESTURE_SETTINGS_FILE.exists():
        try:
            raw = json.loads(GESTURE_SETTINGS_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
        if isinstance(raw, dict):
            for label, value in raw.items():
                if isinstance(label, str) and isinstance(value, dict):
                    expected = value.get("expected_hands", 1)
                    if expected in VALID_EXPECTED_HANDS:
                        settings[label] = {"expected_hands": int(expected)}

    for label in labels or []:
        settings.setdefault(label, {"expected_hands": 1})
    return settings


def save_gesture_settings(settings: dict[str, dict[str, int]]) -> None:
    payload = {
        label: {"expected_hands": int(value.get("expected_hands", 1))}
        for label, value in sorted(settings.items())
        if int(value.get("expected_hands", 1)) in VALID_EXPECTED_HANDS
    }
    GESTURE_SETTINGS_FILE.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def expected_hands_for(settings: dict[str, dict[str, int]], label: str | None) -> int:
    if not label:
        return 1
    value = settings.get(label, {}).get("expected_hands", 1)
    return int(value) if value in VALID_EXPECTED_HANDS else 1
