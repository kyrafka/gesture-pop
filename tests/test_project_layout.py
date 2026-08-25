from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_DIR = ROOT / "NO_TOCAR"


class ProjectLayoutTests(unittest.TestCase):
    def test_launchers_are_grouped_and_point_to_existing_scripts(self) -> None:
        launchers = {
            "ABRIR_GESTURE_POP.bat": "gesture_studio_qt.py",
            "ABRIR_RECONOCIMIENTO.bat": "gesture_launcher.py",
        }
        for filename, script in launchers.items():
            launcher = LAUNCHER_DIR / filename
            self.assertTrue(launcher.is_file())
            contents = launcher.read_text(encoding="utf-8")
            self.assertIn(r"%~dp0..", contents)
            self.assertIn(script, contents)
            self.assertTrue((ROOT / script).is_file())

    def test_root_has_no_legacy_launchers_or_tkinter_studio(self) -> None:
        self.assertFalse((ROOT / "gesture_studio.py").exists())
        self.assertFalse(any(ROOT.glob("iniciar_*.bat")))


if __name__ == "__main__":
    unittest.main()
