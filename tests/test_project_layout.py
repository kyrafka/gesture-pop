from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"


class ProjectLayoutTests(unittest.TestCase):
    def test_launchers_are_grouped_and_point_to_existing_scripts(self) -> None:
        launchers = {
            "run_studio.bat": "gesture_studio_qt.py",
            "run_recognition.bat": "gesture_launcher.py",
        }
        for filename, script in launchers.items():
            launcher = SCRIPTS_DIR / filename
            self.assertTrue(launcher.is_file())
            contents = launcher.read_text(encoding="utf-8")
            self.assertIn(r"%~dp0..", contents)
            self.assertIn(script, contents)
            self.assertTrue((ROOT / script).is_file())

    def test_pose_tools_are_named_and_wired_under_scripts(self) -> None:
        installer = SCRIPTS_DIR / "install_rtmpose.bat"
        diagnostic = SCRIPTS_DIR / "diagnose_rtmpose.py"
        setup = SCRIPTS_DIR / "setup_rtmpose.py"

        self.assertTrue(diagnostic.is_file())
        self.assertTrue(setup.is_file())
        self.assertIn(r"scripts\setup_rtmpose.py", installer.read_text(encoding="utf-8"))

    def test_root_has_no_legacy_launchers_or_tkinter_studio(self) -> None:
        self.assertFalse((ROOT / "gesture_studio.py").exists())
        self.assertFalse((ROOT / "setup_heavy_assist.py").exists())
        self.assertFalse((ROOT / "NO_TOCAR").exists())
        self.assertFalse(any(ROOT.glob("iniciar_*.bat")))


if __name__ == "__main__":
    unittest.main()
