from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gesture_launcher import open_image_file


class OpenImageTests(unittest.TestCase):
    def test_opens_existing_image_once_on_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "gesture.png"
            image.write_bytes(b"image")
            with (
                patch("gesture_launcher.sys.platform", "win32"),
                patch.object(os, "startfile", create=True) as startfile,
            ):
                opened = open_image_file(image)

            self.assertTrue(opened)
            startfile.assert_called_once_with(str(image.resolve()))

    def test_missing_image_is_not_opened(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "missing.png"
            with (
                patch("gesture_launcher.sys.platform", "win32"),
                patch.object(os, "startfile", create=True) as startfile,
            ):
                opened = open_image_file(image)

            self.assertFalse(opened)
            startfile.assert_not_called()


if __name__ == "__main__":
    unittest.main()
