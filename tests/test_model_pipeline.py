from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import joblib
import numpy as np

from app_config import AppConfig
from gesture_launcher import overlay_center
from gesture_runtime import TemporalGestureDecider
import train_gestures


class ModelPipelineTests(unittest.TestCase):
    def test_training_prediction_temporal_trigger_and_overlay(self) -> None:
        labels = ["one", "two", "three", "four", "five", "six"]
        rng = np.random.default_rng(42)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = root / "samples.csv"
            model_file = root / "model.joblib"
            with samples.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["label", *[f"f{i}" for i in range(194)]])
                for label_index, label in enumerate(labels):
                    center = np.full(194, label_index * 3.0, dtype=np.float32)
                    for _sample in range(5):
                        writer.writerow([label, *(center + rng.normal(0, 0.03, 194)).tolist()])

            with (
                patch.object(train_gestures, "SAMPLES_FILE", samples),
                patch.object(train_gestures, "MODEL_FILE", model_file),
            ):
                message = train_gestures.train_model(labels, AppConfig())

            self.assertTrue(message.startswith("Modelo guardado."))
            payload = joblib.load(model_file)
            self.assertEqual(payload["feature_count"], 194)
            self.assertEqual(payload["labels"], labels)
            self.assertEqual(payload["tracking_profile"], "equilibrado")

            query = np.full(194, 6.0, dtype=np.float32)
            probabilities = payload["model"].predict_proba([query])[0]
            classes = payload["model"].classes_
            self.assertEqual(str(classes[int(probabilities.argmax())]), "three")

            decider = TemporalGestureDecider(0.6, 0.2, 5, 3, 2, 0.0)
            state = None
            for frame_index in range(3):
                state = decider.observe(probabilities, classes, float(frame_index))
            self.assertEqual(state.triggered_label, "three")

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            image = np.zeros((120, 160, 4), dtype=np.uint8)
            image[:, :, :3] = (30, 180, 240)
            image[:, :, 3] = 255
            output = overlay_center(frame, image)
            self.assertGreater(int(output.sum()), 0)


if __name__ == "__main__":
    unittest.main()
