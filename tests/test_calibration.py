from __future__ import annotations

import tempfile
import unittest

import numpy as np

from calibration import CalibrationModel, CalibrationSample, calibration_targets


class CalibrationModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(7)

    def test_fit_predict_round_trip(self) -> None:
        model = CalibrationModel()
        samples: list[CalibrationSample] = []

        for x in (-0.2, 0.0, 0.2):
            for y in (0.15, 0.35, 0.55):
                base = np.asarray(
                    [
                        x + 0.5,
                        y,
                        x + 0.48,
                        y + 0.01,
                        x + 0.52,
                        y - 0.01,
                        x * 0.1,
                        y * -0.08,
                        0.32,
                        0.44,
                        0.22,
                        0.21,
                    ],
                    dtype=np.float64,
                )
                target = (
                    int(960 + 900 * x + 120 * y),
                    int(540 - 70 * x + 600 * y),
                )
                for _ in range(8):
                    noise = self.rng.normal(0.0, 0.002, size=base.shape)
                    samples.append(CalibrationSample(target=target, feature_vector=base + noise))

        kept = model.fit(samples)
        self.assertGreaterEqual(kept, 60)

        query = np.asarray(
            [0.62, 0.44, 0.60, 0.45, 0.64, 0.43, 0.01, -0.03, 0.32, 0.44, 0.22, 0.21],
            dtype=np.float64,
        )
        predicted = model.predict(query, (1920, 1080))
        expected_x = 960 + 900 * 0.12 + 120 * 0.44
        expected_y = 540 - 70 * 0.12 + 600 * 0.44

        self.assertLess(abs(predicted[0] - expected_x), 30)
        self.assertLess(abs(predicted[1] - expected_y), 30)

    def test_save_and_load(self) -> None:
        model = CalibrationModel()
        sample = CalibrationSample(
            target=(100, 200),
            feature_vector=np.asarray([0.1, 0.2, 0.3], dtype=np.float64),
        )
        model.fit([sample, sample, sample])

        with tempfile.TemporaryDirectory() as temp_dir:
            path = f"{temp_dir}/calibration.json"
            model.save(path)
            loaded = CalibrationModel.load(path)
            self.assertEqual(loaded.feature_count, model.feature_count)
            self.assertTrue(np.allclose(loaded.coefficients, model.coefficients))

    def test_calibration_targets_cover_screen(self) -> None:
        targets = calibration_targets((1000, 500))
        self.assertEqual(len(targets), 9)
        self.assertIn((500, 250), targets)
        self.assertIn((100, 50), targets)
        self.assertIn((900, 450), targets)


if __name__ == "__main__":
    unittest.main()
