from unittest import TestCase

import pandas as pd
from outliers_detection import detectors, outliers
from scipy.stats import median_abs_deviation


class TestOutlier(TestCase):

    df1 = pd.read_excel("tests/test_data/returns.xlsx")
    series1: pd.Series = df1["Ret1"]

    def test_mean_detector(self):
        """Test that mean-based detector works correctly"""
        mean, std = self.series1.mean(), self.series1.std()

        for factor in [2, 3, 10]:
            detector = detectors.MeanDetector(self.series1, factor)
            self.assertEqual(detector.center, mean)
            self.assertEqual(detector.range_min, mean - factor * std)
            self.assertEqual(detector.range_max, mean + factor * std)

    def test_mad_detector(self):
        """Test that mad-based detector works correctly"""
        center = median_abs_deviation(self.series1)
        std = center * 1.48

        for factor in [2, 3, 10]:
            detector = detectors.MadDetector(self.series1, factor)
            self.assertEqual(detector.center, center)
            self.assertAlmostEqual(detector.range_min, center - factor * std, places=5)
            self.assertAlmostEqual(detector.range_max, center + factor * std, places=5)
