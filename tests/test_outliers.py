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

    def test_outliers_series(self):
        """Test that a series outliers are correctly detected"""
        outs_mean = outliers.series_outliers(self.series1, "mean")
        known_mean = pd.Series(
            [
                20.65081351689612,
                14.542829363949387,
                -11.988791100081352,
                15.431481032436476,
                14.188508759229773,
                -10.759716168521049,
                12.868275522043188,
                -12.838919630020543,
                -14.029581652472746,
                -15.486153272241049,
                15.848746664345391,
                17.33776929576604,
                -10.555994373655013,
                -13.485179073806485,
            ],
            index=[
                11,
                15,
                32,
                33,
                45,
                63,
                75,
                78,
                107,
                109,
                111,
                114,
                124,
                140,
            ],
            name="Ret1",
        )
        pd.testing.assert_series_equal(outs_mean, known_mean)

        outs_mad = outliers.series_outliers(self.series1, "mad")
        known_mad = pd.Series(
            [
                20.65081351689612,
                -8.174272199170124,
                -7.849222172949002,
                14.542829363949387,
                -6.953879280548433,
                -6.099314778783516,
                -11.988791100081352,
                15.431481032436476,
                14.188508759229773,
                -6.985773338654283,
                -8.499056582622899,
                -10.759716168521049,
                -6.129032258064516,
                12.868275522043188,
                -12.838919630020543,
                -7.7153246908458035,
                -10.275918263948375,
                -14.029581652472746,
                -6.517205309051003,
                -15.486153272241049,
                15.848746664345391,
                17.33776929576604,
                -10.555994373655013,
                -8.171613184806954,
                -8.12862326048757,
                -13.485179073806485,
                -8.598929055258472,
                -6.079025534003598,
                -6.202118395326977,
            ],
            index=[
                11,
                12,
                14,
                15,
                19,
                24,
                32,
                33,
                45,
                48,
                54,
                63,
                72,
                75,
                78,
                93,
                101,
                107,
                108,
                109,
                111,
                114,
                124,
                127,
                138,
                140,
                168,
                187,
                201,
            ],
            name="Ret1",
        )
        pd.testing.assert_series_equal(outs_mad, known_mad)
