from unittest import TestCase

import numpy as np
import pandas as pd
from classifiers.mahalanobis import MahalanobisClassifier
from distances.mahalanobis import mahanalobis_from_center, mahanalobis_from_points


class TestMahalanobis(TestCase):
    df1 = pd.read_excel("tests/test_data/movies.xlsx")

    def test_center_distance(self):
        """Test that distance from the series center is computed correctly"""
        exp = np.array(
            [
                [1.42622234],
                [1.24200483],
                [1.59866664],
                [1.42183325],
                [20.68190889],
                [1.08130274],
                [1.10380417],
                [0.8727098],
                [1.87194583],
                [1.97595254],
                [18.28380376],
                [1.60089442],
                [1.62653134],
                [1.06302195],
                [1.09053555],
                [0.8288996],
                [1.36696148],
                [7.2858169],
                [1.47149353],
                [3.52012315],
            ]
        )
        arr = np.array(self.df1[["Budget", "Duration"]])
        res = mahanalobis_from_center(arr)
        np.testing.assert_allclose(res[:20], exp, rtol=1e-5, atol=1e-5)

        exp_sqrt = np.sqrt(exp)
        res_sqrt = mahanalobis_from_center(arr, sqrt=True)
        np.testing.assert_allclose(res_sqrt[:20], exp_sqrt, rtol=1e-5, atol=1e-5)

    def test_points_distance(self):
        exp = np.array(
            [
                [1.41551149, 1.08980545],
                [1.31665212, 1.05116969],
                [1.29927631, 1.31502399],
                [1.21596282, 1.24732306],
                [5.06281735, 4.21684498],
                [1.1452917, 1.05226385],
                [1.20342253, 1.0303845],
                [1.05495851, 0.92586988],
                [1.40005259, 1.41959247],
                [1.40600472, 1.46696041],
                [4.74552411, 3.96824162],
                [1.30069608, 1.31570166],
                [1.31693498, 1.3234666],
                [1.15710169, 1.0292926],
                [1.15167372, 1.05577169],
                [1.05913516, 0.82242966],
                [1.19468852, 1.22329449],
                [3.00686777, 2.48763513],
                [1.47327275, 1.05059172],
                [2.1202364, 1.70433108],
            ]
        )
        arr = self.df1.loc[:, ["Budget", "Duration"]].to_numpy()
        means = np.array([[301.26818811, 85.63948498], [395.92684288, 94.06367041]])
        covs = np.array(
            [
                [[60708.57977331, 2307.08560791], [2307.08560791, 717.66257215]],
                [[142099.42932494, 4063.74304655], [4063.74304655, 910.86435189]],
            ]
        )
        res = mahanalobis_from_points(arr, means, covs)
        res = np.sqrt(res)
        np.testing.assert_allclose(res[:20], exp, rtol=1e-5, atol=1e-5)

    def test_training_categorization(self):
        """Chech that categorization happens correctly."""
        exp = np.array(
            [
                True,
                True,
                False,
                False,
                True,
                True,
                True,
                True,
                False,
                False,
                True,
                False,
                False,
                True,
                True,
                True,
                False,
                True,
                True,
                True,
            ]
        )

        cls = MahalanobisClassifier(self.df1, "IsGood", ["Budget", "Duration"])
        res = cls.categorize_training_data()
        np.testing.assert_array_equal(res[:20], exp)

        df1_mod = self.df1.copy()
        df1_mod["IsGood"] = ~df1_mod["IsGood"]
        cls = MahalanobisClassifier(df1_mod, "IsGood", ["Budget", "Duration"])
        res = cls.categorize_training_data()
        np.testing.assert_array_equal(res[:20], ~exp)

    def test_new_categorization(self):
        """Test that new observations are classified correctly."""
        cls = MahalanobisClassifier(self.df1, "IsGood", ["Budget", "Duration"])

        # Distances: Passing 1 new observation as 1D array
        new_data = np.array([90, 2810])
        dists = mahanalobis_from_points(
            new_data, cls.means_matrix(), cls.cov_matrix(), sqrt=True
        )
        exp = np.array([[108.86547188, 96.65931728]])
        np.testing.assert_allclose(dists, exp, rtol=1e-5, atol=1e-5)
        # Distances: Passing 1 new observation as 2D array
        new_data2 = np.array([[90, 2810]])
        dists = mahanalobis_from_points(
            new_data2, cls.means_matrix(), cls.cov_matrix(), sqrt=True
        )
        np.testing.assert_allclose(dists, exp, rtol=1e-5, atol=1e-5)

        # Distances: Passing 2 new observations
        new_data3 = np.array([[90, 2810], [90, 2810]])
        dists = mahanalobis_from_points(
            new_data3, cls.means_matrix(), cls.cov_matrix(), sqrt=True
        )
        exp = np.array([[108.86547188, 96.65931728], [108.86547188, 96.65931728]])
        np.testing.assert_allclose(dists, exp, rtol=1e-5, atol=1e-5)

        # Categorization: Passing 1 new observation
        classification = cls.categorize_new_data(new_data)
        pd.testing.assert_series_equal(
            classification, pd.Series([True], name="Category")
        )

        # Categorization: Passing 2 new observations
        classification = cls.categorize_new_data(new_data3)
        pd.testing.assert_series_equal(
            classification, pd.Series([True, True], name="Category")
        )
