from unittest import TestCase

import numpy as np
import pandas as pd
from classifiers.mahalanobis import MahalanobisClassifier
from distances.distance import mahanalobis_from_center


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
        arr = np.array(self.df1[["Duration", "Budget"]].head(500))
        res = mahanalobis_from_center(arr)
        np.testing.assert_allclose(res[:20], exp, rtol=1e-5, atol=1e-5)

    def test_point_min_distance(self):
        exp = np.array(
            [
                [1.0],
                [1.0],
                [0.0],
                [0.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [0.0],
                [0.0],
                [1.0],
                [0.0],
                [0.0],
                [1.0],
                [1.0],
                [1.0],
                [0.0],
                [1.0],
                [1.0],
                [1.0],
            ]
        )

        cls = MahalanobisClassifier(
            self.df1.head(500), "IsGood", ["Duration", "Budget"]
        )
        res = cls.distance_training_data()
        np.testing.assert_array_equal(res[:20], exp)
