from unittest import TestCase

import numpy as np
import pandas as pd
from classifiers.kneighbors import KNeighborsClassifier


class TestKneighbors(TestCase):
    df = pd.read_excel("tests/test_data/admissions.xlsx")
    cls = KNeighborsClassifier.from_dataframe(df, "De", standardize=True)

    # THE PROCESS IS STOCHASTIC SO A SEED IS NEEDED
    np.random.seed(9999)

    def test_neighbors(self):
        new_data = np.array([313.0, 3.15])

        for number in range(1, 10):
            new_data = (
                new_data - self.cls.predictors.mean(axis=0)
            ) / self.cls.predictors.std(axis=0)
            dist = self.cls.distances_from_observations(new_data)
            neigh = self.cls.neighbors_from_distances(dist, n_neighbors=number)
            self.assertEqual(len(neigh), number)

    def test_means(self):
        res = self.cls.predictors.mean(axis=0)
        exp = np.array([2.97458824, 488.44705882])
        np.testing.assert_allclose(res, exp, rtol=1e-5, atol=1e-5)

    def test_looclassification(self):
        # Test various numbers of neighbors
        res = self.cls.categorize_training_data(n_neighbors=5)
        exp = pd.Series(
            [
                "admit",
                "border",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "border",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "border",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "notadmit",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
            ]
        )
        pd.testing.assert_series_equal(res, exp)

        res1 = self.cls.categorize_training_data(n_neighbors=10)
        exp1 = pd.Series(
            [
                "admit",
                "border",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "border",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "border",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "notadmit",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
            ]
        )
        pd.testing.assert_series_equal(res1, exp1)

        res3 = self.cls.categorize_training_data(n_neighbors=3)
        exp3 = pd.Series(
            [
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "admit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "notadmit",
                "border",
                "border",
                "border",
                "border",
                "border",
                "notadmit",
                "admit",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "notadmit",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
                "border",
            ]
        )
        pd.testing.assert_series_equal(res3, exp3)

        # Test that the result is invariant from alphabetic order
        exp2 = exp.replace("admit", "zadmit")
        df1 = self.df.copy()
        df1["De"] = df1["De"].replace("admit", "zadmit")
        cls2 = KNeighborsClassifier.from_dataframe(df1, "De", standardize=True)
        res2 = cls2.categorize_training_data(n_neighbors=10)
        pd.testing.assert_series_equal(res2, exp2)

    def test_categorize_new(self):
        """Check that new object is categorized correctly"""
        res = self.cls.categorize_new_data(np.array([3.5, 600]))
        pd.testing.assert_series_equal(res, pd.Series(["admit"]))

        res = self.cls.categorize_new_data(np.array([[3.5, 600], [3.5, 600]]))
        pd.testing.assert_series_equal(res, pd.Series(["admit", "admit"]))
