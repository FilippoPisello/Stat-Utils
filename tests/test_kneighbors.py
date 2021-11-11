from unittest import TestCase

import numpy as np
import pandas as pd
from classifiers.kneighbors import KNeighborsClassifier


class TestKneighbors(TestCase):
    df = pd.read_excel("tests/test_data/admissions.xlsx")
    # THE PROCESS IS STOCHASTIC SO A SEED IS NEEDED
    np.random.seed(9999)

    def test_looclassification(self):
        # Test various numbers of neighbors
        cls = KNeighborsClassifier(self.df, "De", standardize=True)
        res = cls.categorize_training_data(n_neighbors=5)
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

        cls1 = KNeighborsClassifier(self.df, "De", standardize=True)
        res1 = cls1.categorize_training_data(n_neighbors=10)
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

        # Test that the result is invariant from alphabetic order
        exp2 = exp.replace("admit", "zadmit")
        df1 = self.df.copy()
        df1["De"] = df1["De"].replace("admit", "zadmit")
        cls2 = KNeighborsClassifier(df1, "De", standardize=True)
        res2 = cls2.categorize_training_data(n_neighbors=10)
        pd.testing.assert_series_equal(res2, exp2)
