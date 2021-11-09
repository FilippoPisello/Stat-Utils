from unittest import TestCase

import numpy as np
import pandas as pd
from preprocessing import scaling


class TestPandasScaling(TestCase):
    a1 = np.array([1, 2, 3, 5, 6, 9])
    a2 = np.array([5, 5, 5, 5, 5, 5])
    a3 = np.array([3, 3, 3, 3, 3, 3])

    df1 = pd.DataFrame({"Col1": a1, "Col2": a2, "Col3": a3})
    df2 = pd.DataFrame({"Col1": a1, "Col2": a2, "Col3": ["A"] * 6})

    def test_series_scaling(self):
        """Check that series are scaled correctly."""
        # Check that expected result is obtained for regular series
        res = scaling.standardize_series(self.df1["Col1"])
        exp = pd.Series(
            [-1.132277, -0.792594, -0.452911, 0.226455, 0.566139, 1.585188], name="Col1"
        )
        pd.testing.assert_series_equal(res, exp)

        # Check that expected result is obtained for 0 std series
        res = scaling.standardize_series(self.df1["Col2"])
        exp = pd.Series([0.0] * 6, name="Col2")
        pd.testing.assert_series_equal(res, exp)

        # Check that type error is correctly raised with non-numeric series
        self.assertRaises(TypeError, scaling.standardize_series, self.df2["Col3"])

    def test_dataframe_scaling(self):
        """Check that dataframe scaling is correctly executed."""
        # Check that expected result is obtained for numerical only df
        res = scaling.standardize_dataframe(self.df1)
        exp = pd.DataFrame(
            {
                "Col1": [-1.132277, -0.792594, -0.452911, 0.226455, 0.566139, 1.585188],
                "Col2": [0.0] * 6,
                "Col3": [0.0] * 6,
            }
        )
        pd.testing.assert_frame_equal(res, exp)

        # Check that expected result is obtained with non-numerical series
        res = scaling.standardize_dataframe(self.df2)
        exp = pd.DataFrame(
            {
                "Col1": [-1.132277, -0.792594, -0.452911, 0.226455, 0.566139, 1.585188],
                "Col2": [0.0] * 6,
                "Col3": ["A"] * 6,
            }
        )
        pd.testing.assert_frame_equal(res, exp)

        # Check that expected result is obtained when selecting columns
        res = scaling.standardize_columns(self.df1, ["Col1", "Col2"])
        exp = pd.DataFrame(
            {
                "Col1": [-1.132277, -0.792594, -0.452911, 0.226455, 0.566139, 1.585188],
                "Col2": [0.0] * 6,
                "Col3": [3] * 6,
            }
        )
        pd.testing.assert_frame_equal(res, exp, check_dtype=False)


class TestNumpyScaling(TestCase):
    a1 = np.array([1, 2, 3, 5, 6, 9])
    a2 = np.array([5, 5, 5, 5, 5, 5])
    a3 = np.array([3, 3, 3, 3, 3, 3])
    a4 = np.array([a1, a2, a3])

    def test_standardization(self):
        """Test the correct standardization"""
        # Check that 1D standardization works with regular values
        res1 = scaling.standardize_array(self.a1)
        # Axis parameter is to be ignored
        res2 = scaling.standardize_array(self.a1, axis=1)
        res3 = scaling.standardize_array(self.a1, axis=0)

        exp = np.array(
            [-1.24034735, -0.86824314, -0.49613894, 0.24806947, 0.62017367, 1.73648628]
        )
        np.testing.assert_allclose(res1, exp, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(res2, exp, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(res3, exp, rtol=1e-6, atol=1e-6)

        # Check that 1D standardization works with constant values
        res = scaling.standardize_array(self.a2)
        exp = np.array([0.0] * 6)
        np.testing.assert_allclose(res, exp, rtol=1e-6, atol=1e-6)

        # Check 2D standardization
        res = scaling.standardize_array(self.a4)
        exp = np.array(
            [
                [
                    -1.22474487,
                    -1.06904497,
                    -0.70710678,
                    0.70710678,
                    1.06904497,
                    1.33630621,
                ],
                [
                    1.22474487,
                    1.33630621,
                    1.41421356,
                    0.70710678,
                    0.26726124,
                    -0.26726124,
                ],
                [0.0, -0.26726124, -0.70710678, -1.41421356, -1.33630621, -1.06904497],
            ]
        )
        np.testing.assert_allclose(res, exp, rtol=1e-6, atol=1e-6)
