from unittest import TestCase

import numpy as np
import pandas as pd
from distances import hamming


class TestHamming(TestCase):
    h1 = np.array(["A", "B", "C"])
    h2 = np.array(["A", "B", "D"])
    h3 = np.array([h1, h2])
    h4 = np.array([["A"] * 3, ["A"] * 3])
    h5 = np.array(["A", "B", "D", "E"])

    def test_point_distance(self):
        """Test that distance from a single point is computed correctly."""
        # One element vs one point
        res = hamming.hamming_from_element(self.h1, self.h2)
        np.testing.assert_array_equal(res, np.array([1]))
        # One element vs itself
        res = hamming.hamming_from_element(self.h1, self.h1)
        np.testing.assert_array_equal(res, np.array([0]))
        # One element vs one point, with weight
        res = hamming.hamming_from_element(self.h1, self.h2, weight=3)
        np.testing.assert_array_equal(res, np.array([3]))
        res = hamming.hamming_from_element(self.h1, self.h2, weight=np.array([3, 1, 1]))
        np.testing.assert_array_equal(res, np.array([1]))

        # Two elements vs one point
        res = hamming.hamming_from_element(self.h3, self.h2)
        np.testing.assert_array_equal(res, np.array([1, 0]))

    def test_points_distance(self):
        # One element vs two points
        res = hamming.hamming_from_elements(self.h1, self.h3)
        np.testing.assert_array_equal(res, np.array([[0, 1]]))

        # Two elements vs two points
        res = hamming.hamming_from_elements(self.h3, self.h3)
        np.testing.assert_array_equal(res, np.array([[0, 1], [1, 0]]))
