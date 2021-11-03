from typing import Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from plot_tools.basics import plot_series
from plot_tools.extras import add_symmetric_interval

import outliers_detection.detectors as detect

def dataframe_outliers(
    dataframe: pd.DataFrame, column: str, method: str = "mean", factor: int = 2
) -> pd.DataFrame:
    """Return the outliers for the provided data frame computed through the
    specified method.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe containing the data of interest.
    column : str
        The label of the column to be considered.
    method : str, optional
        The method to be used to compute the boundaries. Accepted methods are
        "mean" (uses mean and standard deviation) and "mad" (using median absolute
        deviation and corrected standard deviation). "Mad" requires normality.
    factor : int, optional
        The number of deviations to be used to compute the interval width.
        By default is equal to 2.

    Returns
    -------
    pd.DataFrame
        The data frame containing only the rows having values in the column of
        interest marked as outliers.
    """
    output = dataframe.copy()
    outliers = series_outliers(output[column], method, factor)
    return output.loc[outliers.index, :]


def series_outliers(
    series: Union[pd.Series, np.array], method: str = "mean", factor: int = 2
) -> pd.Series:
    """Return the outliers for the specified series computed through a provided
    method.

    Parameters
    ----------
    series : Union[pd.Series, np.array]
        Series to be considered.
    method : str
        The method to be used to compute the boundaries. Accepted methods are
        "mean" (uses mean and standard deviation) and "mad" (using median absolute
        deviation and corrected standard deviation). "Mad" requires normality.
    factor : int, optional
        The number of deviations to be used to compute the interval width.
        By default is equal to 2.

    Returns
    -------
    pd.Series
        The series containing only the values identified as outliers.
    """
    detector = _detector_from_method(series, method, factor)
    return series[(series < detector.range_min) | (series > detector.range_max)]


def _detector_from_method(
    series: pd.Series, method: str, factor: int
) -> detect.Detector:
    """Return a detector object to identify outliers in a series."""
    if method == "mean":
        return detect.MeanDetector(series, factor)
    if method == "mad":
        return detect.MadDetector(series, factor)
    raise ValueError(f"No method called {method}")
