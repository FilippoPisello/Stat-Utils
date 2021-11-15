"""Contains functions to use on dataframes to detect outliers from series."""
from typing import Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from plot_tools.basics import plot_series
from plot_tools.extras import add_symmetric_interval

import outliers.detectors as detect


def outliers_plot(
    series: Union[pd.Series, np.ndarray],
    method: str = "mean",
    quantile: float = 0.95,
    figsize: tuple[int, int] = (20, 8),
    lines_color: str = "red",
) -> None:
    """Plot the input series denoting the tresholds for outliers.

    Parameters
    ----------
    series : Union[pd.Series, np.array]
        The series to be plotted.
    method : str, optional
        The method to be used to identify the outliers, by default "mean".
    quantile : float, optional
        The quantile to be used to determine the acceptance range. Given a value
        X, it is expected that X% of the observations will fall in the range.
    figsize : tuple[int, int], optional
        The size of the plot, by default (20, 8).
    lines_color : str, optional
        The matplotlib color of the lines delimiting the outliers tresholds, by
        default "red".
    """
    detector = _detector_from_method(series, method, quantile)
    _, ax = plot_series(
        series,
        figsize,
        title=f"Outliers plot for {series.name}",
        return_plot=True,
    )
    add_symmetric_interval(
        ax, detector.center, detector.width, double_variation=False, color=lines_color
    )
    plt.show()


def dataframe_outliers(
    dataframe: pd.DataFrame, column: str, method: str = "mean", quantile: float = 0.95
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
    quantile : float, optional
        The quantile to be used to determine the acceptance range. Given a value
        X, it is expected that X% of the observations will fall in the range.

    Returns
    -------
    pd.DataFrame
        The data frame containing only the rows having values in the column of
        interest marked as outliers.
    """
    output = dataframe.copy()
    outliers = series_outliers(output[column], method, quantile)
    return output.loc[outliers.index, :]


def series_outliers(
    series: Union[pd.Series, np.ndarray], method: str = "mean", quantile: float = 0.95
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
    quantile : float, optional
        The quantile to be used to determine the acceptance range. Given a value
        X, it is expected that X% of the observations will fall in the range.

    Returns
    -------
    pd.Series
        The series containing only the values identified as outliers.
    """
    detector = _detector_from_method(series, method, quantile)
    return series[(series < detector.range_min) | (series > detector.range_max)]


def _detector_from_method(
    series: pd.Series, method: str, quantile: int
) -> detect.Detector:
    """Return a detector object to identify outliers in a series."""
    if method == "mean":
        return detect.MeanDetector(series, quantile)
    if method == "mad":
        return detect.MadDetector(series, quantile)
    raise ValueError(f"No method called {method}")
