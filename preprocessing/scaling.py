"""Functions to rescale data depending on the user's needs"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale


################################################################################
# Functions for pandas objects
################################################################################
def standardize_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the original dataframe where all the numerical columns
    have been individually standardized."""
    new_cols = [
        standardize_series(dataframe[col])
        if pd.api.types.is_numeric_dtype(dataframe[col])
        else dataframe[col]
        for col in dataframe.columns
    ]
    return pd.concat(new_cols, axis=1)


def standardize_columns(dataframe: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a copy of the original dataframe where the columns passed in the
    columns argument have been individually standardized."""
    new_cols = [
        standardize_series(dataframe[col]) if col in columns else dataframe[col]
        for col in dataframe.columns
    ]
    return pd.concat(new_cols, axis=1)


def standardize_series(series: pd.Series) -> pd.Series:
    """Return the standardized version of the passed pandas series. If the
    series is constant, a 0-filled series is returned."""
    try:
        sd = 1 if series.std() == 0 else series.std()
        return (series - series.mean()) / sd
    except TypeError as e:
        raise TypeError(
            "Cannot standardize a non-numerical series."
            f"Series {series.name} is of type {series.dtype}"
        ) from e


def normalize_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the original dataframe where all the numerical columns
    have been individually normalized.

    Normalization happens by diving each element of the series by the maximum
    value of the series itself."""
    new_cols = [
        normalize_series(dataframe[col])
        if pd.api.types.is_numeric_dtype(dataframe[col])
        else dataframe[col]
        for col in dataframe.columns
    ]
    return pd.DataFrame(new_cols)


################################################################################
# Functions for array objects
################################################################################
def standardize_array(array: np.ndarray, axis: int = 0) -> np.ndarray:
    """Return the standardized version of the passed numpy array.

    Parameters
    ----------
    array : np.ndarray
        The array to be standardized.
    axis : int, optional
        If array is unidimensional - array.ndim == 1 - this parameter is
        ignored. Otherwise, axis = 0 should be used if standardization is to
        happen by column, 1 otherwise.

        For general use, axis=0 is used when row represent the observations and
        columns represent the variables.

    Returns
    -------
    np.ndarray
        The standardized version of the array.
    """
    axis = 0 if array.ndim == 1 else axis
    return scale(array, axis=axis)
