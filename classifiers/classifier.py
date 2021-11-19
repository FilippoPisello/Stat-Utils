"""Contains general Classifier superclass."""
import numpy as np
import pandas as pd


class Classifier:
    """
    Class to represent a general classifier.

    Properties
    ----------
    categories : np.ndarray
        One dimensional numpy array of shape (M,) containing all the unique values
        attained by outcomes data. These are all the existing categories.

    number_categories : int
        Number of unique categories.

    number_obs : int
        Number of observations provided. Corresponds to predictors.shape[0].

    number_vars : int
        Number of predictors. Corresponds to predictors.shape[1].
    """

    def __init__(
        self,
        predictors: np.ndarray,
        outcomes: np.ndarray,
    ):
        """Class to represent a general classifier.

        Parameters
        ----------
        predictors : np.ndarray
            Numpy array of shape (N, K) where N is the number of observations
            and K is the number of variables. It contains the data for the
            predictors variables, also referred as Xs or exogenous variables.

        outcomes : np.ndarray
            One dimensional numpy array of shape (N,) containing the data
            relative to the value to be predicted. Also referred as Y or
            endogenous variable.
        """
        self.predictors = predictors
        self.outcomes = outcomes

    @classmethod
    def from_dataframe(
        cls,
        dataframe: pd.DataFrame,
        classifier_col: str,
        usecols: list[str] = None,
        **kwargs,
    ):
        """Create a class instance from a dataframe.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The pandas dataframe containing the data.

        classifier_col : str
            The label of the column in the dataframe that contains the information
            over the classification.

        usecols : list[str], optional
            The list of columns that contain data to be used to assess the
            distance. If None, all the dataframe columns excluding classifier_col
            are used.
        """
        if usecols is None:
            usecols = dataframe.columns.to_list()
            usecols.remove(classifier_col)

        predictors = dataframe[usecols].to_numpy()
        outcomes = dataframe[classifier_col].to_numpy()
        return cls(predictors, outcomes, **kwargs)

    @property
    def categories(self) -> np.ndarray:
        """Return the unique values on the classifier column."""
        return np.unique(self.outcomes, return_counts=False)

    @property
    def number_categories(self) -> int:
        """Return the number of unique values in the classifier column."""
        return len(self.categories)

    @property
    def number_obs(self) -> int:
        """Return the number of observations in the data frame."""
        return self.predictors.shape[0]

    @property
    def number_vars(self) -> int:
        """Return the number of variables in the data frame."""
        return self.predictors.shape[1]
