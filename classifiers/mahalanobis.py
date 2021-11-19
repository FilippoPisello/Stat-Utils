"""Contains MahalanobisClassifier class which allows to categorize observations
based on the computation of the mahanalobis distance from the centers of
of some data categories."""
from typing import Union

import numpy as np
import pandas as pd
from distances.mahalanobis import mahanalobis_from_points
from predictions.prediction import Prediction
from predictions.validation import leave_one_out_validation

from classifiers.classifier import Classifier


class MahalanobisClassifier(Classifier):
    """Class to categorize data using the Discriminant Analysis, the technique
    based on the minimization of Mahalanobis distance.

    The classification builds upon the identification of observations groups
    based on the values attained for the outcome variable. For each of this
    group it is computed the 2D mean accounting for all the predictors. The
    categorization happens by matching the observation in analysis with the
    group whose center is closest.

    Attributes
    ----------
    predictors : np.ndarray
        Numpy array of shape (N, K) where N is the number of observations and K
        is the number of variables. It contains the data for the predictors
        variables, also referred as Xs or exogenous variables.

    outcomes : np.ndarray
        One dimensional numpy array of shape (N,) containing the data relative
        to the value to be predicted. Also referred as Y or endogenous variable.

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

    def __init__(self, predictors: np.ndarray, outcomes: np.ndarray):
        """Class to categorize data using the Discriminant Analysis, the
        technique based on the minimization of Mahalanobis distance.

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
        super().__init__(predictors, outcomes)

    # from EXISTING DATA to CATEGORY
    def categorize_training_data(
        self,
        validation: Union[str, None] = None,
        as_prediction: bool = False,
    ) -> Union[pd.Series, Prediction]:
        """Return a pandas series with length N containing the inferred category
        for each observation in the training data frame.

        Parameters
        ----------
        validation: str, optional
            If str, the method to be used for validation. If None, no validation
            is applied and direct output is returned. Below the validations
            available:
            "loo" or "leave one out": leave one out validation. Each category
            is derived by using every other observation in the dataframe and
            passing the single excluded item as new data.

        as_prediction : bool, optional
            If True, the categorization is returned as a prediction object
            that allows to calculate accuracy metrics. By default False.

        Returns
        -------
        Union[pd.Series, Prediction]
            If as_prediction is False, returns a pd.Series.
            Series of length N. First element contains the category for the first
            observation and so on.

            If as_prediction is True, return a Prediction object, whose attribute
            obj.fitted_values is the series described above, while obj.real_values
            is the predicted series from the dataframe.
        """
        if validation is None:
            distances = self.distances_training_data()
        elif validation in ["loo", "leave one out"]:
            distances = leave_one_out_validation(
                data=self.predictors,
                loo_class_callable=self._loo_distances_training_data,
            )

        categories = self.categories_from_distances(distances)

        if not as_prediction:
            categories = pd.Series(categories)
            return categories

        return Prediction(categories, self.outcomes)

    # from EXISTING DATA to DISTANCES
    def distances_training_data(
        self, sqrt: bool = False, as_dataframe: bool = False
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Return the distances for each variable from the centers of the groups
        identified by the classifier column.

        Parameters
        ----------
        sqrt : bool, optional
            If True, the square root is applied to the distances. By default False.
        as_dataframe : bool, optional
            If True, return the distances in data frame form. Each column matches
            a value from the classifier while each row matches an observation.
            By default False.

        Returns
        -------
        Union[np.ndarray, pd.DataFrame]
            The result has size (N, M) where N is the number of observations while
            M is the number of groups. Each element (x, y) is the distance between
            observation x and the center of group y, where x = [1, ..., N] and
            y = [1, ..., M].
        """
        return self.distances_from_centers(
            self.predictors, sqrt=sqrt, as_dataframe=as_dataframe
        )

    # from EXISTING DATA to DISTANCES, for LOO validation
    def _loo_distances_training_data(self, single_row: np.ndarray, index: int):
        """Method to be passed to the leave one out validator. It performs
        each individual loop of the validation."""
        filt = np.ones(self.number_obs).astype(bool)
        filt[index] = False

        classifier = MahalanobisClassifier(
            predictors=self.predictors[filt],
            outcomes=self.outcomes[filt],
        )
        return classifier.distances_from_centers(single_row)

    # from NEW DATA to CATEGORY
    def categorize_new_data(
        self,
        new_data: Union[pd.DataFrame, np.ndarray],
        usecols: Union[None, list[str]] = None,
    ) -> pd.Series:
        """Given a set of data with size (N, K), return a pandas series with length
        N containing the mahanalobis categories for the input data.

        The observation is assigned to the category whose center is closest.


        Parameters
        ----------
        new_data : Union[pd.DataFrame, np.ndarray]
            The new data to be classified.

            If pandas dataframe any number of columns is allowed as long as the
            ones used for centers calculation are present. More on KeyError note
            down here.

            If array, its shape must be (N, K), where K is the number of columns
            used for the original centers calculation. A single observation can
            also be passed as a 1D array --> [observation].
            It is assumed that the columns match the order of the ones
            in the original data frame.

        Returns
        -------
        pd.Series
            Series of length N. First element contains the category for the first
            observation and so on.

        Raises
        ------
        KeyError
            Raised if among the columns of the data provided in data frame the
            original data columns are not found.
            Example: if the categories centers were calculated over the columns
            "Foo" "Bar", then "Foo" "Bar" must be within new_data columns.

        """
        data = self._preprocess_new_data(new_data, usecols=usecols)
        distances = self.distances_from_centers(data)
        return self.categories_from_distances(distances)

    # from NEW DATA to NEW DATA
    def _preprocess_new_data(
        self,
        new_data: Union[pd.DataFrame, np.ndarray],
        usecols: Union[None, list[str]],
    ) -> np.ndarray:
        """Return new_data in numpy form if not numpy, making sure that it is
        compatible with existing data."""
        data = new_data.copy()

        if isinstance(new_data, pd.DataFrame):
            if usecols is not None:
                data = data.loc[:, usecols]
            data = data.to_numpy()

        return data.squeeze()

    # GENERAL: from DISTANCES to CATEGORY
    def categories_from_distances(self, distances: np.ndarray) -> pd.Series:
        """Return a pandas series with length N containing the inferred category
        for each element in the distances array.

        The observation is assigned to the category whose center is closest.

        Parameters
        ----------
        distances : np.ndarray
            An array of shape (N, M), containing the distances from M points for
            N observations.

        Returns
        -------
        pd.Series
            Series of length N. First element contains the category for the first
            observation and so on.
        """
        categories_indexes = pd.Series(np.argmin(distances, axis=1), name="Category")
        cats = self.categories
        return categories_indexes.apply(lambda x: cats[x])

    # GENERAL: DATA to DISTANCES
    def distances_from_centers(
        self, data: np.ndarray, sqrt: bool = False, as_dataframe: bool = False
    ) -> np.ndarray:
        """Return the distances for each observation from the centers of the groups
        identified by the classifier column.

        Parameters
        ----------
        data: np.ndarray
            The numpy array of size (N, K) containing the data to calculate the
            distances for.
        sqrt : bool, optional
            If True, the square root is applied to the distances. By default False.
        as_dataframe : bool, optional
            If True, return the distances in data frame form. Each column matches
            a value from the classifier while each row matches an observation.
            By default False.

        Returns
        -------
        Union[np.ndarray, pd.DataFrame]
            The result has size (N, M) where N is the number of observations while
            M is the number of groups. Each element (x, y) is the distance between
            observation x and the center of group y, where x = [1, ..., N] and
            y = [1, ..., M].
        """
        distances = mahanalobis_from_points(
            data, self.means_matrix(), self.cov_matrix(), sqrt=sqrt
        )
        if as_dataframe:
            return pd.DataFrame(distances, columns=self.categories)
        return distances

    def means_matrix(self) -> np.ndarray:
        """Return the means with shape (M, K) where K is the number of variables
        and M is the number of different values attained by the classifier col."""
        return np.array(
            [
                (self.predictors[self.outcomes == val, :]).mean(axis=0)
                for val in self.categories
            ]
        )

    def cov_matrix(self) -> np.ndarray:
        """Return the covariances with shape (M, K, K) where K is the number
        of variables and M is the number of different values attained by the
        classifier column.

        Each element (z, i, j) of the matrix represents the covariance between
        the variable i and j, conditional to value z. For example, element
        (0, 1, 0) is the covariance between the first and second variable for the
        group of observations with the first value for the categorization column."""

        return np.array(
            [
                np.cov(self.predictors[self.outcomes == val, :].transpose())
                for val in self.categories
            ]
        )
