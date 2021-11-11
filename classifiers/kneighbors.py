from typing import Any, Union

import numpy as np
import pandas as pd
from distances.euclidean import euclidean_from_point
from predictions.prediction import Prediction
from predictions.validation import leave_one_out_validation
from preprocessing.scaling import standardize_array

from classifiers.classifier import Classifier


class KNeighborsClassifier(Classifier):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        classifier_col: str,
        usecols: list[str] = None,
        standardize: bool = True,
    ):
        super().__init__(dataframe, classifier_col, usecols)
        # Predictors and outcomes as numpy arrays
        self.std_data = standardize_array(self.data) if standardize else self.data

    def categorize_training_data(
        self,
        validation: str = "leave one out",
        as_prediction: bool = False,
        n_neighbors: int = 5,
    ) -> Union[pd.Series, Prediction]:
        if validation in ["loo", "leave one out"]:
            categories = leave_one_out_validation(
                data=self.data,
                loo_class_callable=self._loo_distances_training_data,
                output_shape=(self.number_obs, 1),
                output_type=self.category_series.dtype,
                n_neighbors=n_neighbors,
            )
            categories = pd.Series(categories.squeeze())

        if not as_prediction:
            return categories
        return Prediction(categories, self.category_series)

    def _loo_distances_training_data(
        self, single_row: np.ndarray, index: int, n_neighbors: int
    ):
        """Method to be passed to the leave one out validator. It performs
        each individual loop of the validation."""
        classifier = KNeighborsClassifier(
            dataframe=self.df_all[~self.df_all.index.isin([index])],
            classifier_col=self.class_col,
            usecols=self.data_columns,
            standardize=True,
        )
        return classifier.categorize_element(
            single_row, standardize=True, n_neighbors=n_neighbors
        )

    def categorize_new_data(
        self,
        new_data: Union[pd.DataFrame, np.ndarray],
        n_neighbors: int = 5,
        standardize_new: bool = True,
    ) -> pd.Series:
        """Return a 1D array containing the inferred categories for the passed
        data.

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

        n_neighbors : int, optional
            The number of neighbors to be used to perform the classification,
            by default 5.

        standardize_new : bool, optional
            If True, the new passed data is standardized using the whole dataset
            mean and std before the distance is computed, by default True.

        Returns
        -------
        pd.Series
            [description]
        """
        data = self._preprocess_new_data(new_data, standardize=standardize_new)
        categories = np.apply_along_axis(
            func1d=self.categorize_element,
            axis=1,
            arr=data,
            standardize=False,
            n_neighbors=n_neighbors,
        )
        return pd.Series(categories)

    def categorize_element(
        self, data: np.ndarray, standardize: bool, n_neighbors: int
    ) -> Any:
        """Return the inferred category for the passed element.

        Parameters
        ----------
        data : np.array
            The array containing the new data the distance should be measured
            for. Its shape must be (K, 1) where K is the number of columns used
            to compute the distance.

        standardize : bool, optional
            If True, the new passed data is standardized using the whole dataset
            mean and std before the distance is computed, by default True.

        n_neighbors : int
            The number of neighbors to be considered.

        Returns
        -------
        Any
            The inferred category for the passed element.
        """
        distances = self.distances_from_observations(data, standardize)
        neighbors = self.neighbors_from_distances(distances, n_neighbors)
        return self.category_from_neighbors(neighbors)

    # from NEW DATA to NEW DATA
    def _preprocess_new_data(
        self, new_data: Union[pd.DataFrame, np.ndarray], standardize: bool
    ) -> np.ndarray:
        """Return new_data in numpy form if not numpy, making sure that it is
        compatible with existing data."""
        data = new_data.copy()
        if isinstance(new_data, pd.DataFrame):
            try:
                data = data.loc[:, self.data_columns]
            except KeyError as e:
                raise KeyError(
                    """The dataframe with new data must have the same
                               data columns as the original one"""
                ) from e
            data = data.to_numpy()
        data = data.squeeze()

        if standardize:
            return (data - self.means()) / self.stds()
        return data

    @staticmethod
    def category_from_neighbors(neighbors_categories: np.ndarray) -> Any:
        """Return the category inferred from the categories of the passed
        neighbors.

        Parameters
        ----------
        neighbors_categories : np.ndarray
            A 1D array of shape (X,) containing the categories
            the specified closest neighbors belong to.

        Returns
        -------
        Any
            The inferred category. It will be the same type of the original
            category.
        """
        values, counts = np.unique(neighbors_categories, return_counts=True, axis=None)

        cat = np.random.choice(values[counts == counts.max()])
        return cat

    def neighbors_from_distances(
        self, distances: np.ndarray, n_neighbors: int
    ) -> np.ndarray:
        """Return the categories the specified number of neighbors belong to
        given an array containing the distances from each of them.

        Parameters
        ----------
        distances : np.ndarray
            A 1D array of shape (N,) containing the distances. Each element of
            index i is the distance between data and observation with index i
            in the dataset.

        n_neighbors : int
            The number of neighbors to be considered.

        Returns
        -------
        np.ndarray
            The 1D array of shape (n_neighbors,) containing the categories
            the specified closest neighbors belong to.
        """
        idx_closest = np.argsort(distances)
        return self.outcomes[idx_closest[:n_neighbors], 0]

    def distances_from_observations(
        self, data: np.ndarray, standardize: bool = True
    ) -> np.ndarray:
        """Return the distance between the single item data and all the
        observations in the dataset.

        Parameters
        ----------
        data : np.ndarray
            The array containing the new data the distance should be measured
            for. Its shape must be (K, 1) where K is the number of columns used
            to compute the distance.

        standardize : bool, optional
            If True, the new passed data is standardized using the whole dataset
            mean and std before the distance is computed, by default True.

        Returns
        -------
        np.ndarray
            The 1D array of shape (N,) containing the distances. Each element of
            index i is the distance between data and observation with index i
            in the dataset.
        """
        if standardize:
            data = (data - self.means()) / self.stds()
        return euclidean_from_point(self.std_data, data)

    def stds(self, as_series: bool = False):
        stds = self.df_all.loc[:, self.data_columns].std(axis=0)
        if as_series:
            return stds
        return stds.to_numpy()

    def means(self, as_series: bool = False):
        means = self.df_all.loc[:, self.data_columns].mean(axis=0)
        if as_series:
            return means
        return means.to_numpy()
