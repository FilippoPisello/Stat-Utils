from typing import Any, Union

import numpy as np
import pandas as pd
from distances.euclidean import euclidean_from_point
from predictions.prediction import Prediction
from predictions.validation import leave_one_out_validation

from classifiers.classifier import Classifier


class KNeighborsClassifier(Classifier):
    """
    Class to categorize data using the K-Neighbors-Classification. The
    technique identifies the K nearest observation from the element considered
    and infers its category from the neighbors' categories.

    As an example, if one is classifying "Good" and "Bad" students, if 4 out of
    the 5 considered neighbors are labeled as "Good", than "Good" is inferred.

    Attributes
    ----------
    predictors : np.ndarray
        Numpy array of shape (N, K) where N is the number of observations and K
        is the number of variables. It contains the data for the predictors
        variables, also referred as Xs or exogenous variables.

    std_predictors : np.ndarray
        Numpy array of shape (N, K) containing the standardized predictors.

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

    def __init__(
        self,
        predictors: np.ndarray,
        outcomes: np.ndarray,
        standardize: bool = True,
    ):
        """Create a KNeighborsClassifier instance from a dataframe.

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

        standardize: bool, optional
            If True, the data used to compute the distances - thus the predictors
            - is standardized. This technique presupposes data to be standardized
            so this parameter should be set to True unless the data is not
            already standardized. By default, True.
        """
        super().__init__(predictors, outcomes)
        # Predictors and outcomes as numpy arrays
        self.std_predictors = (
            self._standardize_with_predictors(self.predictors)
            if standardize
            else self.predictors
        )

    @classmethod
    def from_dataframe(
        cls,
        dataframe: pd.DataFrame,
        classifier_col: str,
        usecols: list[str] = None,
        standardize: bool = True,
    ):
        """Class to categorize data using the K-Neighbors-Classification. The
        technique identifies the K nearest observation from the element
        considered and infers its category from the neighbors' categories.

        As an example, if one is classifying "Good" and "Bad" students, if 4 out
        of the 5 considered neighbors are labeled as "Good", than "Good" is
        inferred.

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

        standardize: bool, optional
            If True, the data used to compute the distances - thus the predictors
            - is standardized. This technique presupposes data to be standardized
            so this parameter should be set to True unless the data is not
            already standardized. By default, True.
        """
        return super().from_dataframe(
            dataframe, classifier_col, usecols, standardize=standardize
        )

    def neighbors_performance(
        self,
        neighbors_to_test: list[int] = range(1, 11),
        validation: str = "leave one out",
    ) -> pd.DataFrame:
        """Return a dataframe containing the accuracy achieved over the training
        set for each of the number of neighbors passed with the parameter
        neighbors_to_test.

        Parameters
        ----------
        neighbors_to_test : list, optional
            A list of integers representing the number of neighbors the algorithm
            accuracy is to be tested with. By default range(1, 11)
        validation : str, optional
            The validation technique to be used to compute the distances. By
            default "leave one out".

        Returns
        -------
        pd.DataFrame
            A dataframe of shape (N, 2) where N is the length of the list
            passed as neighbors_to_test. The first columns contains the number
            of neighbors tested and the second columns contains the accuracy
            of the result.
        """
        accuracy = [
            self.categorize_training_data(
                validation=validation, as_prediction=True, n_neighbors=number
            ).percentage_correctly_classified
            for number in neighbors_to_test
        ]
        return pd.DataFrame(
            {"Neighbors Considered": neighbors_to_test, "Accuracy": accuracy}
        )

    def categorize_training_data(
        self,
        n_neighbors: int = 5,
        validation: str = "leave one out",
        as_prediction: bool = False,
    ) -> Union[pd.Series, Prediction]:
        """Return the inferred category for each observations in the training
        data frame.

        Parameters
        ----------
        n_neighbors : int, optional
            The number of neighbors to be used to infer the categories,
            by default 5.

        validation : str, optional
            If str, the method to be used for validation.

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
        if validation in ["loo", "leave one out"]:
            categories = leave_one_out_validation(
                data=self.predictors,
                loo_class_callable=self._loo_distances_training_data,
                n_neighbors=n_neighbors,
            )
            categories = categories.squeeze()

        if not as_prediction:
            categories = pd.Series(categories)
            return categories

        return Prediction(categories, self.outcomes)

    def _loo_distances_training_data(
        self, single_row: np.ndarray, index: int, n_neighbors: int
    ):
        """Method to be passed to the leave one out validator. It performs
        each individual loop of the validation."""
        filt = np.ones(self.number_obs).astype(bool)
        filt[index] = False

        classifier = KNeighborsClassifier(
            predictors=self.predictors[filt],
            outcomes=self.outcomes[filt],
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
        usecols: Union[None, list[str]] = None,
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
        # Standardization happens here so that it is applied to all the passed
        # values at once
        data = self._preprocess_new_data(
            new_data, standardize=standardize_new, usecols=usecols
        )

        axis = 0 if data.ndim == 1 else 1
        categories = np.apply_along_axis(
            func1d=self.categorize_element,
            axis=axis,
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
        data : np.ndarray
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
        if standardize:
            data = self._standardize_with_predictors(data)
        distances = self.distances_from_observations(data)
        neighbors = self.neighbors_from_distances(distances, n_neighbors)
        return self.category_from_neighbors(neighbors)

    # from NEW DATA to NEW DATA
    def _preprocess_new_data(
        self,
        new_data: Union[pd.DataFrame, np.ndarray],
        standardize: bool,
        usecols: Union[None, list[str]],
    ) -> np.ndarray:
        """Return new_data in numpy form if not numpy, making sure that it is
        compatible with existing data."""
        data = new_data.copy()

        if isinstance(new_data, pd.DataFrame):
            if usecols is not None:
                data = data.loc[:, usecols]
            data = data.to_numpy()

        if standardize:
            return self._standardize_with_predictors(data)
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
        return self.outcomes[idx_closest[:n_neighbors]]

    def distances_from_observations(self, data: np.ndarray) -> np.ndarray:
        """Return the distance between the single item data and all the
        observations in the dataset.

        Parameters
        ----------
        data : np.ndarray
            The array containing the new data the distance should be measured
            for. Its shape must be (K, 1) where K is the number of columns used
            to compute the distance.

        Returns
        -------
        np.ndarray
            The 1D array of shape (N,) containing the distances. Each element of
            index i is the distance between data and observation with index i
            in the dataset.
        """
        return euclidean_from_point(self.std_predictors, data)

    def _standardize_with_predictors(self, array: np.ndarray) -> np.ndarray:
        return (array - self.predictors.mean(axis=0)) / self.predictors.std(axis=0)
