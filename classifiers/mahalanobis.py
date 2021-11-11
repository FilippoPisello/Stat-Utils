"""Contains MahalanobisClassifier class which allows to categorize observations
based on the computation of the mahanalobis distance from the centers of
of some data categories."""
from typing import Union

import numpy as np
import pandas as pd
from distances.mahalanobis import mahanalobis_from_points
from predictions.prediction import Prediction
from predictions.validation import leave_one_out_validation


class MahalanobisClassifier:
    def __init__(
        self, dataframe: pd.DataFrame, classifier_col: str, usecols: list[str] = None
    ):
        """Class to categorize data using the Discriminant Analysis, the
        technique based on the minimization of Mahalanobis distance.

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
        self.df_all = dataframe
        self.class_col = classifier_col
        self.data_columns = self._identify_data_columns(usecols)
        self.data = dataframe.loc[:, self.data_columns].to_numpy()

    def _identify_data_columns(
        self, passed_columns: Union[list[str], None]
    ) -> list[str]:
        """Return the correct data columns by excluding the classifier column
        if passed_columns is not provided."""
        if passed_columns is not None:
            passed_columns.sort()
            return passed_columns
        output = self.df_all.columns.to_list()
        output.remove(self.class_col)
        output.sort()
        return output

    @property
    def categories(self) -> pd.Series:
        """Return the unique values on the classifier column."""
        cats = self.df_all[self.class_col].unique()
        cats.sort()
        return cats

    @property
    def category_series(self) -> pd.Series:
        """Return the series of the classifier column"""
        return self.df_all[self.class_col]

    @property
    def number_categories(self) -> int:
        """Return the number of unique values in the classifier column."""
        return len(self.categories)

    @property
    def number_obs(self) -> int:
        """Return the number of observations in the data frame."""
        return self.df_all.shape[0]

    @property
    def number_vars(self) -> int:
        """Return the number of variables in the data frame."""
        return self.df_all.shape[1]

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
                data=self.data,
                output_shape=(self.number_obs, self.number_categories),
                loo_class_callable=self._loo_distances_training_data,
            )

        categories = self.categories_from_distances(distances)
        if not as_prediction:
            return categories
        return Prediction(categories, self.category_series)

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
            self.data, sqrt=sqrt, as_dataframe=as_dataframe
        )

    # from EXISTING DATA to DISTANCES, for LOO validation
    def _loo_distances_training_data(self, single_row: np.ndarray, index: int):
        """Method to be passed to the leave one out validator. It performs
        each individual loop of the validation."""
        classifier = MahalanobisClassifier(
            dataframe=self.df_all[~self.df_all.index.isin([index])],
            classifier_col=self.class_col,
            usecols=self.data_columns,
        )
        return classifier.distances_from_centers(single_row)

    # from NEW DATA to CATEGORY
    def categorize_new_data(
        self, new_data: Union[pd.DataFrame, np.ndarray]
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
        data = self._preprocess_new_data(new_data)
        distances = self.distances_from_centers(data)
        return self.categories_from_distances(distances)

    # from NEW DATA to NEW DATA
    def _preprocess_new_data(
        self, new_data: Union[pd.DataFrame, np.ndarray]
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
            data = data.to_numpy().squeeze()
        return data

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

    def means_matrix(
        self, as_dataframe: bool = False
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Return the means with shape (M, K) where K is the number of variables
        and M is the number of different values attained by the classifier col."""
        means_df = self.df_all.groupby(self.class_col)[self.data_columns].mean()

        if as_dataframe:
            return means_df

        return means_df.to_numpy()

    def cov_matrix(self, as_dataframe: bool = False) -> Union[np.ndarray, pd.DataFrame]:
        """Return the covariances with shape (M, K, K) where K is the number
        of values and M is the number of different values attained by the
        classifier column.

        Each element (z, i, j) of the matrix represents the covariance between
        the variable i and j, conditional to value z. For example, element
        (0, 1, 0) is the covariance between the first and second variable for the
        group of observations with the first value for the categorization column."""
        grouped_df = self.df_all.groupby(self.class_col)

        if as_dataframe:
            return grouped_df[self.data_columns].cov()

        return np.array([group[1][self.data_columns].cov() for group in grouped_df])
