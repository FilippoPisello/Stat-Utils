from typing import Union

import numpy as np
import pandas as pd
from distances.distance import mahanalobis_from_point, mahanalobis_from_points
from numpy.typing import ArrayLike


class MahalanobisClassifier:
    def __init__(
        self, dataframe: pd.DataFrame, classifier_col: str, usecols: list[str] = None
    ):
        self.df_all = dataframe
        self.data_columns = usecols if usecols is not None else dataframe.columns
        self.df = dataframe.loc[:, self.data_columns].copy()
        self.class_col = classifier_col

    @property
    def categories(self) -> ArrayLike:
        """Return the unique values on the classifier column."""
        return self.df_all[self.class_col].unique()

    @property
    def category_series(self) -> ArrayLike:
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

    def means_matrix(
        self, as_dataframe: bool = False
    ) -> Union[ArrayLike, pd.DataFrame]:
        """Return the means with shape (M, K) where K is the number of variables
        and M is the number of different values attained by the classifier col."""
        means_df = self.df_all.groupby(self.class_col)[self.data_columns].mean()

        if as_dataframe:
            return means_df

        return means_df.to_numpy()

    def cov_matrix(self, as_dataframe: bool = False) -> Union[ArrayLike, pd.DataFrame]:
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

    def training_distances_from_categories(
        self, as_dataframe: bool = False
    ) -> Union[ArrayLike, pd.DataFrame]:
        """Return the distances for each variable from the centers of the groups
        identified by the classifier column.

        Parameters
        ----------
        as_dataframe : bool, optional
            If True, return the distances in data frame form. Each column matches
            a value from the classifier while each row matches an observation.
            By default False.

        Returns
        -------
        Union[ArrayLike, pd.DataFrame]
            The result has size (N, M) where N is the number of observations while
            M is the number of groups. Each element (x, y) is the distance between
            observation x and the center of group y, where x = [1, ..., N] and
            y = [1, ..., M].
        """
        data = self.df.loc[:, self.data_columns].to_numpy()
        dists = mahanalobis_from_points(data, self.means_matrix(), self.cov_matrix())
        if as_dataframe:
            dataframe = pd.DataFrame(dists)
            dataframe.columns = self.categories
            return dataframe
        return dists

    def training_categories(self) -> pd.Series:
        """Return a pandas series with length N containing the inferred category
        for each observation in the data frame.

        Returns
        -------
        pd.Series
            Series of length N. First element contains the category for the first
            observation and so on.
        """
        distances = self.training_distances_from_categories()
        categories_indexes = pd.Series(np.argmin(distances, axis=1), name="Category")
        print(categories_indexes)
        return categories_indexes.apply(lambda x: self.categories[x])
