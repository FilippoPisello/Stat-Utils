from typing import Union

import numpy as np
import pandas as pd
from distances.distance import mahanalobis_from_point
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

    def categories_training_data(self) -> pd.Series:
        y_hat = np.zeros([self.number_obs, 1])
        Mana_dist = np.zeros([self.number_categories, 1])

        means = self.means_matrix()
        covs = self.cov_matrix()
        for n in range(self.number_obs):
            for k in range(self.number_categories):
                Mana_dist[k, 0] = np.sqrt(
                    mahanalobis_from_point(
                        self.df.iloc[n, :],
                        points=means[k].transpose(),
                        cov=covs[k],
                    )
                )
            y_hat[n, 0] = np.argmin(Mana_dist[:, 0])
        return y_hat
