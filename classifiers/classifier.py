from typing import Union

import numpy as np
import pandas as pd


class Classifier:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        classifier_col: str,
        usecols: list[str] = None,
    ):
        self.df_all = dataframe
        self.class_col = classifier_col
        self.data_columns = self._identify_data_columns(usecols)
        # Predictors and outcomes as numpy arrays
        self.data = self.df_all.loc[:, self.data_columns].to_numpy()
        self.outcomes = self._preprocess_outcomes()

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

    def _preprocess_outcomes(self) -> np.ndarray:
        return self.df_all[self.class_col].to_numpy().reshape(self.data.shape[0], 1)

    @property
    def categories(self) -> np.ndarray:
        """Return the unique values on the classifier column."""
        cats = self.df_all[self.class_col].unique()
        cats.sort()
        return cats

    @property
    def category_series(self) -> np.ndarray:
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
