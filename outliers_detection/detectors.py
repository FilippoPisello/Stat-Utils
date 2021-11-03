from dataclasses import dataclass
from typing import Protocol, Union

import pandas as pd
from scipy.stats import median_abs_deviation


class Detector(Protocol):
    """Represent a generic detector with a minimum, maximum and center."""

    def range_min(self):
        ...

    def range_max(self):
        ...

    def center(self):
        ...


@dataclass
class MeanDetector:
    """Represent the outlier detector obtained using the mean and the standard
    deviation.

    Returns
    -------
    [type]
        [description]
    """

    series: pd.Series
    factor: Union[int, float]

    @property
    def width(self):
        return self.factor * self.series.std()

    @property
    def center(self):
        return self.series.mean()

    @property
    def range_min(self):
        return self.series - self.width

    @property
    def range_max(self):
        return self.series + self.width


@dataclass
class MadDetector:
    """Represent the outlier detector obtained using the median absolute
    deviation and the corrected standard deviation.
    """

    series: pd.Series
    factor: Union[int, float]

    @property
    def width(self):
        return self.factor * 1.48 * self.series.std()

    @property
    def center(self):
        return median_abs_deviation(self.series)

    @property
    def range_min(self):
        return self.center - self.width

    @property
    def range_max(self):
        return self.center + self.width
