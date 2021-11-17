"""Contains detector interface and implementations. Detectors provide criteria
to identify outliers."""
from dataclasses import dataclass
from statistics import NormalDist
from typing import Protocol

import numpy as np
import pandas as pd


class Detector(Protocol):
    """Represent a generic detector with a minimum, maximum and center."""

    def range_min(self) -> float:
        ...

    def range_max(self) -> float:
        ...

    def center(self) -> float:
        ...


@dataclass
class MeanDetector:
    """Represent the outlier detector obtained using the mean and the standard
    deviation.
    """

    series: pd.Series
    quantile: float

    @property
    def width(self) -> float:
        crit_value = NormalDist().inv_cdf((1 + self.quantile) / 2.0)
        return crit_value * self.series.std()

    @property
    def center(self) -> float:
        return self.series.mean()

    @property
    def range_min(self) -> float:
        return self.center - self.width

    @property
    def range_max(self) -> float:
        return self.center + self.width


@dataclass
class MadDetector:
    """Represent the outlier detector obtained using the median absolute
    deviation and the corrected standard deviation.
    """

    series: pd.Series
    quantile: float

    @property
    def width(self) -> float:
        crit_value = NormalDist().inv_cdf((1 + self.quantile) / 2.0)
        median_abs_deviation = np.median(np.abs(self.series - np.median(self.series)))
        return crit_value * 1.48 * median_abs_deviation

    @property
    def center(self) -> float:
        return np.median(self.series)

    @property
    def range_min(self) -> float:
        return self.center - self.width

    @property
    def range_max(self) -> float:
        return self.center + self.width
