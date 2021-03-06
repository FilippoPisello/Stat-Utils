from typing import Any, Callable

import numpy as np


def leave_one_out_validation(
    data: np.ndarray,
    loo_class_callable: Callable[[np.ndarray, int, Any], np.ndarray],
    **methods_parameters: Any
) -> np.ndarray:
    """Return the output of a prediction function by applying the leave one out
    validation technique.

    The leave one out validation takes each observation from a dataset and
    uses all the remaining data to estimate the left out observation. This allows
    to critically assess the gooddes of the algorithm.

    The function is designed to run iteratively the loo_class_method, getting
    a prediction for a single excluded value at the time.

    Parameters
    ----------
    data : np.ndarray
        The array of size (N, K) containing the data to be passed to the
        algorithm. The data is iterated over to pass a single observation at
        the time to the method passed.

    loo_class_callable : Callable[[np.ndarray, int, ...], np.ndarray]
        The callable that returns the prediction for the single excluded
        observation. Its first argument must be single_row, accepting the np
        array of length K representing the single observation. Its second
        argument must be index, integer for the index of the observation.

    **methods_parameters : Any
        Keyword arguments that are directly passed to loo_class_callable.

    Returns
    -------
    np.ndarray
        The array with the predictions with dimension (X, Y). The array is
        squeezed before being returned.
    """
    return np.array(
        [
            loo_class_callable(single_row=obs, index=index, **methods_parameters)
            for index, obs in enumerate(data)
        ]
    ).squeeze()
