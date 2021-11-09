"""Contains functions to compute the Euclidean distance"""
import numpy as np
from preprocessing.scaling import standardize_array


def euclidean_from_center(array: np.ndarray, standardized: bool = False) -> np.ndarray:
    """Return the euclidean distance for each series item from the center of
    the data.

    The euclidean distance is the square root of the sum of the squared
    variable-to-variable distances. If the number of variables is 1, then it
    coincides with the difference under absolute value.

    Parameters
    ----------
    array : ArrayLike
        The array the distance is to be computed on.

        If the array is one dimensional - array.ndim is equal to 1 - then it is
        interpreted as N elements with 1 variable.

        If the array is multi dimensional with size (N, K) then it is interpreted
        as N elements with K variables each.
    standardized : bool, optional
        If True, the array is standardized before the distance is computed. In
        this way the standardized euclidean distance is returned. False by
        default.

    Returns
    -------
    ArrayLike
        An array of size (N, 1) containing the euclidean distances, where
        N is the number of elements in the input array.
    """
    if standardized:
        array = standardize_array(array)
    axis = None if array.ndim == 1 else 0
    means = np.mean(array, axis=axis)
    return euclidean_from_point(array, means)


def euclidean_from_points(
    array: np.ndarray, points: np.ndarray, standardized: bool = False
) -> np.ndarray:
    """Return an array of array made of the euclidean distances between each
    element and each of the specified points.

    Parameters
    ----------
    array : np.array
        The array with size (N, K) the distance is to be computed on. It can be
        either 2-D or 1-D.
    points : np.array
        The points in respect of which the distance is computed. It must be of
        size (M, K). It contains M arrays with K values each.
    standardized : bool, optional
        If True, the array is standardized before the distance is computed. In
        this way the standardized euclidean distance is returned. False by
        default.

    Returns
    -------
    np.array
        The array of distances of dimensions (N, M). For each of the N elements,
        there is a distance from each of the M points.
        Each element (x, y) is the euclidean distance of observation x from
        point y, where x=[1, ..., N] and y=[1, ..., M].
    """
    if standardized:
        array = standardize_array(array)
    number_of_points = points.shape[0]

    # To handle the case where 1-D array is passed
    if array.ndim == 1:
        n_distances = 1  # If array is 1-D just one observation -> 1 distance
    else:
        n_distances = array.shape[0]  # Else: N observations -> N distances

    result = np.zeros((n_distances, number_of_points), dtype=np.float)
    for i in range(number_of_points):
        result[:, i] = euclidean_from_point(array, points[i]).squeeze()
    return result


def euclidean_from_point(
    array: np.ndarray, point: np.ndarray, standardized: bool = False
) -> np.ndarray:
    """Return an array made of the euclidean distances between each element
    and the specified point.

    Parameters
    ----------
    array : np.array
        The array with size (N, K) the distance is to be computed on. It can be
        either 2-D or 1-D.

        If the array is one dimensional - array.ndim is equal to 1 - then N is
        considered len(array) with K = 1.

        If the array is multi dimensional with size (N, K) then it is interpreted
        as N elements with K variables each.
    point : np.array
        The 1-D array identifying the point in respect of which the distance is
        computed. It must have length K.
    standardized : bool, optional
        If True, the array is standardized before the distance is computed. In
        this way the standardized euclidean distance is returned. False by
        default.

    Returns
    -------
    np.ndarray
        The array of distances of dimensions (N, 1).
    """
    if standardized:
        array = standardize_array(array)
    # Case where 1D array and point is just a number
    if (array.ndim == 1) & (point.ndim == 0):
        return np.abs(array - point)
    # Computation is different if the array is uni-dimensional
    axis = None if array.ndim == 1 else 1
    return np.linalg.norm(array - point, axis=axis)
