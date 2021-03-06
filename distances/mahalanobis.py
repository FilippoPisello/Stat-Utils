"""Contains functions to compute the mahanalobis distance."""
import numpy as np


def mahanalobis_from_center(array: np.ndarray, sqrt: bool = False) -> np.ndarray:
    """Return the mahanalobis distance for each series item from the center of
    the data.

    Parameters
    ----------
    array : np.ndarray
        The array the distance is to be computed on. It must be two-dimensional.
    sqrt: bool, optional
        If True, the distance is square rooted. By default is False.

    Returns
    -------
    np.ndarray
        An array of size (N, 1) containing the distances.
    """
    if array.ndim == 1:
        raise ValueError("The array cannot be uni-dimensional")

    array = _make_vertical_if_horizontal(array)
    cov = np.cov(array.transpose())
    means = np.mean(array, axis=0)
    return mahanalobis_from_point(array, means, cov, sqrt)


def mahanalobis_from_points(
    array: np.ndarray, points: np.ndarray, cov: np.ndarray, sqrt: bool = False
) -> np.ndarray:
    """Return an array of array made of the mahanalobis distances between each
    element and each of the specified points.

    Parameters
    ----------
    array : np.array
        The array with size (N, K) the distance is to be computed on. It can be
        either 2-D or 1-D.
    points : np.array
        The points in respect of which the distance is computed. It must be of
        size (M, K). It contains M arrays with K values each.
    cov : np.array
        The covariances conditional to some grouping related to the different
        points. Generally the points are the means for groups of data. The array
        must be of size (M, K, K), where K is the number of variables and M is
        the number of different points.

        Each element (z, i, j) of the matrix represents the covariance between
        the variable i and j, conditional to value z.
    sqrt: bool, optional
        If True, the distance is square rooted. By default is False.

    Returns
    -------
    np.array
        The array of distances of dimensions (N, M). For each of the N elements,
        there is a distance from each of the M points.
        Each element (x, y) is the mahanalobis distance of observation x from
        point y, where x=[1, ..., N] and y=[1, ..., M].
    """
    number_of_points = points.shape[0]

    # To handle the case where 1-D array is passed
    if array.ndim == 1:
        n_distances = 1  # If array is 1-D just one observation -> 1 distance
    else:
        n_distances = array.shape[0]  # Else: N observations -> N distances

    result = np.zeros((n_distances, number_of_points), dtype=float)
    for i in range(number_of_points):
        result[:, i] = mahanalobis_from_point(array, points[i], cov[i], sqrt).squeeze()
    return result


def mahanalobis_from_point(
    array: np.ndarray, points: np.ndarray, cov: np.ndarray, sqrt: bool = False
) -> np.ndarray:
    """Return an array made of the mahanalobis distances between each element
    and the specified point.

    Parameters
    ----------
    array : np.array
        The array with size (N, K) the distance is to be computed on. It can be
        either 2-D or 1-D.
    point : np.array
        The 1-D array identifying the point in respect of which the distance is
        computed. It must have length K.
    cov : np.array
        The matrix containing the covariance between the array's columns. It
        must have dimension (K, K).
    sqrt: bool, optional
        If True, the distance is square rooted. By default is False.

    Returns
    -------
    np.array
        The array of distances of dimensions (N, 1).
    """
    # Case where uni-dimensional array
    if array.squeeze().ndim == 1:
        distances = _unidim_mahanalobis_from_point(array.squeeze(), points, cov)
    # Case where multi-dimensional array
    else:
        distances = _multidim_mahanalobis_from_point(array, points, cov)

    if sqrt:
        return np.sqrt(distances)
    return distances


def _unidim_mahanalobis_from_point(
    array: np.ndarray, points: np.ndarray, cov: np.ndarray
) -> np.ndarray:
    """Return mahanalobis distance for a uni-dimensional array.

    Parameters
    ----------
    array : np.array
        A uni-dimensional array of length K.
    point : np.array
        A uni-dimensional array of length K.
    cov : np.array
        The covariance matrix of dimension (K, K).

    Returns
    -------
    np.array
        The array of distances of dimensions (1, 1)."""
    if len(array) == 1:
        return (array - points) ** 2 / cov  # Univariate

    diff_array_points = np.array(array - points, ndmin=2)  # array 1xK
    return diff_array_points @ np.linalg.inv(cov) @ diff_array_points.transpose()


def _multidim_mahanalobis_from_point(
    array: np.ndarray, points: np.ndarray, cov: np.ndarray
) -> np.ndarray:
    """Return mahanalobis distance for a multi-dimensional array.

    Parameters
    ----------
    array : np.array
        The data array with size (N, K).
    point : np.array
        A uni-dimensional array of length K.
    cov : np.array
        The covariance matrix of dimension (K, K).

    Returns
    -------
    np.array
        The array of distances of dimensions (N, 1)."""
    array = _make_vertical_if_horizontal(array)

    # Get the dimensions ready for matrix multiplication
    diff_array_points = array - points
    inv_covmat = np.linalg.inv(cov)

    return np.array(
        [  # Single loop dimensionality: (1, K) x (K x K) x (1, K) = 1
            [diff_array_points[i, :] @ inv_covmat @ diff_array_points[i, :]]
            for i in range(array.shape[0])
        ]
    )


def _make_vertical_if_horizontal(array: np.ndarray) -> np.ndarray:
    """Return transposed array if array is horizontal."""
    if array.shape[0] < array.shape[1]:
        return array.transpose()
    return array
