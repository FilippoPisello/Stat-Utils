import numpy as np
from numpy.typing import ArrayLike


def mahanalobis_from_center(array: ArrayLike) -> ArrayLike:
    """Return the mahanalobis distance for each series item from the center of
    the data.

    Parameters
    ----------
    array : ArrayLike
        The array the distance is to be computed on. It must be two-dimensional.

    Returns
    -------
    ArrayLike
        An array of size (N, 1) containing the distances.
    """
    if array.ndim == 1:
        raise ValueError("The array cannot be uni-dimensional")

    array = _make_vertical_if_horizontal(array)
    cov = np.cov(array.transpose())
    means = np.mean(array, axis=0)
    return mahanalobis_from_point(array, means, cov)


def mahanalobis_from_point(
    array: ArrayLike, points: ArrayLike, cov: ArrayLike
) -> ArrayLike:
    """Return the mahanalobis distance from a specified point.

    Parameters
    ----------
    array : np.array
        The array the distance is to be computed on. It can be either 2-D or 1-D.
        If 1-D, the points and cov parameters must be provided.
    points : np.array
        The points in respect of which the distance is computed. If array has
        dimension (N, K), points must have dimension (K, M).
    cov : np.array
        If array has dimension (N, K) and points has dimension (K, M), cov must
        have dimension (K, K, M).

    Returns
    -------
    np.array
        The array of distances of dimensions (N, M).
    """
    # Case where uni-dimensional array
    if array.ndim == 1:
        if len(array) == 1:
            return (array - points) ** 2 / cov  # Univariate

        diff_array_points = np.array(array - points, ndmin=2)  # array 1xK
        return diff_array_points @ np.linalg.inv(cov) @ diff_array_points.transpose()

    # Case where multi-dimensional array
    array = _make_vertical_if_horizontal(array)

    # Get the dimensions ready for matrix multiplication
    diff_array_points = array - points
    inv_covmat = np.linalg.inv(cov)

    return np.array(
        [  # Single loop dimensionality: (M, K) x (K x K x M) x (M, K) = M
            [diff_array_points[i, :] @ inv_covmat @ diff_array_points[i, :]]
            for i in range(array.shape[0])
        ]
    )


def _make_vertical_if_horizontal(array: ArrayLike) -> ArrayLike:
    """Return transposed array if array is horizontal."""
    if array.shape[0] < array.shape[1]:
        return array.transpose()
    return array
