from typing import Union

import numpy as np


def hamming_from_elements(
    array: np.ndarray, elements: np.ndarray, weight: Union[float, int, np.ndarray] = 1
) -> np.ndarray:
    """Return an array of array made of the hamming distances between each
    element and each of the specified points.

    The hamming distance allows to quantify the distance between non-numerical
    elements. Given two elements with K features, the value weight is added
    to the hamming distance for each non-identical feature.

    Parameters
    ----------
    array : np.array
        The array with size (N, K) the distance is to be computed on. It can be
        either 2-D or 1-D.

        If the array is one dimensional - array.ndim is equal to 1 - then N is
        considered len(array) with K = 1.

        If the array is multi dimensional with size (N, K) then it is interpreted
        as N elements with K variables each.
    element : np.ndarray
        The elements in respect of which the distance is computed. It must be of
        size (M, K). It contains M arrays with K values each.
    weight : Union[float, int, np.ndarray], optional
        The value for which the distances should be multiplied by.

        If int or float, every variable is assigned the same provided fixed
        weight. Every non-match will have a value equal to weight.

        If np.ndarray it represents the value for each column. In this case
        it must be of length K.

    Returns
    -------
    np.ndarray
        The array of distances of dimensions (N, M). For each of the N elements,
        there is a distance from each of the M points.
        Each element (x, y) is the hamming distance of observation x from
        item y, where x=[1, ..., N] and y=[1, ..., M].
    """
    number_of_points = elements.shape[0]

    # To handle the case where 1-D array is passed
    if array.ndim == 1:
        n_distances = 1  # If array is 1-D just one observation -> 1 distance
    else:
        n_distances = array.shape[0]  # Else: N observations -> N distances

    result = np.zeros((n_distances, number_of_points), dtype=np.float)
    for i in range(number_of_points):
        result[:, i] = hamming_from_element(array, elements[i], weight).squeeze()
    return result


def hamming_from_element(
    array: np.ndarray, element: np.ndarray, weight: Union[float, int, np.ndarray] = 1
) -> np.ndarray:
    """Return an array made of the hamming distances between each element
    and the specified comparison item.

    The hamming distance allows to quantify the distance between non-numerical
    elements. Given two elements with K features, the value weight is added
    to the hamming distance for each non-identical feature.

    Parameters
    ----------
    array : np.array
        The array with size (N, K) the distance is to be computed on. It can be
        either 2-D or 1-D.

        If the array is one dimensional - array.ndim is equal to 1 - then N is
        considered len(array) with K = 1.

        If the array is multi dimensional with size (N, K) then it is interpreted
        as N elements with K variables each.
    element : np.ndarray
        The 1-D array identifying the item in respect of which the distance is
        computed. It must have length K.
    weight : Union[float, int, np.ndarray], optional
        The value for which the distances should be multiplied by.

        If int or float, every variable is assigned the same provided fixed
        weight. Every non-match will have a value equal to weight.

        If np.ndarray it represents the value for each column. In this case
        it must be of length K.

    Returns
    -------
    np.ndarray
        The array of distances of dimensions (N, 1).
    """
    # Case where 1D array and element is a single item
    if (array.ndim == 1) & (element.ndim == 0):
        return (array != element).astype(int) * weight
    # Computation is different if the array is uni-dimensional
    axis = None if array.ndim == 1 else 1

    matches_array = (array != element) * weight
    return matches_array.sum(axis=axis)
