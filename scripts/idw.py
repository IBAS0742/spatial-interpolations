#!/usr/bin/env python
# =============================================================================
# Date:     June, 2019
# Author:   Marcelo Villa P.
# Purpose:  creates a raster surface from point data by implementing an Inverse
# #         Distance Weighting (IDW) interpolation.
# =============================================================================
import numpy as np


def euclidean_distance(shape, ind):
    """

    :param shape:
    :param ind:
    :return:
    """
    y, x = shape
    xx, yy = np.meshgrid(np.arange(x), np.arange(y))

    ix = np.array(ind[1]).reshape((-1, 1, 1))
    iy = np.array(ind[0]).reshape((-1, 1, 1))

    dx = np.abs(iy - yy)
    dy = np.abs(ix - xx)

    # return np.sqrt(np.power(dx, 2) + np.power(dy, 2))
    return np.hypot(dx, dy)


def inverse_distance_weighting(z, d, ind, p=1):
    """

    :param z:   1D array with values of interest to interpolate.
    :param d:   3D numpy array with the distance matrices for each z value.
    :param ind: tuple with the indices of z values.
    :param p:   power parameter
    :return:
    """

    np.seterr(divide='ignore', invalid='ignore')

    # reshape z values to a 3D array
    z = z.reshape(-1, 1, 1)

    u = np.sum((z / np.power(d, p)), axis=0)
    w = np.sum((1 / np.power(d, p)), axis=0)
    zi = u / w

    np.seterr(divide='warn', invalid='warn')

    zi[ind] = z.reshape(-1)
    return zi


if __name__ == '__main__':

    arr = np.array([
        [0, 0, 15, 0],
        [4, 0, 0, 0],
        [2, 0, 0, 12],
        [0, 9, 0, 0],
    ])

    shape = arr.shape
    ind = np.nonzero(arr)
    d = euclidean_distance(arr.shape, ind)

    z = arr[ind]
    zi = inverse_distance_weighting(arr[ind], d, ind)
