#!/usr/bin/env python
# =============================================================================
# Date:     June, 2019
# Author:   Marcelo Villa P.
# Purpose:  creates a raster surface from point data by implementing an Inverse
#           Distance Weighting (IDW) interpolation.
# =============================================================================
import numpy as np


def euclidean_distance(shape, ind):
    """

    :param shape:
    :param ind:
    :return:
    """
    # create meshgrid
    y, x = shape
    xx, yy = np.meshgrid(np.arange(x), np.arange(y))

    # reshape indices
    ix = ind[1].reshape((-1, 1, 1))
    iy = ind[0].reshape((-1, 1, 1))

    # compute legs
    dx = np.abs(iy - yy)
    dy = np.abs(ix - xx)

    return np.hypot(dx, dy)


def inverse_distance_weighting(arr, p=1):
    """

    :param arr:
    :param ind:
    :param p:
    :return:
    """

    # get indices of z values and compute distance matrices
    ind = np.nonzero(arr)
    d = euclidean_distance(arr.shape, ind)

    # mask distances to avoid zeros
    mask = np.any((d == 0), axis=0)
    d = np.ma.array(d, mask=np.broadcast_to(mask, d.shape))

    # get z values and reshape z values to a 3D array
    zi = arr[ind]
    zi = zi.reshape(-1, 1, 1)

    # compute weights and interpolated values
    w = (1 / np.power(d, p))
    z = np.sum(zi * w, axis=0) / np.sum(w, axis=0)

    # replace masked values with z values
    z[ind] = zi.reshape(-1)

    return np.array(z)


if __name__ == '__main__':

    # create array filled with zeros
    arr = np.zeros((1000, 1000), dtype=np.int)

    # create random values and randomly populate the array
    n = 100
    x = np.random.randint(40, 80, size=n)
    arr.ravel()[np.random.choice(arr.size, n, replace=False)] = x

    # call IDW interpolation
    z = inverse_distance_weighting(arr, 2)
