#!/usr/bin/env python
# =============================================================================
# Date:     June, 2019
# Author:   Marcelo Villa P.
# Purpose:  creates a raster surface from point data by implementing an Inverse
#           Distance Weighting (IDW) interpolation.
# =============================================================================
import os

import gdalconst
import geopandas as gpd
import numpy as np
import ogr
import pandas as pd

from helper_functions import array_to_tiff


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


def get_indices(x, y, ox, oy, pw, ph):
    """
    Gets the row (i) and column (j) indices in an array for a given set of
    coordinates. Based on https://gis.stackexchange.com/a/92015/86131

    :param x:   array of x coordinates (longitude)
    :param y:   array of y coordinates (latitude)
    :param ox:  raster x origin
    :param oy:  raster y origin
    :param pw:  raster pixel width
    :param ph:  raster pixel height
    :return:    row (i) and column (j) indices
    """

    i = np.floor((oy-y) / ph).astype('int')
    j = np.floor((x-ox) / pw).astype('int')

    return i, j


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
    zi = arr[ind].reshape(-1, 1, 1)

    # compute weights and interpolated values
    w = (1 / np.power(d, p))
    z = np.sum(zi * w, axis=0) / np.sum(w, axis=0)

    # replace masked values with z values
    z[ind] = zi.reshape(-1)

    return np.array(z)


def test1(rows, cols, n, lower_limit=0, upper_limit=100, p=1):
    """

    :param rows:
    :param cols:
    :param n:
    :param lower_limit:
    :param upper_limit:
    :param p:
    :return:
    """

    # create array filled with zeros
    arr = np.zeros((rows, cols), dtype=np.int)

    # create random values and randomly populate the array
    x = np.random.randint(lower_limit, upper_limit, size=n)
    arr.ravel()[np.random.choice(arr.size, n, replace=False)] = x

    # call IDW interpolation
    return inverse_distance_weighting(arr, p)


def test2(fn, field_z, pw, ph, out_fn, field_x=None, field_y=None, sr=None,
          nd_val=-9999, p=1):
    """

    :param fn:
    :param field_z:
    :param pw:
    :param ph:
    :param out_fn:
    :param field_x:
    :param field_y:
    :param sr:
    :param nd_val:
    :return:
    """

    # open file and get array of x coordinates and array of y coordinates
    ext = os.path.splitext(fn)[1]
    if ext == '.shp':
        df = gpd.read_file(fn)
        x = df['geometry'].x
        y = df['geometry'].y
        ds = ogr.Open(fn, 0)
        sr = ds.GetLayer().GetSpatialRef()
        del ds
    elif ext == '.csv':
        df = pd.read_csv(fn)
        x = df[field_x].astype('float')
        y = df[field_y].astype('float')
    else:
        raise Exception('File format not supported.')

    # get target raster x and y origin
    ox = x.min()
    oy = y.max()

    # calculate number of rows and columns
    rows = np.int(np.ceil((oy - y.min()) / ph))
    cols = np.int(np.ceil((x.max() - ox) / pw))

    # create output array
    arr = np.zeros((rows, cols))
    ind = get_indices(x, y, ox, oy, pw, ph)

    # replace values in array and compute interpolated values
    arr[ind] = df[field_z]
    z = inverse_distance_weighting(arr)

    # create geotransform
    gt = (ox, pw, 0, oy, 0, -ph)

    # create array
    array_to_tiff(z, out_fn, sr.ExportToWkt(), gt, gdalconst.GDT_Float32,
                  nd_val)


if __name__ == '__main__':

    # run test 1
    # z = test1(800, 1200, 50, 40, 160)

    # run test 2
    fn = '../data/shp/precipitation.shp'
    out_fn = '../data/tiff/ppt_idw.tif'
    test2(fn, 'annual', 0.0173486, 0.0173486, out_fn, p=2)