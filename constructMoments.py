#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: luca (lucabeale@gmail.com)
"""

import numpy as np, os.path as path
from PiscesA import constant as con
from astropy.io import fits
from matplotlib.path import Path

# # data import
filepath = con.fullpath + 'Tests/'

data = {c: None for c in con.names}
for c in con.names:
    file = path.join(filepath, f'PA_{c}_pbcor.fits')
    data[c] = fits.getdata(file)

# # masking
# keep only within the selected spatial region (determined by the coords of the vertices)
img_shape = {c: data[c].shape for c in con.names}
img_gridPixCoords = {c: np.meshgrid(np.arange(img_shape[c][1]), np.arange(img_shape[c][2])) for c in con.names}
img_allXPixCoords = {c: img_gridPixCoords[c][0].flatten() for c in con.names}
img_allYPixCoords = {c: img_gridPixCoords[c][1].flatten() for c in con.names}
img_pixCoordsList = {c: np.vstack((img_allXPixCoords[c], img_allYPixCoords[c])).T for c in con.names}

img_spatialMask = {c: Path(con.polygon[c]) for c in con.names}
img_booleanMaskListInclusive = {c: img_spatialMask[c].contains_points(img_pixCoordsList[c]) for c in con.names}
img_booleanMaskGridInclusive = {c: img_booleanMaskListInclusive[c].reshape(img_shape[c][1:]) for c in con.names}

data_spatialMask = {c: data[c].copy() for c in con.names}
for c in con.names:
    for plane in range(img_shape[c][0]):
        data_spatialMask[c][plane][~img_booleanMaskGridInclusive[c]] = np.nan

# keep only data brighter than a user-specified threshold (if this holds for at least 3 contiguous pixels)
def thresholdMask(arr, threshold, fill_value=np.nan):
    arr_masked = arr.copy()
    for i, val in enumerate(arr):
        if val >= threshold:
            lo = i-1 if i > 0 else 1  # first entry only compares to the second entry
            hi = i+1 if i < len(arr) - 1 else len(arr) - 2  # last entry only compares to the second last entry
            if not ((arr[lo] >= threshold) or (arr[hi] >= threshold)):
                arr_masked[i] = np.nan
        else:
            arr_masked[i] = np.nan
    return arr_masked

data_thresholdMask = {c: data_spatialMask[c].copy() for c in con.names}
thresholdFunc = lambda idx, coeff: coeff * con.cuberms[idx]
for i, c in enumerate(con.names):
    for ix in range(img_shape[c][1]):
        for iy in range(img_shape[c][2]):
            data_thresholdMask[c][:, ix, iy] = thresholdMask(data_thresholdMask[c][:, ix, iy], thresholdFunc(i, 3))

# # computing moments
# moment 0
mom0 = {c: np.zeros((img_shape[c][1], img_shape[c][2])) for c in con.names}
for i, c in enumerate(con.names):
    for ix in range(img_shape[c][1]):
        for iy in range(img_shape[c][2]):
            mom0[c][ix, iy] = np.nansum(data_thresholdMask[c][:, ix, iy]) * con.cwidth[i]
# moment 1
mom1 = {c: np.zeros((img_shape[c][1], img_shape[c][2])) for c in con.names}
for i, c in enumerate(con.names):
    for ix in range(img_shape[c][1]):
        for iy in range(img_shape[c][2]):
            mom1[c][ix, iy] = np.nansum(data_thresholdMask[c][:, ix, iy] * con.vel[i]) / np.nansum(data_thresholdMask[c][:, ix, iy])

# moment 2
mom2 = {c: np.zeros((img_shape[c][1], img_shape[c][2])) for c in con.names}
for i, c in enumerate(con.names):
    for ix in range(img_shape[c][1]):
        for iy in range(img_shape[c][2]):
            mom2[c][ix, iy] = np.sqrt(np.nansum(data_thresholdMask[c][:, ix, iy] * (con.vel[i]-mom1[c][ix, iy])**2) / np.nansum(data_thresholdMask[c][:, ix, iy]))

# # writing to FITS files (using headers from original moment maps)
for c in con.names:
    # mom0
    hdr0 = fits.getheader(path.join(filepath, f'PA_{c}_mom0.fits'))
    save0 = path.join(filepath, f'PA_{c}_mom0_custom.fits')
    HDU0 = fits.hdu.PrimaryHDU(data=np.array([mom0[c]]), header=hdr0)
    HDU0.writeto(name=save0, overwrite=True)
    # mom1
    hdr1 = fits.getheader(path.join(filepath, f'PA_{c}_mom1.fits'))
    save1 = path.join(filepath, f'PA_{c}_mom1_custom.fits')
    HDU1 = fits.hdu.PrimaryHDU(data=np.array([mom1[c]]), header=hdr1)
    HDU1.writeto(name=save1, overwrite=True)
    # mom2
    hdr2 = fits.getheader(path.join(filepath, f'PA_{c}_mom2.fits'))
    save2 = path.join(filepath, f'PA_{c}_mom2_custom.fits')
    HDU2 = fits.hdu.PrimaryHDU(data=np.array([mom2[c]]), header=hdr2)
    HDU2.writeto(name=save2, overwrite=True)
