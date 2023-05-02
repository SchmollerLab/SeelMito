import os
import re

import numpy as np

import skimage.measure

import matplotlib.pyplot as plt

def stain_index(positive_signal, negative_signal):
    # see https://denovosoftware.com/full-access/knowledge-base/stain-index/
    try:
        neg_std = np.std(negative_signal)
        pos_median = np.median(positive_signal)
        neg_median = np.median(negative_signal)
        stain_index = (pos_median-neg_median)/(2*neg_std)
    except Exception as e:
        # Some cells will have zero spots which means no positive signal
        stain_index = np.nan
    return stain_index

def compute_stain_index(intens_data, positive_mask, segm_data, ref_ch_mask=None):
    rp = skimage.measure.regionprops(segm_data)
    stain_indexes = {}
    for obj in rp:
        obj_mask = positive_mask[obj.slice]
        obj_signal = intens_data[obj.slice]
        obj_positive_mask = np.logical_and(obj_mask, obj.image)
        obj_negative_mask = np.logical_and(~obj_mask, obj.image)
        if ref_ch_mask is not None:
            obj_ref_ch_mask = ref_ch_mask[obj.slice]
            obj_negative_mask = np.logical_and(obj_negative_mask, obj_ref_ch_mask)
        obj_positive_signal = obj_signal[obj_positive_mask]
        obj_negative_signal = obj_signal[obj_negative_mask]
        obj_stain_index = stain_index(obj_positive_signal, obj_negative_signal)
        stain_indexes[obj.label] = obj_stain_index
        # fig, ax = plt.subplots(1, 3)
        # zc = int(len(obj_positive_mask)/2)
        # ax[0].imshow(obj_positive_mask[zc])
        # ax[1].imshow(obj_negative_mask[zc])
        # ax[2].imshow(obj.image[zc])
        # plt.show()
    return stain_indexes

def get_local_spot_mask(a, c):
    """_summary_

    Parameters
    ----------
    a : float or int
        Spheroid radius in X and Y direction
    c : float or int
        Spheroid radius in Z direction

    Returns
    -------
    np.ndarray
        Spheroid mask like a structuring element
    """    
    a_int = int(np.ceil(a))
    c_int = int(np.ceil(c))
    # Generate a sparse meshgrid to evaluate 3D spheroid mask
    z, y, x = np.ogrid[-c_int:c_int+1, -a_int:a_int+1, -a_int:a_int+1]
    # 3D spheroid equation
    local_spot_mask = (x**2 + y**2)/(a**2) + z**2/(c**2) <= 1
    return local_spot_mask

def mask_global_to_local(a, c, zyx_center, Z, Y, X):
    a_int = int(np.ceil(a))
    c_int = int(np.ceil(c))
    zc, yc, xc = zyx_center

    z_min = zc-c_int
    z_max = zc+c_int+1
    z_min_crop, z_max_crop = None, None
    y_min_crop, y_max_crop = None, None
    x_min_crop, x_max_crop = None, None

    # Check z size and crop if needed
    if z_min < 0:
        z_min_crop = abs(z_min)
        z_min = 0
    if z_max > Z:
        z_max_crop = Z-z_max
        z_max = Z

    # Check y size and crop if needed
    y_min = yc-a_int
    y_max = yc+a_int+1
    if y_min < 0:
        y_min_crop = abs(y_min)
        y_min = 0
    if y_max > Y:
        y_max_crop = Y-y_max
        y_max = Y

    # Check x size and crop if needed
    x_min = xc-a_int
    x_max = xc+a_int+1
    if x_min < 0:
        x_min_crop = abs(x_min)
        x_min = 0
    if x_max > X:
        x_max_crop = X-x_max
        x_max = X

    slice_global_to_local = (
        slice(z_min,z_max), slice(y_min,y_max), slice(x_min,x_max))
    slice_crop = (
        slice(z_min_crop,z_max_crop), 
        slice(y_min_crop,y_max_crop),
        slice(x_min_crop,x_max_crop)
    )
    return slice_global_to_local, slice_crop

def global_spot_mask(df_inputs, df_spots, spots_mask):
    Z, Y, X = spots_mask.shape

    s = df_inputs.at['ZYX minimum spot volume (vox)', 'Values']
    zyx_spot_radii = re.findall(r'([0-9]+\.[0-9]+)', s)
    a, c = float(zyx_spot_radii[1]), float(zyx_spot_radii[0])
    
    local_spot_mask = get_local_spot_mask(a, c)

    zz, yy, xx = df_spots['z'], df_spots['y'], df_spots['x']
    for zc, yc, xc in zip(zz, yy, xx):
        zyx_center = (zc, yc, xc)
        slice_to_local, slice_crop = mask_global_to_local(
            a, c, zyx_center, Z, Y, X
        )
        _mask = local_spot_mask[slice_crop]
        spots_mask[slice_to_local][_mask] = True
    return spots_mask