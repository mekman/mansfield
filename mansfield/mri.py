#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal
import nibabel as nib

__all__ = ['load_mri', 'save_mri', 'lowpass_filter']


def load_mri(func, mask):
    """load MRI voxel data

    The data is converted into a 2D (n_voxel, n_tps) array.

    Parameters
    ----------
    func : string
        Path to imaging data (e.g. nifti).
    mask : string
        Path to binary mask (e.g. nifti) that defines brain regions. Values > 0
        are regarded as brain tissue.

    Returns
    -------
    ts : ndarray, shape(n_voxel, n_tps)
        Timeseries information in a 2D array.

    See Also
    --------
    save_mri: save MRI voxel data to disk.

    Examples
    --------
    >>> ts = load_mri(func='localizer.nii.gz', mask='V1_mask.nii.gz')
    """

    # load mask data
    m = nib.load(mask).get_data()

    # load func data
    d = nib.load(func).get_data()

    # mask the data
    func_data = d[m != 0]
    # nib.load(func).get_data()[nib.load(mask).get_data()!=0]

    del d

    return func_data


def save_mri(data, mask, fname=None):
    """save MRI voxel data

    Parameters
    ----------
    data : ndarray, shape(n_voxel,) **or** shape(n_voxel, n_tps)
       Voxel data to save to disk.
    mask : string
        Path to binary mask (e.g. nifti) that defines brain regions. Values > 0
        are regarded as brain tissue.
    fname : string
        Filename.

    Examples
    --------
    >>> ts = load_mri(func='localizer.nii.gz', mask='V1_mask.nii.gz')
    >>> ts = ts + 1. # some operation
    >>> save_mri(ts, 'V1_mask.nii.gz', 'localizer_plus_one.nii.gz')
    """
    # load mask data
    f = nib.load(mask)
    m = f.get_data()
    aff = f.get_affine()

    s = m.shape
    if len(data.shape) == 2:
        n_tps = data.shape[1]
    else:
        n_tps = 1
        data = data[:, np.newaxis]

    res = np.zeros((s[0], s[1], s[2], n_tps))  # + time
    res[m != 0] = data

    # save to disk
    if fname is not None:
        nib.save(nib.Nifti1Image(res, aff), fname)


def lowpass_filter(ts, window_length=7, polyorder=3):
    r"""Smooth data with a Savitzky-Golay filter.

    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.

    Parameters
    ----------
    ts : array_like, shape (n_tps,) or (n_voxel, n_tps)
        Time-series data.
    window_length : int
        Length of the window. Must be an odd integer number (default=7).
    polyorder : int
        Order of the polynomial used in the filtering.
        Must be less then `window_length` - 1.

    Returns
    -------
    tsf : ndarray, shape (n_tps,) or (n_voxel, n_tps)
        Filtered time-series data .

    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.

    Examples
    --------
    >>> t = np.linspace(-4, 4, 500)
    >>> ts = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    >>> tsf = lowpass_filter(ts, window_length=31, polyorder=4)

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(t, ts, label='Noisy signal')
    >>> plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    >>> plt.plot(t, tsf, 'r', label='Filtered signal')
    >>> plt.legend()

    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
           Data by Simplified Least Squares Procedures. Analytical Chemistry,
           1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
           W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
           Cambridge University Press ISBN-13: 9780521880688
    """

    # try:
    #     window_size = np.abs(np.int(window_size))
    #     order = np.abs(np.int(order))
    # except ValueError, msg:
    #     raise ValueError("window_size and order have to be of type int")
    # if window_size % 2 != 1 or window_size < 1:
    #     raise TypeError("window_size size must be a positive odd number")
    # if window_size < order + 2:
    #     raise TypeError("window_size is too small for the polynomials order")
    #
    # order_range = range(order+1)
    # half_window = (window_size -1) // 2
    #
    # # precompute coefficients
    # b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    # m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    #
    # # pad the signal at the extremes with
    # # values taken from the signal itself
    # firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    # lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    # y = np.concatenate((firstvals, y, lastvals))
    #
    # return np.convolve( m[::-1], y, mode='valid')
    # # # replaced after scipy 0.14 release with this
    return signal.savgol_filter(ts, window_length, polyorder, deriv=0.0,
                                delta=1.0, axis=-1, mode='interp', cval=0.0)
