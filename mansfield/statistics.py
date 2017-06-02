#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import norm

__all__ = ['pval_to_zscore']


def pval_to_zscore(data, two_sided=True, inv=False):
    """convert p-values to z-scores (and the inverse)

    Parameters
    -----------
    data : ndarray, shape(n, )
        Data values
    two_sided : boolean
        Values from two-sided test (default=True).
    inv : boolean
        Inverse transformation: convert z-scores to p-values (default=False).

    Returns
    -------
    v : ndarray, shape(n, )
        Transformed data values.

    Examples
    --------
    >>> p = np.array([0.09, 0.001, 0.05])
    >>> z = pval_2zscore(p)
    >>> print z
    [ 1.69539771 3.29052673 1.95996398]
    >>> print pval_2zscore(z, inv=True)
    [ 0.09 0.001 0.05 ]
    """

    if two_sided:
        mul = 2.
    else:
        mul = 1.

    if inv:
        # zscore --> pval
        v = norm.cdf(-np.abs(data)) * mul

        # TODO better use survival function?
        # v  = norm.sf(data) * mul
    else:
        # pval --> zscore
        v = np.abs(norm.ppf(data / mul))

    return v
