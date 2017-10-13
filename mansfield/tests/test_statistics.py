# from __future__ import absolute_import, division, print_function
# import os.path as op
import numpy as np
import numpy.testing as npt
import mansfield as ma


def test_pval_to_zscore():
    p = np.array([0.001, 0.05])
    z = ma.pval_to_zscore(p)

    values = [3.29052673, 1.95996398]
    npt.assert_almost_equal(values, z, decimal=5)

    z_inv = ma.pval_to_zscore(z, inv=True)
    npt.assert_almost_equal(p, z_inv, decimal=5)
