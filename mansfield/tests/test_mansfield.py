# from __future__ import absolute_import, division, print_function
# import os.path as op
import numpy as np
import numpy.testing as npt
import nibabel as nib
import mansfield as ma


def test_searchlight():
    # bad boy
    npt.assert_equal((32, 32), (32, 32))


def test_get_searchlight_neighbours_matrix():
    # bad boy
    nii = nib.Nifti1Image(np.ones((1, 1, 2)), affine=np.identity(4))
    nib.save(nii, 'temp_mask.nii')
    A = ma.get_searchlight_neighbours_matrix('temp_mask.nii')

    npt.assert_equal(A.toarray().shape, (2, 2))
