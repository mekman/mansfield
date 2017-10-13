# from __future__ import absolute_import, division, print_function
# import os.path as op
import os
import numpy as np
import numpy.testing as npt
import nibabel as nib
import mansfield as ma


def test_save_mri():
    # bad boy
    nii = nib.Nifti1Image(np.ones((5, 5, 5)), affine=np.identity(4))
    nib.save(nii, 'temp_mask.nii')
    ma.save_mri(np.ones(5*5*5), 'temp_mask.nii', fname='temp_data.nii')
    data = ma.load_mri('temp_data.nii', 'temp_mask.nii')
    os.remove('temp_mask.nii')
    os.remove('temp_data.nii')

    npt.assert_equal(np.ones((5*5*5, 1)), data)


def test_lowpass_filter():
    # smoke test
    t = np.linspace(-4, 4, 500)
    ts = np.exp(-t**2) + np.random.normal(0, 0.05, t.shape)
    tsf = ma.lowpass_filter(ts, window_length=31, polyorder=2)
    npt.assert_equal(tsf.shape, ts.shape)
