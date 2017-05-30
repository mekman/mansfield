import numpy as np
import nibabel as nib
from .mri import save_mri
# import nibabel as nib
# from sklearn.cross_validation import KFold
# from sklearn.cross_validation import LeaveOneLabelOut, PredefinedSplit

__all__ = ["searchlight", "get_searchlight_neighbours_matrix"]


def searchlight(X, y, cv, mask_file, fname, radius=6, n_jobs=1,
                estimator='svc', verbose=0):
    """Searchlight with sklearn syntax

    Parameters
    ----------
    X : ndarray, shape(n_samples, n_voxel)
        Voxel data.
    y : ndarray, shape(n_samples)
        Class label for each volume.
    mask_file : string
        Path to mask-file.
    fname : string
        File-name (e.g., 'searchlight_acc_r6.nii').
    radius : integer
        Searchlight radius in mm.
    n_jobs : int
        Number of CPUs to use.
    estimator : Class
        Classifier.
    verbose : int
        Verbose level.

    Returns
    -------
    sl : ndarray, shape(n_voxel)
        Searchlight scores.
    """
    from nilearn.decoding import SearchLight

    ma = nib.load(mask_file).get_data().astype(np.bool)
    mask_img = nib.load(mask_file)

    shape = np.zeros(4)
    shape[:3] = mask_img.get_shape()
    shape[-1] = X.T.shape[-1]

    fmri_img = np.zeros((shape))
    fmri_img[ma] = X.T
    fmri_img = nib.Nifti1Image(fmri_img, mask_img.get_affine())

    searchlight = SearchLight(
        mask_img,
        radius=radius, estimator=estimator, n_jobs=n_jobs,
        verbose=verbose, cv=cv, process_mask_img=mask_img)
    searchlight.fit(fmri_img, y)

    save_mri(searchlight.scores_[ma], mask_file, fname)
    return searchlight.scores_[ma]


def get_searchlight_neighbours_matrix(mask_file, radius=6):
    """

    Parameters
    ----------
    mask_file : string
        Path to mask-file.
    radius : integer
        Searchlight radius in mm.

    Returns
    -------
    A : ndarray, shape(n_voxel, n_voxel)
        Affinity matrix. A[i,j]==1 if voxel_i, voxel_j are neighbours
        and 0 otherwise

    Examples
    --------
    >>> A = get_searchlight_neighbours_matrix('MNI152_2mm_brain_mask.nii')
    >>> A.shape
    """

    # heavily borrowed from nilearn: http://nilearn.github.io/
    from nilearn import image
    from nilearn import masking
    from nilearn.image.resampling import coord_transform
    from distutils.version import LooseVersion
    import sklearn
    from sklearn import neighbors
    from nilearn._utils.niimg_conversions import check_niimg_3d

    def _apply_mask_and_get_affinity(seeds, niimg, radius, allow_overlap,
                                     mask_img=None):
        seeds = list(seeds)
        aff = niimg.get_affine()

        # Compute world coordinates of all in-mask voxels.
        if mask_img is not None:
            mask_img = check_niimg_3d(mask_img)
            mask_img = image.resample_img(mask_img, target_affine=aff,
                                          target_shape=niimg.shape[:3],
                                          interpolation='nearest')
            mask, _ = masking._load_mask_img(mask_img)
            mask_coords = list(zip(*np.where(mask != 0)))

            # X = masking._apply_mask_fmri(niimg, mask_img)
        else:
            mask_coords = list(np.ndindex(niimg.shape[:3]))

        # For each seed, get coordinates of nearest voxel
        nearests = []
        for sx, sy, sz in seeds:
            nearest = np.round(coord_transform(sx, sy, sz, np.linalg.inv(aff)))
            nearest = nearest.astype(int)
            nearest = (nearest[0], nearest[1], nearest[2])
            try:
                nearests.append(mask_coords.index(nearest))
            except ValueError:
                nearests.append(None)

        mask_coords = np.asarray(list(zip(*mask_coords)))
        mask_coords = coord_transform(mask_coords[0], mask_coords[1],
                                      mask_coords[2], aff)
        mask_coords = np.asarray(mask_coords).T

        if (radius is not None and
                LooseVersion(sklearn.__version__) < LooseVersion('0.16')):
            # Fix for scikit learn versions below 0.16. See
            # https://github.com/scikit-learn/scikit-learn/issues/4072
            radius += 1e-6

        clf = neighbors.NearestNeighbors(radius=radius)
        A = clf.fit(mask_coords).radius_neighbors_graph(seeds)
        A = A.tolil()

        for i, nearest in enumerate(nearests):
            if nearest is None:
                continue
            A[i, nearest] = True

        # Include the voxel containing the seed itself if not masked
        mask_coords = mask_coords.astype(int).tolist()
        for i, seed in enumerate(seeds):
            try:
                A[i, mask_coords.index(seed)] = True
            except ValueError:
                # seed is not in the mask
                pass

        if not allow_overlap:
            if np.any(A.sum(axis=0) >= 2):
                raise ValueError('Overlap detected between spheres')

        return A

    process_mask_img = nib.load(mask_file)

    # Compute world coordinates of the seeds
    process_mask, process_mask_affine = masking._load_mask_img(
        process_mask_img)
    process_mask_coords = np.where(process_mask != 0)
    process_mask_coords = coord_transform(
        process_mask_coords[0], process_mask_coords[1],
        process_mask_coords[2], process_mask_affine)
    process_mask_coords = np.asarray(process_mask_coords).T

    A = _apply_mask_and_get_affinity(
        process_mask_coords, process_mask_img, radius, True,
        mask_img=process_mask_img)

    return A  # .toarray().astype('bool')
