import numpy as np
import nibabel as nib


def numpy_to_nii(name, vox, affine):
    assert isinstance(vox, np.ndarray), \
        'Type error, the type of vox is expected to be np.ndarray, ' + \
        f'but {type(vox)} got.'

    if 'nii' not in name:
        name = f'{name}.nii.gz'

    nii = nib.Nifti1Image(vox, affine)
    nib.save(nii, name)
