import nibabel as nib
import numpy as np

nifti_path = (
    "data/data-multi-subject/"
    "sub-brnoCeitec01/anat/"
    "sub-brnoCeitec01_T2w.nii.gz"
)

img = nib.load(nifti_path)

data = img.get_fdata()

print("Shape:", data.shape)
print("Data type:", data.dtype)
print("Min:", np.min(data))
print("Max:", np.max(data))

print("Voxel size:", img.header.get_zooms())