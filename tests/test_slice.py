import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

path = "data/data-multi-subject/sub-brnoCeitec01/anat/sub-brnoCeitec01_T2w.nii.gz"

img = nib.load(path)
data = img.get_fdata()

# middle slice
z = data.shape[2] // 2
slice_img = data[:, :, z]

plt.imshow(slice_img.T, cmap="gray", origin="lower")
plt.title(f"Spine MRI Slice {z}")
plt.axis("off")
plt.show()