import numpy as np
import matplotlib.pyplot as plt
from dipy.data import get_fnames
from dipy.io.image import load_nifti_data
from dipy.segment.tissue import TissueClassifierHMRF
import time
import argparse
import os
import nibabel as nib


T1_path = "/home/local/USHERBROOKE/levj1404/Documents/TrackToLearn/data/datasets/ismrm2015_2mm/anat/ismrm2015_T1.nii.gz"
out_dir = "/home/local/USHERBROOKE/levj1404/Documents/TrackToLearn/dipy_segment_tissues_b03"

os.makedirs(out_dir, exist_ok=True)
print("Output directory: %s" % out_dir)

t1_nifti = nib.load(T1_path)
t1 = t1_nifti.get_fdata()
print("t1.shape (%d, %d, %d)" % t1.shape)

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
img_ax = np.rot90(t1[:, :, t1.shape[2] // 2])
ax.imshow(img_ax, cmap='gray')
ax.set_title('axial')
ax.axis('off')

ax = fig.add_subplot(1, 2, 2)
img_ax = np.rot90(t1[:, t1.shape[1] // 2, :])
ax.imshow(img_ax, cmap='gray')
ax.set_title('coronal')
ax.axis('off')

plt.savefig(os.path.join(out_dir, 't1_ref.png'))

# Now we segment the tissues
print("Segmenting the tissues...")
nclass = 3
beta = 0.3

start_time = time.time()
hmrf = TissueClassifierHMRF()
initial_segmentation, final_segmentation, PVE = hmrf.classify(t1, nclass, beta)
end_time = time.time()
print("Segmentation time: %f seconds" % (end_time - start_time))

# Plot the probability results
fig = plt.figure()
ax = fig.add_subplot(1, 3, 1)
csf_data = np.rot90(PVE[:, :, t1.shape[2] // 2, 0])
ax.imshow(csf_data, cmap='gray')
ax.set_title('CSF')
ax.axis('off')

ax = fig.add_subplot(1, 3, 2)
gm_data = np.rot90(PVE[:, :, t1.shape[2] // 2, 1])
ax.imshow(gm_data, cmap='gray')
ax.set_title('GM')
ax.axis('off')

ax = fig.add_subplot(1, 3, 3)
wm_data = np.rot90(PVE[:, :, t1.shape[2] // 2, 2])
ax.imshow(wm_data, cmap='gray')
ax.set_title('WM')
ax.axis('off')

plt.savefig(os.path.join(out_dir, 't1_segmentation_probs.png'))

# Since PVE is the probability of each tissue, we can take the argmax to get the masks
csf_mask = (np.argmax(PVE, axis=3) == 0).astype(np.uint8)
gm_mask = (np.argmax(PVE, axis=3) == 1).astype(np.uint8)
wm_mask = (np.argmax(PVE, axis=3) == 2).astype(np.uint8)

# Plot the results
fig = plt.figure()
ax = fig.add_subplot(1, 3, 1)
img_data = np.rot90(csf_mask[:, :, csf_mask.shape[2] // 2])
ax.imshow(img_data, cmap='gray')
ax.set_title('CSF mask')
ax.axis('off')

ax = fig.add_subplot(1, 3, 2)
img_data = np.rot90(gm_mask[:, :, gm_mask.shape[2] // 2])
ax.imshow(img_data, cmap='gray')
ax.set_title('GM mask')
ax.axis('off')

ax = fig.add_subplot(1, 3, 3)
img_data = np.rot90(wm_mask[:, :, wm_mask.shape[2] // 2])
ax.imshow(img_data, cmap='gray')
ax.set_title('WM mask')
ax.axis('off')

plt.savefig(os.path.join(out_dir, 't1_masks.png'))

# Save the masks
print("Saving the resulting nifti images...")
csf_img = nib.Nifti1Image(csf_mask, t1_nifti.affine)
gm_img = nib.Nifti1Image(gm_mask, t1_nifti.affine)
wm_img = nib.Nifti1Image(wm_mask, t1_nifti.affine)

nib.save(csf_img, os.path.join(out_dir, 't1_csf.nii.gz'))
nib.save(gm_img, os.path.join(out_dir, 't1_gm.nii.gz'))
nib.save(wm_img, os.path.join(out_dir, 't1_wm.nii.gz'))

