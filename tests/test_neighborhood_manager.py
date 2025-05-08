import pytest

import torch
import numpy as np
import nibabel as nib
from FineTrack.environments.neighborhood_manager import NeighborhoodManager
from FineTrack.utils.utils import SimpleTimer

nib_img_path = '/home/jeremi/Documents/FineTrack/data/datasets/archives/1mm/ismrm2015_fodf.nii.gz'
nifti_image = nib.load(nib_img_path)
image_data = nifti_image.get_fdata().astype(np.float32) # Shape: (1, W, D, 45)

@pytest.fixture
def prepare_multiple_targets_grid():
    radius = 9

    D, H, W, _ = image_data.shape
    coords = [
        [D // 2, H // 2, W // 2],
        [D // 2+5, H // 2+5, W // 2+5],
        [D // 2-5, H // 2-5, W // 2-5],
        [D // 2+5, H // 2-5, W // 2-5],
        [D // 2+5, H // 2+5, W // 2-5]
    ]
    coords = torch.tensor(coords, dtype=torch.float32)
    targets = []
    for coord in coords:
        target = image_data[
            coord[0].long()-radius:coord[0].long()+radius+1,
            coord[1].long()-radius:coord[1].long()+radius+1,
            coord[2].long()-radius:coord[2].long()+radius+1]
        targets.append(target)
    targets = np.array(targets)
    targets = torch.from_numpy(targets).float()
    
    return image_data, targets, coords, radius

def test_multiple_targets_grid(prepare_multiple_targets_grid):
    device="cuda"
    image_data, targets, coords, radius = prepare_multiple_targets_grid

    img_tensor = torch.from_numpy(image_data).to(device)
    img_tensor_2 = img_tensor.clone().to(device)
    coord_tensor = coords.to(device)
    coord_tensor_2 = coord_tensor.clone().to(device)

    # EfficientNeighborhoodManager
    # data_volume, radius, add_neighborhood_vox, flatten, neighborhood_type, device=get_device()
    nm_efficient = NeighborhoodManager(img_tensor, radius, 1, False,
                                       'grid', device=device,
                                       method='efficient')
    
    nm_dwi_ml = NeighborhoodManager(img_tensor_2, radius, 1, False,
                                    'grid', device=device,
                                    method='dwi_ml')


    # EfficientNeighborhoodManager test
    with SimpleTimer() as timer_custom:
        interpolated_img = nm_efficient.get(coord_tensor)
    interpolated_img = interpolated_img.cpu()

    # Compare each target
    for i, target in enumerate(targets):
        difference = torch.mean(torch.abs(target - interpolated_img[i]))
        assert difference < 1e-8

    # DWI-ML style
    with SimpleTimer() as timer_dwi_ml:
        interpolated_img_dwi_ml = nm_dwi_ml.get(coord_tensor_2)
    interpolated_img_dwi_ml = interpolated_img_dwi_ml.cpu()
    
    # Compare each target
    for i, target in enumerate(targets):
        difference = torch.mean(torch.abs(target - interpolated_img_dwi_ml[i]))
        assert difference < 1e-9

    assert timer_custom.interval < timer_dwi_ml.interval, f"Custom interpolation is slower than dwi_ml interpolation: {timer_custom.interval} > {timer_dwi_ml.interval}"

