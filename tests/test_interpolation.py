import pytest

import torch
import numpy as np
import nibabel as nib
from dwi_ml.data.processing.volume.interpolation import \
    interpolate_volume_in_neighborhood
from dwi_ml.data.processing.space.neighborhood import \
    unflatten_neighborhood, prepare_neighborhood_vectors

from FineTrack.utils.interpolation import neighborhood_interpolation, calc_neighborhood_grid
from FineTrack.garbage.view_image import display_image
# @pytest.fixture
# def prepare_fodf_volume():
#     fodf_volume = torch.rand(100, 100, 100, 28)
#     return fodf_volume

@pytest.fixture
def prepare_fodf_volume():
    nib_img = nib.load('/home/local/USHERBROOKE/levj1404/Documents/FineTrack/data/datasets/ismrm2015_1mm/fodfs/ismrm2015_fodf.nii.gz')
    img = nib_img.get_fdata()
    img_transposed = img.T
    fodf_volume = torch.tensor(img, dtype=torch.float32)

    # fodf_volume = torch.rand(45, 132, 168, 136)
    # volume = torch.arange(100*100*100*28, dtype=torch.float32).reshape(100, 100, 100, 28)
    return fodf_volume, torch.tensor(img_transposed, dtype=torch.float32), nib_img.affine, img_transposed.shape[0]

@pytest.fixture
def prepare_dwi_ml_neighborhood():
    neighborhood_type = 'grid'
    neighborhood_radius = 100
    neighborhood_resolution = 1.0
    neighborhood_vectors = prepare_neighborhood_vectors(
        neighborhood_type, neighborhood_radius, neighborhood_resolution)
    return neighborhood_vectors, neighborhood_radius

def test_custom_interpolation(prepare_fodf_volume, prepare_dwi_ml_neighborhood):
    # Prepare data
    fodf_volume, vol_transposed, affine, n_coef = prepare_fodf_volume
    neighborhood_vectors, radius = prepare_dwi_ml_neighborhood
    
    grid_side_size = radius*2 + 1

    # Prepare coordinates
    # coords = torch.tensor([[10, 10, 10], [20, 20, 20]], dtype=torch.float32)
    coords = torch.tensor([[61, 66, 69]], dtype=torch.float32)

    # Interpolate with dwi_ml
    target_signal, _ = interpolate_volume_in_neighborhood(
        fodf_volume, coords, neighborhood_vectors, clear_cache=False)
    assert target_signal.shape == (len(coords), grid_side_size*grid_side_size*grid_side_size*n_coef)

    # Unflatten the neighborhood
    target_signal = unflatten_neighborhood(
                    target_signal, neighborhood_vectors, 'grid',
                    radius, 1)
    target_signal = target_signal.permute(0, 4, 3, 2, 1)

    display_image(nib.Nifti1Image(target_signal[0].cpu().numpy(), affine), default_slice=radius)

    assert target_signal.shape == (len(coords), n_coef, grid_side_size, grid_side_size, grid_side_size)

    # Now, interpolate with custom interpolation technique
    grid = calc_neighborhood_grid(radius, device=fodf_volume.device, resolution=1.0)
    signal = neighborhood_interpolation(vol_transposed, coords, grid)

    display_image(nib.Nifti1Image(signal[0].cpu().numpy(), affine), default_slice=radius)
    assert signal.shape == (len(coords), n_coef, grid_side_size, grid_side_size, grid_side_size)

    difference = torch.abs(target_signal - signal)

    point1 = target_signal[0, :, 9, 9, 9]
    point2 = signal[0, :, 9, 9, 9]

    assert torch.all(difference < 1e-6)

    
