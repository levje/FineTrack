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

nib_img_path = '/home/local/USHERBROOKE/levj1404/Documents/FineTrack/data/datasets/ismrm2015_1mm/fodfs/ismrm2015_fodf.nii.gz'

@pytest.fixture
def prepare_fodf_volume_and_target():
    nib_img = nib.load(nib_img_path)
    img = nib_img.get_fdata()
    fodf_volume = torch.tensor(img, dtype=torch.float32)

    coord = torch.tensor([img.shape[0]//2, img.shape[1]//2, img.shape[2]//2], dtype=torch.long)
    radius = 10

    # Crop the fodf_volume to the neighborhood
    target = fodf_volume[
        coord[0].long()-radius:coord[0].long()+radius+1,
        coord[1].long()-radius:coord[1].long()+radius+1,
        coord[2].long()-radius:coord[2].long()+radius+1]

    return fodf_volume, target, coord.float(), radius

@pytest.fixture
def prepare_fodf_volume_and_targets():
    nib_img = nib.load(nib_img_path)
    img = nib_img.get_fdata()
    fodf_volume = torch.tensor(img, dtype=torch.float32)

    coords = torch.tensor([
        [img.shape[0]//2, img.shape[1]//2, img.shape[2]//2],
        [img.shape[0]//2+5, img.shape[1]//2+5, img.shape[2]//2+5],
        [img.shape[0]//2-5, img.shape[1]//2-5, img.shape[2]//2-5],
        [img.shape[0]//2+5, img.shape[1]//2-5, img.shape[2]//2-5],
        [img.shape[0]//2+5, img.shape[1]//2+5, img.shape[2]//2-5]
    ], dtype=torch.long)
    radius = 10

    targets = []
    for coord in coords:
        target = fodf_volume[
            coord[0].long()-radius:coord[0].long()+radius+1,
            coord[1].long()-radius:coord[1].long()+radius+1,
            coord[2].long()-radius:coord[2].long()+radius+1]
        targets.append(target)

    return fodf_volume, targets, coords.float(), radius

def test_dwiml_interpolation_crop(prepare_fodf_volume_and_target):
    fodf_volume, target, coord, radius = prepare_fodf_volume_and_target

    n_coef = fodf_volume.shape[-1]
    neighborhood_type = 'grid'
    neighborhood_resolution = 1.0
    neighborhood_vectors = prepare_neighborhood_vectors(
        neighborhood_type, radius, neighborhood_resolution)
    
    grid_side_size = radius*2 + 1

    # Interpolate with dwi_ml
    signal, _ = interpolate_volume_in_neighborhood(fodf_volume, coord.unsqueeze(0), neighborhood_vectors)
    assert signal.shape == (1, grid_side_size*grid_side_size*grid_side_size*n_coef)

    # Unflatten the neighborhood
    signal = unflatten_neighborhood(
                    signal, neighborhood_vectors, 'grid',
                    radius, neighborhood_resolution)
    assert signal.shape == (1, grid_side_size, grid_side_size, grid_side_size, n_coef)

    difference = torch.abs(target - signal[0])
    error_ratio = difference.sum() / (difference>=0).sum()
    assert error_ratio < 0.03


def test_custom_interpolation_crop(prepare_fodf_volume_and_target):
    # Prepare data
    fodf_volume, target, coord, radius = prepare_fodf_volume_and_target

    grid = calc_neighborhood_grid(radius, device=fodf_volume.device, resolution=1.0)

    # Interpolate with custom interpolation technique
    signal = neighborhood_interpolation(fodf_volume, coord.unsqueeze(0), grid)

    difference = torch.abs(target - signal[0])
    error_ratio = difference.sum() / (difference>=0).sum()
    print("error_ratio:", error_ratio)
    assert error_ratio < 0.04  # TODO: We need to reduce the error ratio here.

def test_custom_interpolation_mutiple_coordinates(prepare_fodf_volume_and_targets):
    fodf_volume, targets, coords, radius = prepare_fodf_volume_and_targets

    grid = calc_neighborhood_grid(radius, device=fodf_volume.device, resolution=1.0)

    # Interpolate with custom interpolation technique
    signal = neighborhood_interpolation(fodf_volume, coords, grid)
    
    differences = []
    for i, target in enumerate(targets):
        difference = torch.abs(target - signal[i])
        error_ratio = difference.sum() / (difference>=0).sum()
        differences.append(error_ratio)
    
    for error_ratio in differences:
        assert error_ratio < 0.04 # TODO: We need to reduce the error ratio here.

def test_custom_unflattening():
    # Unflatten the signal into (N, W, H, D, C) shape
    other_signal = unflatten_neighborhood(
        signal, self.neighborhood_directions, self.neighborhood_type,
        self.radius, self.add_neighborhood_vox)
    other_signal_flat = other_signal.view(other_signal.shape[0], -1, 28)
    signal = self._unflatten_neighborhood(signal)

    signal = signal.cpu()
    other_signal = other_signal.cpu()
    # Test 1 (signal of shape (N, W, H, D, C))
    # if (torch.abs(signal - other_signal) < 1e-6).all():
    #     raise ValueError('Found it!')
    difference = torch.abs(signal - other_signal)
    
    # diff_flat = torch.abs(flat - other_signal_flat)[0] # First point only
    # eq_zero = diff_flat == 0
    # all_eq_zero = torch.all(eq_zero, dim=1)
    # idx_where_zero = torch.arange(len(diff_flat))[all_eq_zero.cpu()]
    # coords_where_good = self.neighborhood_directions[idx_where_zero]
    
    # raise ValueError('Unflattening is not working correctly.')